"""
DeepFingerprinting ISCX Dataset Processor

Converts ISCX-VPN/Tor datasets to DeepFingerprinting format (direction sequences ±1).
Uses dpkt for fast PCAP parsing with multi-processing support.
Supports multiple link layer types (Ethernet, Linux SLL, Raw IP, etc.)

Output: NPZ file with direction sequences (±1) and labels

Usage:
    1. Modify the configuration below (DATASET, LABEL_MAP_PATH, etc.)
    2. Run: python iscx_processor.py
"""

import os
import csv
import json
import socket
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False
    print("Error: dpkt is required. Install with: pip install dpkt")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# Configuration - Modify these settings
# =============================================================================
# Input paths - modify these to match your dataset location
LABEL_MAP_PATH = "/home/pcz/DL/ML_DL/public_dataset/USTC-TFC2016/label_map.csv"
VOCAB_PATH = "/home/pcz/DL/ML_DL/public_dataset/USTC-TFC2016/service_vocab.csv"

# Output path
OUTPUT_DIR = './data/ustc'

# Dataset name (for metadata)
DATASET_NAME = 'USTC-TFC'

# Flow extraction parameters (Paper Section 4: Data Collection)
MIN_PACKETS = 50          # Minimum packets per flow (Paper: "too short – less than 50 packets")
MAX_PACKETS = 5000        # Maximum packets per flow (Paper Section 5.1: "5,000 cells provide the best results")
FLOW_TIMEOUT = 60.0       # Flow timeout in seconds (for splitting long connections)
PAYLOAD_ONLY = True       # Only count packets with payload (skip SYN/ACK/FIN without data)

# Random seed
RANDOM_SEED = 42

# Multi-process
NUM_WORKERS = 8


# =============================================================================
# Data Structure
# =============================================================================

@dataclass
class FlowData:
    """Extracted flow data."""
    directions: List[int]  # Direction sequence (±1)


# =============================================================================
# Link Layer Handling
# =============================================================================

def extract_ip_from_buf(buf: bytes, datalink: int):
    """
    Extract IP packet from buffer based on datalink type.

    Args:
        buf: Raw packet bytes
        datalink: PCAP datalink type

    Returns:
        IP packet object or None
    """
    try:
        # DLT_EN10MB = 1 (Ethernet)
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                return eth.data
            elif isinstance(eth.data, dpkt.ip6.IP6):
                return eth.data  # Support IPv6

        # DLT_RAW = 101 (Raw IP)
        elif datalink == 101:
            if len(buf) > 0:
                version = (buf[0] >> 4) & 0xF
                if version == 4:
                    return dpkt.ip.IP(buf)
                elif version == 6:
                    return dpkt.ip6.IP6(buf)

        # DLT_LINUX_SLL = 113 (Linux cooked capture)
        elif datalink == 113:
            if len(buf) >= 16:
                proto = (buf[14] << 8) | buf[15]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[16:])
                elif proto == 0x86DD:  # IPv6
                    return dpkt.ip6.IP6(buf[16:])

        # DLT_LINUX_SLL2 = 276 (Linux cooked capture v2)
        elif datalink == 276:
            if len(buf) >= 20:
                proto = (buf[0] << 8) | buf[1]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[20:])
                elif proto == 0x86DD:  # IPv6
                    return dpkt.ip6.IP6(buf[20:])

        # DLT_NULL = 0 (BSD loopback)
        elif datalink == 0:
            if len(buf) >= 4:
                family = buf[0] if buf[0] != 0 else buf[3]
                if family == 2:  # AF_INET
                    return dpkt.ip.IP(buf[4:])
                elif family in (24, 28, 30):  # AF_INET6 variants
                    return dpkt.ip6.IP6(buf[4:])

    except Exception:
        pass

    return None


def get_ip_addr(ip) -> Tuple[Optional[str], Optional[str]]:
    """Extract IP address string from dpkt IP object."""
    if isinstance(ip, dpkt.ip.IP):
        return socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
    elif isinstance(ip, dpkt.ip6.IP6):
        return socket.inet_ntop(socket.AF_INET6, ip.src), socket.inet_ntop(socket.AF_INET6, ip.dst)
    return None, None


# =============================================================================
# PCAP Processing
# =============================================================================

def extract_flows_from_pcap(pcap_path: str, return_stats: bool = False):
    """
    Extract flows from PCAP using dpkt.

    Returns list of FlowData objects with direction sequences (±1).
    If return_stats=True, also returns statistics dict.

    Note: Only packets with payload are counted (consistent with unified_novpn_processor.py)
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    stats = {'total_packets': 0, 'ip_packets': 0, 'tcp_udp_packets': 0,
             'payload_packets': 0, 'flows_before_filter': 0, 'flows_after_filter': 0,
             'datalink': 0}

    # Group packets by flow: flow_key -> [(time, src_ip), ...]
    flows = defaultdict(list)

    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            # Get datalink type
            datalink = pcap.datalink()
            stats['datalink'] = datalink

            for ts, buf in pcap:
                stats['total_packets'] += 1

                # Extract IP packet based on datalink type
                ip = extract_ip_from_buf(buf, datalink)
                if ip is None:
                    continue

                stats['ip_packets'] += 1
                src_ip, dst_ip = get_ip_addr(ip)
                if src_ip is None:
                    continue

                # Get transport layer info and payload
                payload = b''
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    sport, dport, proto = tcp.sport, tcp.dport, 6
                    payload = bytes(tcp.data) if tcp.data else b''
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    sport, dport, proto = udp.sport, udp.dport, 17
                    payload = bytes(udp.data) if udp.data else b''
                else:
                    continue

                stats['tcp_udp_packets'] += 1

                # Skip packets without payload (consistent with unified_novpn_processor)
                if PAYLOAD_ONLY and len(payload) == 0:
                    continue

                stats['payload_packets'] += 1

                # Normalize flow key (smaller tuple first for bidirectional flow)
                if (src_ip, sport) < (dst_ip, dport):
                    flow_key = (src_ip, dst_ip, sport, dport, proto)
                else:
                    flow_key = (dst_ip, src_ip, dport, sport, proto)

                flows[flow_key].append((float(ts), src_ip))

    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        if return_stats:
            return [], stats
        return []

    stats['flows_before_filter'] = len(flows)

    # Convert to FlowData objects with direction sequences
    result = []
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) < MIN_PACKETS:
            continue

        flow_packets.sort(key=lambda x: x[0])

        # Determine client IP (first packet's source)
        client_ip = flow_packets[0][1]

        directions = []
        last_time = flow_packets[0][0]

        for pkt_time, src_ip in flow_packets:
            # Split by timeout
            if pkt_time - last_time > FLOW_TIMEOUT and len(directions) >= MIN_PACKETS:
                result.append(FlowData(directions=directions[:MAX_PACKETS]))
                directions = []
                client_ip = src_ip

            # Direction: +1 = outgoing (from client), -1 = incoming
            if src_ip == client_ip:
                directions.append(1)
            else:
                directions.append(-1)

            last_time = pkt_time

        if len(directions) >= MIN_PACKETS:
            result.append(FlowData(directions=directions[:MAX_PACKETS]))

    stats['flows_after_filter'] = len(result)
    del flows

    if return_stats:
        return result, stats
    return result


def process_single_pcap(args: Tuple[str, int]) -> Tuple[int, List[List[int]], Dict]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, label)

    Returns:
        (label, list of direction sequences, debug_info)
    """
    pcap_path, label = args
    debug_info = {'path': pcap_path, 'exists': False, 'total_packets': 0,
                  'ip_packets': 0, 'tcp_udp_packets': 0, 'payload_packets': 0,
                  'flows_before_filter': 0, 'flows_after_filter': 0,
                  'datalink': 0}

    if not os.path.exists(pcap_path):
        return (label, [], debug_info)

    debug_info['exists'] = True
    flows, stats = extract_flows_from_pcap(pcap_path, return_stats=True)
    debug_info.update(stats)

    # Extract direction sequences
    sequences = [flow.directions for flow in flows]

    return (label, sequences, debug_info)


# =============================================================================
# Dataset Creation
# =============================================================================

def load_label_map(csv_path: str) -> List[Tuple[str, int]]:
    """Load label map from CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row['path'], int(row['label'])))
    return data


def load_vocab(csv_path: str) -> Dict[int, str]:
    """Load service vocabulary from CSV."""
    vocab = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab[int(row['id'])] = row['name']
    return vocab


def process_dataset():
    """Process ISCX dataset to DeepFingerprinting format."""
    np.random.seed(RANDOM_SEED)

    # Check input files
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}")

    # Load metadata
    label_map = load_label_map(LABEL_MAP_PATH)
    vocab = load_vocab(VOCAB_PATH)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Loaded {len(label_map)} PCAP files")
    print(f"Classes: {list(vocab.values())}")
    print(f"Using {NUM_WORKERS} workers")

    # Prepare tasks
    tasks = [(pcap_path, label) for pcap_path, label in label_map]

    # Count files per class
    files_per_class = defaultdict(int)
    for _, label in label_map:
        files_per_class[label] += 1
    print("\nPCAP files per class:")
    for label_id in sorted(vocab.keys()):
        print(f"  [{label_id:2d}] {vocab[label_id]:15s}: {files_per_class.get(label_id, 0)} files")

    # Collect all sequences and labels
    all_sequences = []
    all_labels = []
    class_counts = defaultdict(int)
    failed_files = defaultdict(list)
    debug_infos = []

    # Process with multi-processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_pcap, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            task = futures[future]
            pcap_path, label = task
            try:
                result_label, sequences, debug_info = future.result()

                if len(sequences) == 0:
                    failed_files[label].append(pcap_path)
                    debug_infos.append((label, debug_info))

                for seq in sequences:
                    all_sequences.append(seq)
                    all_labels.append(result_label)
                    class_counts[result_label] += 1

            except Exception as e:
                failed_files[label].append(f"{pcap_path} (Error: {e})")

    # Print failed files summary
    print("\nFiles with 0 flows extracted:")
    for label_id in sorted(failed_files.keys()):
        if failed_files[label_id]:
            print(f"  [{label_id}] {vocab.get(label_id, 'Unknown')}: {len(failed_files[label_id])} files failed")

    # Print detailed debug info
    sample_debug = debug_infos[:5] if debug_infos else []
    if sample_debug:
        print("\nSample files debug info (first 5):")
        datalink_names = {0: 'NULL', 1: 'Ethernet', 101: 'Raw IP', 113: 'Linux SLL', 276: 'Linux SLL2'}
        for label, info in sample_debug:
            fname = os.path.basename(info['path'])
            dl = info.get('datalink', 0)
            dl_name = datalink_names.get(dl, f'Unknown({dl})')
            print(f"  {fname}: [datalink={dl_name}]")
            print(f"    total={info['total_packets']}, ip={info['ip_packets']}, "
                  f"tcp_udp={info['tcp_udp_packets']}, payload={info.get('payload_packets', 0)}")
            print(f"    flows_before_filter={info['flows_before_filter']}, "
                  f"flows_after_filter={info['flows_after_filter']} (min_packets={MIN_PACKETS})")

    # Filter out classes with less than MIN_SAMPLES samples
    MIN_SAMPLES = 10
    valid_classes = sorted([label_id for label_id in vocab.keys() if class_counts.get(label_id, 0) >= MIN_SAMPLES])
    removed_classes = sorted([label_id for label_id in vocab.keys() if class_counts.get(label_id, 0) < MIN_SAMPLES])

    if removed_classes:
        print(f"\nRemoving {len(removed_classes)} classes with < {MIN_SAMPLES} samples:")
        for label_id in removed_classes:
            count = class_counts.get(label_id, 0)
            print(f"  [{label_id:2d}] {vocab[label_id]} ({count} samples)")

    # Create new label mapping (old_id -> new_id)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(valid_classes)}
    new_vocab = {new_id: vocab[old_id] for new_id, old_id in enumerate(valid_classes)}

    # Filter out samples belonging to removed classes and remap labels
    valid_indices = [i for i, label in enumerate(all_labels) if label in old_to_new]
    all_sequences = [all_sequences[i] for i in valid_indices]
    all_labels_remapped = [old_to_new[all_labels[i]] for i in valid_indices]
    labels = np.array(all_labels_remapped, dtype=np.int64)

    print(f"\nTotal flows after filtering: {len(labels)}")
    print(f"Valid classes: {len(valid_classes)} (removed {len(removed_classes)} classes)")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert sequences to numpy arrays (variable length, stored as object array)
    X = np.array([np.array(seq, dtype=np.int8) for seq in all_sequences], dtype=object)
    y = labels

    # Save as NPZ (compatible with DeepFingerprinting train.py)
    npz_path = output_path / 'data.npz'
    np.savez(npz_path, X=X, y=y, allow_pickle=True)

    # Save labels.json
    labels_json = {
        'label2id': {name: idx for idx, name in new_vocab.items()},
        'id2label': {str(idx): name for idx, name in new_vocab.items()}
    }
    labels_path = output_path / 'labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    # Save class count CSV
    class_count_path = output_path / 'class_count.csv'
    with open(class_count_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'class_id', 'count'])
        for new_id, class_name in new_vocab.items():
            old_id = valid_classes[new_id]
            count = class_counts.get(old_id, 0)
            writer.writerow([class_name, new_id, count])

    # Save metadata
    meta = {
        'dataset': DATASET_NAME,
        'format': 'direction_sequence',
        'feature': '±1 direction (outgoing=+1, incoming=-1)',
        'payload_only': PAYLOAD_ONLY,
        'min_packets': MIN_PACKETS,
        'max_packets': MAX_PACKETS,
        'flow_timeout': FLOW_TIMEOUT,
        'num_classes': len(new_vocab),
        'total_samples': len(labels),
        'class_names': list(new_vocab.values()),
    }
    meta_path = output_path / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {len(new_vocab)}")
    print("\nFlows per class:")
    total_flows = len(labels)
    for new_id, class_name in new_vocab.items():
        old_id = valid_classes[new_id]
        count = class_counts.get(old_id, 0)
        pct = count / total_flows * 100 if total_flows > 0 else 0
        print(f"  [{new_id:2d}] {class_name:15s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nDataset saved to:")
    print(f"  {npz_path}")
    print(f"  {labels_path}")
    print(f"  {class_count_path}")
    print(f"  {meta_path}")

    print(f"\nUsage:")
    print(f"  # In DeepFingerprinting/train.py, set:")
    print(f"  data_path: str = '{OUTPUT_DIR}'")


def main():
    print("=" * 60)
    print("DeepFingerprinting ISCX Dataset Processor")
    print("=" * 60)
    print(f"Dataset:    {DATASET_NAME}")
    print(f"Label map:  {LABEL_MAP_PATH}")
    print(f"Vocab:      {VOCAB_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Workers:    {NUM_WORKERS}")
    print("=" * 60)

    process_dataset()


if __name__ == '__main__':
    main()
