"""
FS-Net ISCX Dataset Processor

Converts ISCX-VPN-NonVPN dataset to FS-Net format (packet length sequences).
Uses dpkt for fast PCAP parsing with multi-processing support.
Supports multiple link layer types (Ethernet, Linux SLL, Raw IP, etc.)

Output: pickle file with all flows (no pre-split, split during training)

Usage:
    python iscx_vpn_processor.py
"""

import os
import csv
import pickle
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
# Configuration
# =============================================================================

# Input paths
LABEL_MAP_PATH = "/home/pcz/DL/ML_DL/public_dataset/ISCX-Tor-NonTor-2017/label_map.csv"
VOCAB_PATH = "/home/pcz/DL/ML_DL/public_dataset/ISCX-Tor-NonTor-2017/service_vocab.csv"

# Output path
OUTPUT_DIR = "/home/pcz/DL/ML_DL/FS-Net/data/iscxtor"

# Flow extraction parameters
MIN_PACKETS = 10          # Minimum packets per flow
MAX_PACKETS = 100         # Maximum packets per flow (FS-Net sequence length)
MAX_PACKET_LEN = 1500     # Maximum packet length (cap to MTU)
FLOW_TIMEOUT = 60.0       # Flow timeout in seconds

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
    lengths: List[int]  # Signed packet lengths


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
                return None  # Skip IPv6 for now

        # DLT_RAW = 101 (Raw IP)
        elif datalink == 101:
            if len(buf) > 0:
                version = (buf[0] >> 4) & 0xF
                if version == 4:
                    return dpkt.ip.IP(buf)

        # DLT_LINUX_SLL = 113 (Linux cooked capture)
        elif datalink == 113:
            if len(buf) >= 16:
                proto = (buf[14] << 8) | buf[15]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[16:])

        # DLT_LINUX_SLL2 = 276 (Linux cooked capture v2)
        elif datalink == 276:
            if len(buf) >= 20:
                proto = (buf[0] << 8) | buf[1]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[20:])

        # DLT_NULL = 0 (BSD loopback)
        elif datalink == 0:
            if len(buf) >= 4:
                family = buf[0] if buf[0] != 0 else buf[3]
                if family == 2:  # AF_INET
                    return dpkt.ip.IP(buf[4:])

    except Exception:
        pass

    return None


# =============================================================================
# PCAP Processing
# =============================================================================

def extract_flows_from_pcap(pcap_path: str, return_stats: bool = False):
    """
    Extract flows from PCAP using dpkt.

    Returns list of FlowData objects with signed packet lengths.
    If return_stats=True, also returns statistics dict.
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    stats = {'total_packets': 0, 'ip_packets': 0, 'tcp_udp_packets': 0,
             'flows_before_filter': 0, 'flows_after_filter': 0, 'datalink': 0}

    # Group packets by flow: flow_key -> [(time, length, src_ip), ...]
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
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    sport, dport, proto = tcp.sport, tcp.dport, 6
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    sport, dport, proto = udp.sport, udp.dport, 17
                else:
                    continue

                stats['tcp_udp_packets'] += 1

                # Normalize flow key
                if (src_ip, sport) < (dst_ip, dport):
                    flow_key = (src_ip, dst_ip, sport, dport, proto)
                else:
                    flow_key = (dst_ip, src_ip, dport, sport, proto)

                flows[flow_key].append((float(ts), len(buf), src_ip))

    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        if return_stats:
            return [], stats
        return []

    stats['flows_before_filter'] = len(flows)

    # Convert to FlowData objects
    result = []
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) < MIN_PACKETS:
            continue

        flow_packets.sort(key=lambda x: x[0])

        # Determine client IP (first packet's source)
        client_ip = flow_packets[0][2]

        lengths = []
        last_time = flow_packets[0][0]

        for pkt_time, pkt_len, src_ip in flow_packets:
            # Split by timeout
            if pkt_time - last_time > FLOW_TIMEOUT and len(lengths) >= MIN_PACKETS:
                result.append(FlowData(lengths=lengths[:MAX_PACKETS]))
                lengths = []
                client_ip = src_ip

            # Cap packet length to MTU
            pkt_len = min(pkt_len, MAX_PACKET_LEN)

            # Signed length: positive=outgoing (from client), negative=incoming
            if src_ip == client_ip:
                lengths.append(pkt_len)
            else:
                lengths.append(-pkt_len)

            last_time = pkt_time

        if len(lengths) >= MIN_PACKETS:
            result.append(FlowData(lengths=lengths[:MAX_PACKETS]))

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
        (label, list of length sequences, debug_info)
    """
    pcap_path, label = args
    debug_info = {'path': pcap_path, 'exists': False, 'total_packets': 0,
                  'ip_packets': 0, 'flows_before_filter': 0, 'flows_after_filter': 0,
                  'datalink': 0}

    if not os.path.exists(pcap_path):
        return (label, [], debug_info)

    debug_info['exists'] = True
    flows, stats = extract_flows_from_pcap(pcap_path, return_stats=True)
    debug_info.update(stats)

    # Extract length sequences
    sequences = [flow.lengths for flow in flows]

    return (label, sequences, debug_info)


# =============================================================================
# Dataset Creation
# =============================================================================

def load_label_map(csv_path: str) -> List[Tuple[str, int]]:
    """Load label map from CSV."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row['path'], int(row['label'])))
    return data


def load_vocab(csv_path: str) -> Dict[int, str]:
    """Load service vocabulary from CSV."""
    vocab = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab[int(row['service_id'])] = row['service']
    return vocab


def process_iscx_dataset():
    """Process ISCX dataset to FS-Net format."""
    np.random.seed(RANDOM_SEED)

    # Load metadata
    label_map = load_label_map(LABEL_MAP_PATH)
    vocab = load_vocab(VOCAB_PATH)

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

    # Print detailed debug info for VPN classes (6-11)
    vpn_debug = [d for l, d in debug_infos if l >= 6]
    if vpn_debug:
        print("\nVPN files debug info (first 5):")
        datalink_names = {0: 'NULL', 1: 'Ethernet', 101: 'Raw IP', 113: 'Linux SLL', 276: 'Linux SLL2'}
        for info in vpn_debug[:5]:
            fname = os.path.basename(info['path'])
            dl = info.get('datalink', 0)
            dl_name = datalink_names.get(dl, f'Unknown({dl})')
            print(f"  {fname}: [datalink={dl_name}]")
            print(f"    total_packets={info['total_packets']}, ip_packets={info['ip_packets']}, "
                  f"tcp_udp={info['tcp_udp_packets']}")
            print(f"    flows_before_filter={info['flows_before_filter']}, "
                  f"flows_after_filter={info['flows_after_filter']} (min_packets={MIN_PACKETS})")

    # Convert labels to numpy array
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal flows extracted: {len(labels)}")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle (all data, no split - split will be done during training)
    data = {
        'sequences': all_sequences,  # List of variable-length sequences
        'labels': labels,
        'label_map': vocab,
        'num_classes': len(vocab),
        'max_seq_len': MAX_PACKETS,
        'max_packet_len': MAX_PACKET_LEN,
        'min_packets': MIN_PACKETS,
    }

    pickle_path = output_path / 'iscxvpn_fsnet.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {len(vocab)}")
    print("\nFlows per class:")
    total_flows = len(labels)
    for label_id in sorted(vocab.keys()):
        count = class_counts.get(label_id, 0)
        pct = count / total_flows * 100 if total_flows > 0 else 0
        print(f"  [{label_id:2d}] {vocab[label_id]:15s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nDataset saved to:")
    print(f"  {pickle_path}")

    print(f"\nUsage:")
    print(f"  python train.py --data_path {pickle_path} --num_classes {len(vocab)}")


if __name__ == '__main__':
    process_iscx_dataset()
