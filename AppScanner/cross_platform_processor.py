"""
AppScanner ISCXVPN Dataset Processor

Converts ISCX-VPN-NonVPN dataset to AppScanner format (54-dim statistical features).
Uses dpkt for fast PCAP parsing with multi-processing support.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

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
LABEL_MAP_PATH = "/home/pcz/DL/ML_DL/public_dataset/Cross-Platform/label_map.csv"
VOCAB_PATH = "/home/pcz/DL/ML_DL/public_dataset/Cross-Platform/app_vocab.csv"

# Output path
OUTPUT_DIR = "/home/pcz/DL/ML_DL/AppScanner/data/cross_platform"

# Flow extraction parameters
MIN_PACKETS = 7           # Minimum packets per flow (AppScanner default)
MAX_PACKETS = 260         # Maximum packets per flow (AppScanner default)

# Random seed for reproducibility
RANDOM_SEED = 42

# Multi-process
NUM_WORKERS = 8


# =============================================================================
# Statistical Feature Extraction (54 features)
# =============================================================================

@dataclass
class FlowData:
    """Extracted flow data."""
    lengths: List[int]
    directions: List[int]  # 1=incoming, -1=outgoing


def extract_direction_features(lengths: np.ndarray) -> np.ndarray:
    """
    Extract 18 statistical features from packet lengths.

    Features:
    1. Packet count
    2-6. Min, Max, Mean, Std, Variance
    7-8. Skewness, Kurtosis
    9. Median Absolute Deviation (MAD)
    10-18. Percentiles (10, 20, 30, 40, 50, 60, 70, 80, 90)
    """
    if len(lengths) == 0:
        return np.zeros(18)

    features = []

    # 1. Packet count
    features.append(len(lengths))

    # 2-6. Basic statistics
    features.append(np.min(lengths))
    features.append(np.max(lengths))
    features.append(np.mean(lengths))
    features.append(np.std(lengths))
    features.append(np.var(lengths))

    # 7-8. Shape statistics
    if len(lengths) >= 3 and np.std(lengths) > 1e-10:
        # Only compute if there's enough variance to avoid precision issues
        features.append(scipy_stats.skew(lengths))
        features.append(scipy_stats.kurtosis(lengths))
    else:
        features.extend([0.0, 0.0])

    # 9. Median Absolute Deviation
    median = np.median(lengths)
    mad = np.median(np.abs(lengths - median))
    features.append(mad)

    # 10-18. Percentiles
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        features.append(np.percentile(lengths, p))

    return np.array(features)


def extract_flow_features(flow: FlowData) -> Optional[np.ndarray]:
    """
    Extract 54-dimensional feature vector from flow.

    18 features × 3 directions (incoming, outgoing, bidirectional) = 54
    """
    lengths = np.array(flow.lengths[:MAX_PACKETS])
    directions = np.array(flow.directions[:MAX_PACKETS])

    if len(lengths) < MIN_PACKETS:
        return None

    # Separate by direction
    incoming_mask = directions > 0
    outgoing_mask = directions < 0

    incoming_lengths = lengths[incoming_mask]
    outgoing_lengths = lengths[outgoing_mask]
    bidirectional_lengths = lengths

    # Extract features for each direction
    incoming_features = extract_direction_features(incoming_lengths)
    outgoing_features = extract_direction_features(outgoing_lengths)
    bidirectional_features = extract_direction_features(bidirectional_lengths)

    # Concatenate: 18 * 3 = 54 features
    features = np.concatenate([
        incoming_features,
        outgoing_features,
        bidirectional_features,
    ])

    return features


# =============================================================================
# PCAP Processing
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
            # First byte indicates IP version
            if len(buf) > 0:
                version = (buf[0] >> 4) & 0xF
                if version == 4:
                    return dpkt.ip.IP(buf)

        # DLT_LINUX_SLL = 113 (Linux cooked capture)
        elif datalink == 113:
            if len(buf) >= 16:
                # Linux SLL header is 16 bytes
                # Protocol type is at bytes 14-15
                proto = (buf[14] << 8) | buf[15]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[16:])

        # DLT_LINUX_SLL2 = 276 (Linux cooked capture v2)
        elif datalink == 276:
            if len(buf) >= 20:
                # SLL2 header is 20 bytes, protocol at bytes 0-1
                proto = (buf[0] << 8) | buf[1]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[20:])

        # DLT_NULL = 0 (BSD loopback)
        elif datalink == 0:
            if len(buf) >= 4:
                # 4-byte header, check for AF_INET
                family = buf[0] if buf[0] != 0 else buf[3]
                if family == 2:  # AF_INET
                    return dpkt.ip.IP(buf[4:])

    except Exception:
        pass

    return None


def extract_flows_from_pcap(pcap_path: str, return_stats: bool = False):
    """
    Extract flows from PCAP using dpkt.

    Returns list of FlowData objects with packet lengths and directions.
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
        directions = []

        for pkt_time, pkt_len, src_ip in flow_packets:
            lengths.append(pkt_len)
            # Outgoing: from client, Incoming: to client
            if src_ip == client_ip:
                directions.append(-1)  # Outgoing
            else:
                directions.append(1)   # Incoming

        if len(lengths) >= MIN_PACKETS:
            result.append(FlowData(lengths=lengths, directions=directions))

    stats['flows_after_filter'] = len(result)
    del flows

    if return_stats:
        return result, stats
    return result


def process_single_pcap(args: Tuple[str, int], debug: bool = False) -> Tuple[int, List[np.ndarray], Dict]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, label)
        debug: if True, return debug info

    Returns:
        (label, list of 54-dim feature vectors, debug_info)
    """
    pcap_path, label = args
    debug_info = {'path': pcap_path, 'exists': False, 'total_packets': 0,
                  'ip_packets': 0, 'flows_before_filter': 0, 'flows_after_filter': 0}

    if not os.path.exists(pcap_path):
        return (label, [], debug_info)

    debug_info['exists'] = True
    flows, stats = extract_flows_from_pcap(pcap_path, return_stats=True)
    debug_info.update(stats)

    features_list = []
    for flow in flows:
        features = extract_flow_features(flow)
        if features is not None:
            features_list.append(features)

    debug_info['flows_after_filter'] = len(features_list)

    return (label, features_list, debug_info)


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
    """Load app_id vocabulary from CSV."""
    vocab = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab[int(row['app_id'])] = row['app_name']
    return vocab


def process_iscxvpn_dataset():
    """Process ISCXVPN dataset to AppScanner format."""
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

    # Collect all features and labels
    all_features = []
    all_labels = []
    class_counts = defaultdict(int)
    failed_files = defaultdict(list)  # Track failed files per class
    debug_infos = []  # Collect debug info for failed files

    # Process with multi-processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_pcap, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            task = futures[future]
            pcap_path, label = task
            try:
                result_label, features_list, debug_info = future.result()

                if len(features_list) == 0:
                    failed_files[label].append(pcap_path)
                    debug_infos.append((label, debug_info))

                for features in features_list:
                    all_features.append(features)
                    all_labels.append(result_label)
                    class_counts[result_label] += 1

            except Exception as e:
                failed_files[label].append(f"{pcap_path} (Error: {e})")

    # Print failed files summary with debug info
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
    all_features = [all_features[i] for i in valid_indices]
    all_labels_remapped = [old_to_new[all_labels[i]] for i in valid_indices]

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels_remapped, dtype=np.int64)

    print(f"\nTotal flows after filtering: {len(labels)}")
    print(f"Feature shape: {features.shape}")
    print(f"Valid classes: {len(valid_classes)} (removed {len(removed_classes)} classes)")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle (all data, no split - split will be done during training)
    data = {
        'features': features,
        'labels': labels,
        'label_map': new_vocab,
        'num_classes': len(new_vocab),
        'num_features': 54,
    }

    pickle_path = output_path / 'cross_platform_appscanner.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    # Also save as npz for convenience
    npz_path = output_path / 'cross_platform_appscanner.npz'
    np.savez(
        npz_path,
        features=features,
        labels=labels,
    )

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
    print(f"  {pickle_path}")
    print(f"  {npz_path}")

    print(f"\nUsage:")
    print(f"  from data import load_dataset, create_dataloaders")
    print(f"  features, labels, label_map = load_dataset('{pickle_path}')")


if __name__ == '__main__':
    process_iscxvpn_dataset()
