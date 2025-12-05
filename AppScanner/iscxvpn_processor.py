"""
AppScanner ISCXVPN Dataset Processor

Converts ISCX-VPN-NonVPN dataset to AppScanner format (54-dim statistical features).
Uses dpkt for fast PCAP parsing with multi-processing support.

Usage:
    python iscxvpn_processor.py
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
LABEL_MAP_PATH = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/label_map.csv"
VOCAB_PATH = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/service_vocab.csv"

# Output path
OUTPUT_DIR = "/home/pcz/DL/ML&DL/AppScanner/data/iscxvpn"

# Flow extraction parameters
MIN_PACKETS = 7           # Minimum packets per flow (AppScanner default)
MAX_PACKETS = 260         # Maximum packets per flow (AppScanner default)
FLOW_TIMEOUT = 60.0       # Flow timeout in seconds

# Dataset split
TRAIN_RATIO = 0.8
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
    if len(lengths) >= 3:
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

def extract_flows_from_pcap(pcap_path: str) -> List[FlowData]:
    """
    Extract flows from PCAP using dpkt.

    Returns list of FlowData objects with packet lengths and directions.
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    # Group packets by flow: flow_key -> [(time, length, src_ip), ...]
    flows = defaultdict(list)

    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue

                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data
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

                # Normalize flow key
                if (src_ip, sport) < (dst_ip, dport):
                    flow_key = (src_ip, dst_ip, sport, dport, proto)
                else:
                    flow_key = (dst_ip, src_ip, dport, sport, proto)

                flows[flow_key].append((float(ts), len(buf), src_ip))

    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        return []

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
        last_time = flow_packets[0][0]

        for pkt_time, pkt_len, src_ip in flow_packets:
            # Split by timeout
            if pkt_time - last_time > FLOW_TIMEOUT and len(lengths) >= MIN_PACKETS:
                result.append(FlowData(lengths=lengths, directions=directions))
                lengths = []
                directions = []
                client_ip = src_ip

            lengths.append(pkt_len)
            # Outgoing: from client, Incoming: to client
            if src_ip == client_ip:
                directions.append(-1)  # Outgoing
            else:
                directions.append(1)   # Incoming

            last_time = pkt_time

        if len(lengths) >= MIN_PACKETS:
            result.append(FlowData(lengths=lengths, directions=directions))

    del flows
    return result


def process_single_pcap(args: Tuple[str, int]) -> Tuple[int, List[np.ndarray]]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, label)

    Returns:
        (label, list of 54-dim feature vectors)
    """
    pcap_path, label = args

    if not os.path.exists(pcap_path):
        return (label, [])

    flows = extract_flows_from_pcap(pcap_path)

    features_list = []
    for flow in flows:
        features = extract_flow_features(flow)
        if features is not None:
            features_list.append(features)

    return (label, features_list)


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

    # Collect all features and labels
    all_features = []
    all_labels = []
    class_counts = defaultdict(int)

    # Process with multi-processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_pcap, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                label, features_list = future.result()

                for features in features_list:
                    all_features.append(features)
                    all_labels.append(label)
                    class_counts[label] += 1

            except Exception as e:
                print(f"Error: {e}")

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal flows extracted: {len(labels)}")
    print(f"Feature shape: {features.shape}")

    # Shuffle and split
    indices = np.random.permutation(len(labels))
    n_train = int(len(labels) * TRAIN_RATIO)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    data = {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'label_map': vocab,
        'num_classes': len(vocab),
        'num_features': 54,
    }

    pickle_path = output_path / 'iscxvpn_appscanner.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    # Also save as npz for convenience
    npz_path = output_path / 'iscxvpn_appscanner.npz'
    np.savez(
        npz_path,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print("\nFlows per class:")
    for label_id in sorted(class_counts.keys()):
        print(f"  {vocab[label_id]}: {class_counts[label_id]}")

    print(f"\nDataset saved to:")
    print(f"  {pickle_path}")
    print(f"  {npz_path}")

    print(f"\nUsage:")
    print(f"  from data import load_dataset, create_dataloaders")
    print(f"  features, labels, label_map = load_dataset('{pickle_path}')")


if __name__ == '__main__':
    process_iscxvpn_dataset()
