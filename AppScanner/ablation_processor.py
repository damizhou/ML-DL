"""
AppScanner Ablation Study Dataset Processor

处理消融实验数据集，包括：
- 数据集 A (batch): 连续访问10个URL -> 1个pcap (真实会话)
- 数据集 B (single): 10个URL分别独立访问 -> 10个pcap (精细粒度特征库)

目录结构：
/netdisk/dataset/ablation_study/
├── batch/           # 数据集A：连续访问
│   ├── website1/
│   │   ├── batch_1.pcap
│   │   ├── batch_2.pcap
│   │   └── ...
│   └── website2/
│       └── ...
└── single/          # 数据集B：单独访问
    ├── website1/
    │   ├── homepage.pcap      # 首页
    │   ├── subpage1.pcap      # 子页面1
    │   ├── subpage2.pcap      # 子页面2
    │   └── ...
    └── website2/
        └── ...

Usage:
    python ablation_processor.py
"""

import os
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
DATA_ROOT = "/netdisk/dataset/ablation_study"
BATCH_DIR = os.path.join(DATA_ROOT, "batch")    # 数据集A
SINGLE_DIR = os.path.join(DATA_ROOT, "single")  # 数据集B

# Output path
OUTPUT_DIR = "/home/pcz/DL/ML_DL/AppScanner/data/ablation_study"

# Flow extraction parameters
MIN_PACKETS = 7           # Minimum packets per flow (AppScanner default)
MAX_PACKETS = 260         # Maximum packets per flow (AppScanner default)

# Random seed for reproducibility
RANDOM_SEED = 42

# Multi-process
NUM_WORKERS = 8

# Homepage identifier: files starting with "1_" in single dataset
HOMEPAGE_PREFIX = '1_'


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
    """Extract IP packet from buffer based on datalink type."""
    try:
        # DLT_EN10MB = 1 (Ethernet)
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                return eth.data
            elif isinstance(eth.data, dpkt.ip6.IP6):
                return None  # Skip IPv6

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


def extract_flows_from_pcap(pcap_path: str) -> List[FlowData]:
    """Extract flows from PCAP using dpkt."""
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    flows = defaultdict(list)

    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            datalink = pcap.datalink()

            for ts, buf in pcap:
                ip = extract_ip_from_buf(buf, datalink)
                if ip is None:
                    continue

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
        client_ip = flow_packets[0][2]

        lengths = []
        directions = []

        for pkt_time, pkt_len, src_ip in flow_packets:
            lengths.append(pkt_len)
            if src_ip == client_ip:
                directions.append(-1)  # Outgoing
            else:
                directions.append(1)   # Incoming

        if len(lengths) >= MIN_PACKETS:
            result.append(FlowData(lengths=lengths, directions=directions))

    return result


def is_homepage(filename: str) -> bool:
    """Determine if a PCAP file is a homepage based on filename.

    Homepage files start with '1_' in the single dataset.
    Example: 1_20251219_19_05_56_website.pcap
    """
    return filename.startswith(HOMEPAGE_PREFIX)


def process_single_pcap(args: Tuple[str, int, str]) -> Tuple[int, List[np.ndarray], str, str]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, label, page_type)
            page_type: 'homepage', 'subpage', 'aggregate'

    Returns:
        (label, list of 54-dim feature vectors, page_type, filename)
    """
    pcap_path, label, page_type = args

    if not os.path.exists(pcap_path):
        return (label, [], page_type, os.path.basename(pcap_path))

    flows = extract_flows_from_pcap(pcap_path)
    features_list = []

    for flow in flows:
        features = extract_flow_features(flow)
        if features is not None:
            features_list.append(features)

    return (label, features_list, page_type, os.path.basename(pcap_path))


# =============================================================================
# Dataset Creation
# =============================================================================

def collect_pcap_files(dataset_dir: str, dataset_type: str) -> Tuple[List[Tuple[str, int, str]], Dict]:
    """
    Collect PCAP files from dataset directory.

    Directory structure:
    dataset_dir/
    ├── website1/
    │   └── pcap/
    │       ├── batch_*.pcap (for batch dataset)
    │       ├── 1_*.pcap (homepage, for single dataset)
    │       ├── 2_*.pcap (subpage, for single dataset)
    │       └── ...

    Args:
        dataset_dir: Root directory (batch or single)
        dataset_type: 'batch' or 'single'

    Returns:
        List of (pcap_path, label, page_type) tuples, label_map
    """
    tasks = []
    label_map = {}  # website_name -> label_id
    label_id = 0

    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} does not exist")
        return tasks, label_map

    for website in sorted(os.listdir(dataset_dir)):
        website_dir = os.path.join(dataset_dir, website)
        if not os.path.isdir(website_dir):
            continue

        # Look for pcap subdirectory
        pcap_dir = os.path.join(website_dir, 'pcap')
        if not os.path.exists(pcap_dir):
            continue

        # Assign label
        if website not in label_map:
            label_map[website] = label_id
            label_id += 1

        label = label_map[website]

        # Collect PCAP files from pcap subdirectory
        for filename in os.listdir(pcap_dir):
            if not filename.endswith('.pcap'):
                continue

            pcap_path = os.path.join(pcap_dir, filename)

            # Determine page type
            if dataset_type == 'batch':
                page_type = 'aggregate'
            else:  # single
                page_type = 'homepage' if is_homepage(filename) else 'subpage'

            tasks.append((pcap_path, label, page_type))

    return tasks, label_map


def process_dataset(
    dataset_dir: str,
    dataset_type: str,
    output_name: str,
):
    """
    Process a dataset (batch or single).

    Args:
        dataset_dir: Dataset directory
        dataset_type: 'batch' or 'single'
        output_name: Output filename prefix
    """
    print(f"\n{'=' * 70}")
    print(f"Processing Dataset: {dataset_type.upper()}")
    print(f"{'=' * 70}")

    # Collect files
    tasks, label_map = collect_pcap_files(dataset_dir, dataset_type)
    print(f"Found {len(tasks)} PCAP files from {len(label_map)} websites")

    if len(tasks) == 0:
        print("No PCAP files found. Skipping.")
        return

    # Count files by type
    type_counts = defaultdict(int)
    for _, _, page_type in tasks:
        type_counts[page_type] += 1

    print(f"File distribution:")
    for page_type, count in sorted(type_counts.items()):
        print(f"  {page_type:12s}: {count:6d} files")

    # Process with multi-processing
    all_features = []
    all_labels = []
    all_page_types = []  # Track page type for each sample

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_pcap, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                label, features_list, page_type, filename = future.result()

                for features in features_list:
                    all_features.append(features)
                    all_labels.append(label)
                    all_page_types.append(page_type)

            except Exception as e:
                print(f"Error processing file: {e}")

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    page_types = np.array(all_page_types)

    print(f"\nTotal flows extracted: {len(labels)}")
    print(f"Feature shape: {features.shape}")

    # Statistics by page type
    print(f"\nFlows by page type:")
    for page_type in np.unique(page_types):
        count = (page_types == page_type).sum()
        print(f"  {page_type:12s}: {count:6d} flows")

    # Create label mapping (id -> website_name)
    id_to_website = {v: k for k, v in label_map.items()}

    # Split data by page type (for single dataset)
    if dataset_type == 'single':
        # Separate homepage and subpage
        homepage_mask = page_types == 'homepage'
        subpage_mask = page_types == 'subpage'

        data = {
            'all_features': features,
            'all_labels': labels,
            'homepage_features': features[homepage_mask],
            'homepage_labels': labels[homepage_mask],
            'subpage_features': features[subpage_mask],
            'subpage_labels': labels[subpage_mask],
            'label_map': id_to_website,
            'num_classes': len(label_map),
            'num_features': 54,
        }
    else:  # batch
        data = {
            'features': features,
            'labels': labels,
            'label_map': id_to_website,
            'num_classes': len(label_map),
            'num_features': 54,
        }

    # Save
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    pickle_path = output_path / f"{output_name}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nDataset saved to: {pickle_path}")

    # Print summary
    print(f"\nFlows per website:")
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1

    for label_id in sorted(class_counts.keys()):
        website = id_to_website[label_id]
        count = class_counts[label_id]
        pct = count / len(labels) * 100
        print(f"  [{label_id:3d}] {website:30s}: {count:6d} ({pct:5.1f}%)")


def main():
    """Process both datasets for ablation study."""
    np.random.seed(RANDOM_SEED)

    print("AppScanner Ablation Study Dataset Processor")
    print("=" * 70)

    # Process Dataset B (single)
    if os.path.exists(SINGLE_DIR):
        process_dataset(
            dataset_dir=SINGLE_DIR,
            dataset_type='single',
            output_name='dataset_b_single',
        )
    else:
        print(f"Warning: Single dataset directory not found: {SINGLE_DIR}")

    # Process Dataset A (batch)
    if os.path.exists(BATCH_DIR):
        process_dataset(
            dataset_dir=BATCH_DIR,
            dataset_type='batch',
            output_name='dataset_a_batch',
        )
    else:
        print(f"Warning: Batch dataset directory not found: {BATCH_DIR}")

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - dataset_b_single.pkl  (数据集B: 单独访问，区分首页/子页面)")
    print("  - dataset_a_batch.pkl   (数据集A: 连续访问)")
    print("\nNext steps:")
    print("  python train_ablation.py --experiment 1  # 实验1: 仅首页训练")
    print("  python train_ablation.py --experiment 2  # 实验2: 全站训练")
    print("  python train_ablation.py --experiment 3  # 实验3: 连续会话训练")


if __name__ == '__main__':
    main()
