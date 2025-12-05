"""
FS-Net ISCX Dataset Processor (Multi-process, Memory Efficient)

Converts ISCX-VPN-NonVPN dataset to FS-Net format.
Each PCAP file contains multiple flows, which need to be extracted separately.

Memory optimization: process and save immediately, don't accumulate in memory.

Usage:
    python iscx_processor.py
"""

import os
import csv
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# Configuration - Hardcoded Parameters
# =============================================================================

# Input paths
LABEL_MAP_PATH = "/home/dev/DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/label_map.csv"
VOCAB_PATH = "/home/dev/DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/service_vocab.csv"

# Output path
OUTPUT_DIR = "/home/dev/DL/FS-Net/data/iscx_fsnet"

# Flow extraction parameters
MIN_PACKETS = 10          # Minimum packets per flow
MAX_PACKETS = 100         # Maximum packets per flow (FS-Net sequence length)
MAX_PACKET_LEN = 1500     # Maximum packet length (cap to MTU)
FLOW_TIMEOUT = 60.0       # Flow timeout in seconds

# Dataset split
TRAIN_RATIO = 0.8         # Training set ratio
RANDOM_SEED = 42          # Random seed for reproducibility

# Multi-process (reduce if memory is still an issue)
NUM_WORKERS = 32           # Number of parallel workers


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


def extract_flows_from_pcap(pcap_path: str) -> List[List[int]]:
    """
    Extract flows from PCAP using dpkt (faster than scapy).

    Uses dpkt to read packets with minimal parsing overhead.
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    import socket

    # Group packets by flow: flow_key -> [(time, length, src_ip), ...]
    flows = defaultdict(list)

    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                # Try pcapng format
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
                pkt_len = len(buf)

                # Get src/dst IP as string
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

                # Use tuple as key (more memory efficient than dataclass)
                if (src_ip, sport) < (dst_ip, dport):
                    flow_key = (src_ip, dst_ip, sport, dport, proto)
                else:
                    flow_key = (dst_ip, src_ip, dport, sport, proto)

                flows[flow_key].append((float(ts), pkt_len, src_ip))
    except Exception:
        return []

    # Process flows
    result = []
    for flow_packets in flows.values():
        if len(flow_packets) < MIN_PACKETS:
            continue

        flow_packets.sort(key=lambda x: x[0])

        current_lengths = []
        last_time = flow_packets[0][0]
        client_ip = flow_packets[0][2]

        for pkt_time, pkt_len, src_ip in flow_packets:
            # Split by timeout
            if pkt_time - last_time > FLOW_TIMEOUT and len(current_lengths) >= MIN_PACKETS:
                result.append(current_lengths[:MAX_PACKETS])
                current_lengths = []
                client_ip = src_ip

            # Cap packet length to MAX_PACKET_LEN
            pkt_len = min(pkt_len, MAX_PACKET_LEN)

            # Signed length: positive=outgoing, negative=incoming
            if src_ip == client_ip:
                current_lengths.append(pkt_len)
            else:
                current_lengths.append(-pkt_len)
            last_time = pkt_time

        if len(current_lengths) >= MIN_PACKETS:
            result.append(current_lengths[:MAX_PACKETS])

    del flows
    return result


def process_single_pcap(args: Tuple[str, str]) -> Tuple[str, List[List[int]]]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, class_name)

    Returns:
        (class_name, list of length sequences)
    """
    pcap_path, class_name = args

    if not os.path.exists(pcap_path):
        return (class_name, [])

    flows = extract_flows_from_pcap(pcap_path)
    return (class_name, flows)


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
    """Process ISCX dataset with multi-processing, memory efficient."""
    random.seed(RANDOM_SEED)

    # Load metadata
    label_map = load_label_map(LABEL_MAP_PATH)
    vocab = load_vocab(VOCAB_PATH)

    print(f"Loaded {len(label_map)} PCAP files")
    print(f"Classes: {list(vocab.values())}")
    print(f"Using {NUM_WORKERS} workers")

    # Create output directories
    output_path = Path(OUTPUT_DIR)
    for split in ['train', 'test']:
        for class_name in vocab.values():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Prepare tasks: (pcap_path, class_name)
    tasks = [(pcap_path, vocab[label]) for pcap_path, label in label_map]

    # Counters for file naming
    flow_counts = defaultdict(int)
    total_flows = 0

    # Process with multi-processing, save immediately as results come in
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_pcap, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                class_name, flows = future.result()

                if not flows:
                    continue

                # Shuffle and save immediately
                random.shuffle(flows)
                n_train = int(len(flows) * TRAIN_RATIO)

                for i, lengths in enumerate(flows):
                    split = 'train' if i < n_train else 'test'
                    flow_id = flow_counts[class_name]
                    flow_counts[class_name] += 1
                    total_flows += 1

                    out_path = output_path / split / class_name / f"flow_{flow_id:06d}.json"
                    with open(out_path, 'w') as f:
                        json.dump({'lengths': lengths}, f)

                # Release memory
                del flows

            except Exception as e:
                print(f"Error: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Total flows extracted: {total_flows}")
    print("\nFlows per class:")
    for class_name, count in sorted(flow_counts.items()):
        print(f"  {class_name}: {count}")

    # Save dataset info
    info = {
        'num_classes': len(vocab),
        'classes': list(vocab.values()),
        'flow_counts': dict(flow_counts),
        'total_flows': total_flows,
        'min_packets': MIN_PACKETS,
        'max_packets': MAX_PACKETS,
    }
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print(f"Use with FS-Net:")
    print(f"  python train.py --data_path {OUTPUT_DIR} --num_classes {len(vocab)}")


if __name__ == '__main__':
    process_iscx_dataset()
