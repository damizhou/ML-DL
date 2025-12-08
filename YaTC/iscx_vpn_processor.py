"""
YaTC ISCXVPN Dataset Processor

Converts ISCX-VPN-NonVPN dataset to YaTC MFR format (40x40 grayscale images).
Uses dpkt for fast PCAP parsing with multi-processing support.

Usage:
    python iscx_vpn_processor.py
"""

import os
import csv
import json
import socket
import ipaddress
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

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

# Input paths (same as AppScanner)
LABEL_MAP_PATH = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/label_map.csv"
VOCAB_PATH = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/ISCXVPN/artifacts/iscx/service_vocab.csv"

# Output path
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "iscxvpn"

# MFR parameters (5 packets × 320 bytes = 1600 bytes = 40×40)
NUM_PACKETS = 5           # Number of packets per flow
HEADER_LEN = 80           # IP + TCP/UDP header (consistent with pcap_to_mfr.py)
PAYLOAD_LEN = 240         # Payload length (consistent with pcap_to_mfr.py)
BYTES_PER_PACKET = HEADER_LEN + PAYLOAD_LEN  # 320 bytes
TOTAL_BYTES = NUM_PACKETS * BYTES_PER_PACKET  # 1600 bytes
IMAGE_SIZE = 40           # 40×40 image

# Flow extraction parameters
MIN_PACKETS = NUM_PACKETS  # Minimum packets per flow (need at least 5 for MFR)
FLOW_TIMEOUT = 60.0        # Flow timeout in seconds

# Multi-process
NUM_WORKERS = 8

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Packet Data Structure
# =============================================================================

@dataclass
class PacketInfo:
    """Extracted packet information for MFR generation."""
    timestamp: float
    ip_header: bytes      # IP header bytes
    l4_header: bytes      # TCP/UDP header bytes
    payload: bytes        # Payload bytes
    src_ip: str
    direction: int        # 1=incoming, -1=outgoing


@dataclass
class FlowData:
    """Extracted flow data."""
    packets: List[PacketInfo]


# =============================================================================
# MFR Image Generation
# =============================================================================

def extract_packet_bytes(pkt: PacketInfo) -> bytes:
    """
    Extract packet bytes for MFR generation.

    Args:
        pkt: PacketInfo object

    Returns:
        BYTES_PER_PACKET bytes (header + payload, zero-padded if needed)
    """
    result = bytearray(BYTES_PER_PACKET)

    # Combine IP header and L4 header
    header = pkt.ip_header + pkt.l4_header
    header = header[:HEADER_LEN]  # Truncate to HEADER_LEN bytes

    # Payload
    payload = pkt.payload[:PAYLOAD_LEN]  # Truncate to PAYLOAD_LEN bytes

    # Fill result
    result[:len(header)] = header
    result[HEADER_LEN:HEADER_LEN + len(payload)] = payload

    return bytes(result)


def flow_to_mfr_image(flow: FlowData) -> Optional[np.ndarray]:
    """
    Convert flow to MFR image.

    Args:
        flow: FlowData object with packets

    Returns:
        40×40 uint8 numpy array, or None if not enough packets
    """
    if len(flow.packets) < MIN_PACKETS:
        return None

    # Extract bytes from first NUM_PACKETS packets
    packet_bytes_list = []
    for pkt in flow.packets[:NUM_PACKETS]:
        pkt_bytes = extract_packet_bytes(pkt)
        packet_bytes_list.append(pkt_bytes)

    # Pad if needed (shouldn't happen if MIN_PACKETS >= NUM_PACKETS)
    while len(packet_bytes_list) < NUM_PACKETS:
        packet_bytes_list.append(bytes(BYTES_PER_PACKET))

    # Concatenate all bytes
    all_bytes = b''.join(packet_bytes_list)

    # Ensure correct length
    if len(all_bytes) < TOTAL_BYTES:
        all_bytes = all_bytes + bytes(TOTAL_BYTES - len(all_bytes))
    else:
        all_bytes = all_bytes[:TOTAL_BYTES]

    # Convert to numpy array and reshape
    arr = np.frombuffer(all_bytes, dtype=np.uint8)
    image = arr.reshape(IMAGE_SIZE, IMAGE_SIZE)

    return image


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
        (IP packet object, raw_buf) or (None, None)
    """
    try:
        # DLT_EN10MB = 1 (Ethernet)
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                return eth.data, buf
            elif isinstance(eth.data, dpkt.ip6.IP6):
                return eth.data, buf

        # DLT_RAW = 101 (Raw IP)
        elif datalink == 101:
            if len(buf) > 0:
                version = (buf[0] >> 4) & 0xF
                if version == 4:
                    return dpkt.ip.IP(buf), buf
                elif version == 6:
                    return dpkt.ip6.IP6(buf), buf

        # DLT_LINUX_SLL = 113 (Linux cooked capture)
        elif datalink == 113:
            if len(buf) >= 16:
                proto = (buf[14] << 8) | buf[15]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[16:]), buf
                elif proto == 0x86DD:  # IPv6
                    return dpkt.ip6.IP6(buf[16:]), buf

        # DLT_LINUX_SLL2 = 276 (Linux cooked capture v2)
        elif datalink == 276:
            if len(buf) >= 20:
                proto = (buf[0] << 8) | buf[1]
                if proto == 0x0800:  # IPv4
                    return dpkt.ip.IP(buf[20:]), buf
                elif proto == 0x86DD:  # IPv6
                    return dpkt.ip6.IP6(buf[20:]), buf

        # DLT_NULL = 0 (BSD loopback)
        elif datalink == 0:
            if len(buf) >= 4:
                family = buf[0] if buf[0] != 0 else buf[3]
                if family == 2:  # AF_INET
                    return dpkt.ip.IP(buf[4:]), buf

    except Exception:
        pass

    return None, None


def extract_flows_from_pcap(pcap_path: str, return_stats: bool = False):
    """
    Extract flows from PCAP using dpkt.

    Returns list of FlowData objects with packet information.
    If return_stats=True, also returns statistics dict.
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required")

    stats = {
        'total_packets': 0,
        'ip_packets': 0,
        'tcp_udp_packets': 0,
        'flows_before_filter': 0,
        'flows_after_filter': 0,
        'datalink': 0
    }

    # Group packets by flow: flow_key -> [PacketInfo, ...]
    flows: Dict[tuple, List[PacketInfo]] = defaultdict(list)

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
                ip, raw_buf = extract_ip_from_buf(buf, datalink)
                if ip is None:
                    continue

                stats['ip_packets'] += 1

                # Extract headers and payload
                ip_header = b''
                l4_header = b''
                payload = b''
                src_ip = ''
                dst_ip = ''
                sport = dport = 0
                proto_str = None

                if isinstance(ip, dpkt.ip.IP):
                    ip_header = bytes(ip.pack_hdr())
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)

                    if ip.p == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP):
                        proto_str = "TCP"
                        tcp = ip.data
                        sport, dport = tcp.sport, tcp.dport
                        l4_header = bytes(tcp.pack_hdr())
                        payload = bytes(tcp.data) if tcp.data else b''
                    elif ip.p == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP):
                        proto_str = "UDP"
                        udp = ip.data
                        sport, dport = udp.sport, udp.dport
                        l4_header = bytes(udp.pack_hdr())
                        payload = bytes(udp.data) if udp.data else b''

                elif isinstance(ip, dpkt.ip6.IP6):
                    ip_header = bytes(ip.pack_hdr())
                    src_ip = str(ipaddress.ip_address(ip.src))
                    dst_ip = str(ipaddress.ip_address(ip.dst))

                    if ip.nxt == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP):
                        proto_str = "TCP"
                        tcp = ip.data
                        sport, dport = tcp.sport, tcp.dport
                        l4_header = bytes(tcp.pack_hdr())
                        payload = bytes(tcp.data) if tcp.data else b''
                    elif ip.nxt == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP):
                        proto_str = "UDP"
                        udp = ip.data
                        sport, dport = udp.sport, udp.dport
                        l4_header = bytes(udp.pack_hdr())
                        payload = bytes(udp.data) if udp.data else b''

                if proto_str is None:
                    continue

                stats['tcp_udp_packets'] += 1

                # Normalize flow key (bidirectional)
                if (src_ip, sport) < (dst_ip, dport):
                    flow_key = (proto_str, src_ip, dst_ip, sport, dport)
                    direction = -1  # Outgoing (from smaller to larger)
                else:
                    flow_key = (proto_str, dst_ip, src_ip, dport, sport)
                    direction = 1   # Incoming (from larger to smaller)

                pkt_info = PacketInfo(
                    timestamp=float(ts),
                    ip_header=ip_header,
                    l4_header=l4_header,
                    payload=payload,
                    src_ip=src_ip,
                    direction=direction
                )

                flows[flow_key].append(pkt_info)

    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        if return_stats:
            return [], stats
        return []

    stats['flows_before_filter'] = len(flows)

    # Convert to FlowData objects
    result = []
    for flow_key, packets in flows.items():
        if len(packets) < MIN_PACKETS:
            continue

        # Sort by timestamp
        packets.sort(key=lambda x: x.timestamp)

        # Split by timeout
        current_packets = []
        last_time = packets[0].timestamp

        for pkt in packets:
            if pkt.timestamp - last_time > FLOW_TIMEOUT and len(current_packets) >= MIN_PACKETS:
                result.append(FlowData(packets=current_packets))
                current_packets = []

            current_packets.append(pkt)
            last_time = pkt.timestamp

        if len(current_packets) >= MIN_PACKETS:
            result.append(FlowData(packets=current_packets))

    stats['flows_after_filter'] = len(result)
    del flows

    if return_stats:
        return result, stats
    return result


def process_single_pcap(args: Tuple[str, int]) -> Tuple[int, List[np.ndarray], Dict]:
    """
    Worker: process one PCAP file.

    Args:
        args: (pcap_path, label)

    Returns:
        (label, list of MFR images, debug_info)
    """
    pcap_path, label = args
    debug_info = {
        'path': pcap_path,
        'exists': False,
        'total_packets': 0,
        'ip_packets': 0,
        'flows_before_filter': 0,
        'flows_after_filter': 0
    }

    if not os.path.exists(pcap_path):
        return (label, [], debug_info)

    debug_info['exists'] = True
    flows, stats = extract_flows_from_pcap(pcap_path, return_stats=True)
    debug_info.update(stats)

    images_list = []
    for flow in flows:
        image = flow_to_mfr_image(flow)
        if image is not None:
            images_list.append(image)

    debug_info['images_generated'] = len(images_list)

    return (label, images_list, debug_info)


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
    """Process ISCXVPN dataset to YaTC MFR format."""
    np.random.seed(RANDOM_SEED)

    # Load metadata
    label_map = load_label_map(LABEL_MAP_PATH)
    vocab = load_vocab(VOCAB_PATH)

    print(f"YaTC ISCXVPN Dataset Processor")
    print("=" * 60)
    print(f"Loaded {len(label_map)} PCAP files")
    print(f"Classes: {list(vocab.values())}")
    print(f"Using {NUM_WORKERS} workers")
    print(f"\nMFR Configuration:")
    print(f"  Packets per flow: {NUM_PACKETS}")
    print(f"  Header length: {HEADER_LEN} bytes")
    print(f"  Payload length: {PAYLOAD_LEN} bytes")
    print(f"  Bytes per packet: {BYTES_PER_PACKET} bytes")
    print(f"  Total bytes: {TOTAL_BYTES} bytes")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")

    # Prepare tasks
    tasks = [(pcap_path, label) for pcap_path, label in label_map]

    # Count files per class
    files_per_class = defaultdict(int)
    for _, label in label_map:
        files_per_class[label] += 1
    print("\nPCAP files per class:")
    for label_id in sorted(vocab.keys()):
        print(f"  [{label_id:2d}] {vocab[label_id]:15s}: {files_per_class.get(label_id, 0)} files")

    # Collect all images and labels
    all_images = []
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
                result_label, images_list, debug_info = future.result()

                if len(images_list) == 0:
                    failed_files[label].append(pcap_path)
                    debug_infos.append((label, debug_info))

                for image in images_list:
                    all_images.append(image)
                    all_labels.append(result_label)
                    class_counts[result_label] += 1

            except Exception as e:
                failed_files[label].append(f"{pcap_path} (Error: {e})")

    # Print failed files summary
    print("\nFiles with 0 flows extracted:")
    for label_id in sorted(failed_files.keys()):
        if failed_files[label_id]:
            print(f"  [{label_id}] {vocab.get(label_id, 'Unknown')}: {len(failed_files[label_id])} files failed")

    # Print detailed debug info for VPN classes
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

    # Convert to numpy arrays
    images = np.stack(all_images, axis=0).astype(np.uint8)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal MFR images extracted: {len(labels)}")
    print(f"Images shape: {images.shape}")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save per-class NPZ files (same format as unified_vpn_processor.py)
    # Format: <label>.npz in root directory
    saved_labels = []
    for label_id in sorted(vocab.keys()):
        label_name = vocab[label_id]
        mask = labels == label_id
        if mask.sum() > 0:
            class_images = images[mask]
            class_path = output_path / f'{label_name}.npz'
            np.savez_compressed(
                class_path,
                images=class_images,
                label=label_name,
                label_id=label_id
            )
            saved_labels.append(label_name)
            print(f"  Saved {class_path.name}: {len(class_images)} images")

    # Save label mapping as JSON (same format as unified_vpn_processor.py)
    # Only include labels that were actually saved
    final_label2id = {label: i for i, label in enumerate(saved_labels)}
    final_id2label = {i: label for i, label in enumerate(saved_labels)}

    labels_json_path = output_path / 'labels.json'
    with open(labels_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': final_label2id,
            'id2label': {str(k): v for k, v in final_id2label.items()},
            'num_classes': len(saved_labels),
            'mfr_config': {
                'num_packets': NUM_PACKETS,
                'header_len': HEADER_LEN,
                'payload_len': PAYLOAD_LEN,
                'bytes_per_packet': BYTES_PER_PACKET,
                'total_bytes': TOTAL_BYTES,
                'image_size': IMAGE_SIZE
            }
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {len(saved_labels)}")
    print(f"Image shape: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("\nMFR images per class:")
    total_images = len(labels)
    for label_id in sorted(vocab.keys()):
        count = class_counts.get(label_id, 0)
        pct = count / total_images * 100 if total_images > 0 else 0
        print(f"  [{label_id:2d}] {vocab[label_id]:15s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nDataset saved to:")
    print(f"  {output_path}/")
    print(f"  ├── labels.json")
    for label_name in saved_labels:
        print(f"  ├── {label_name}.npz")

    print(f"\nUsage (same as unified_vpn_processor.py):")
    print(f"  python train_vpn.py pretrain")
    print(f"  python train_vpn.py finetune")
    print(f"")
    print(f"  # Or manually load:")
    print(f"  from YaTC.refactor.data import build_npz_pretrain_dataloader")
    print(f"  dataloader = build_npz_pretrain_dataloader('{output_path}')")


if __name__ == '__main__':
    process_iscxvpn_dataset()
