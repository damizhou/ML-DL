#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpn_build_npz.py - YaTC VPN 数据处理

功能：
- 递归扫描 ROOT 目录（按子目录名作为 label）下的 .pcap / .pcapng
- 多进程并行处理 PCAP 文件
- 生成 MFR（Multi-level Flow Representation）图像
- 每个 label 的所有流合并为一个 NPZ 文件
- 输出格式兼容 YaTC 训练脚本

MFR 格式：
    - 每个流取 5 个数据包
    - 每个数据包: 160 字节头部 + 480 字节载荷 = 320 字节
    - 总计: 5 × 320 = 1600 字节 → 40×40 像素矩阵

输出 NPZ 结构：
    - images: uint8 数组 (N, 40, 40)
    - labels: int64 数组 (N,)
"""

from __future__ import annotations

import os
import json
import ipaddress
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable
from collections import defaultdict
import multiprocessing as mp

import numpy as np

try:
    import dpkt
except ImportError:
    raise SystemExit("需要安装 dpkt: pip install dpkt")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ========== 配置 ==========
ROOT = Path("/netdisk/dataset/vpn/data")  # 输入：一级子目录名为标签
OUT_ROOT = Path(__file__).resolve().parent / "vpn_yatc_data"  # 输出目录
MAX_PROCS = 16  # 并发进程数

# MFR 参数
NUM_PACKETS = 5  # 每个流使用的数据包数
HEADER_LEN = 160  # 头部字节数 (实际使用的字节)
PAYLOAD_LEN = 480  # 载荷字节数
BYTES_PER_PACKET = HEADER_LEN + PAYLOAD_LEN  # 320 字节
TOTAL_BYTES = NUM_PACKETS * BYTES_PER_PACKET  # 1600 字节
IMAGE_SIZE = 40  # 40×40 图像

MIN_PKTS_PER_FLOW = NUM_PACKETS  # 最少需要 5 个包
EXCLUDE_UDP_PORTS = {5353}  # 排除 mDNS
MIN_PCAP_SIZE = 20 * 1024  # 最小 PCAP 文件大小 (20KB)
# ==========================


def _parse_l3(buf: bytes):
    """解析 L3 层（IPv4/IPv6），兼容 Ethernet/SLL/RAW"""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data, buf
    except Exception:
        pass

    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data, buf
    except Exception:
        pass

    try:
        ver = buf[0] >> 4
        if ver == 4:
            return dpkt.ip.IP(buf), buf
        elif ver == 6:
            return dpkt.ip6.IP6(buf), buf
    except Exception:
        pass

    return None, None


def _iter_packets(pcap_path: Path) -> Iterable[Tuple[float, bytes]]:
    """迭代 PCAP/PCAPNG 文件中的数据包"""
    with pcap_path.open("rb") as f:
        try:
            f.seek(0)
            reader = dpkt.pcap.Reader(f)
            for ts, buf in reader:
                yield ts, buf
            return
        except Exception:
            pass

        f.seek(0)
        try:
            reader = dpkt.pcapng.Reader(f)
            for rec in reader:
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    yield rec[0], rec[1]
        except Exception:
            pass


def extract_packet_bytes(l3, raw_buf: bytes) -> bytes:
    """
    提取数据包的头部和载荷字节

    Args:
        l3: 解析后的 IP 层对象
        raw_buf: 原始数据包字节

    Returns:
        BYTES_PER_PACKET 字节的数据（不足则填充 0）
    """
    result = bytearray(BYTES_PER_PACKET)

    # 提取 IP 头部（最多 HEADER_LEN 字节）
    if isinstance(l3, dpkt.ip.IP):
        ip_header = bytes(l3.pack_hdr())
    elif isinstance(l3, dpkt.ip6.IP6):
        ip_header = bytes(l3.pack_hdr())
    else:
        ip_header = b''

    # 提取传输层头部
    l4_header = b''
    payload = b''

    if hasattr(l3, 'data') and l3.data is not None:
        l4 = l3.data
        if isinstance(l4, dpkt.tcp.TCP):
            l4_header = bytes(l4.pack_hdr())
            payload = bytes(l4.data) if l4.data else b''
        elif isinstance(l4, dpkt.udp.UDP):
            l4_header = bytes(l4.pack_hdr())
            payload = bytes(l4.data) if l4.data else b''

    # 组合头部
    header = ip_header + l4_header
    header = header[:HEADER_LEN]  # 截断到 160 字节

    # 载荷
    payload = payload[:PAYLOAD_LEN]  # 截断到 480 字节

    # 填充到 result
    result[:len(header)] = header
    result[HEADER_LEN:HEADER_LEN + len(payload)] = payload

    return bytes(result)


def bytes_to_mfr_image(packet_bytes_list: List[bytes]) -> np.ndarray:
    """
    将多个数据包的字节转换为 MFR 图像

    Args:
        packet_bytes_list: 数据包字节列表（每个 320 字节）

    Returns:
        40×40 的灰度图像 (uint8)
    """
    # 确保有足够的数据包
    while len(packet_bytes_list) < NUM_PACKETS:
        packet_bytes_list.append(bytes(BYTES_PER_PACKET))

    # 只取前 NUM_PACKETS 个
    packet_bytes_list = packet_bytes_list[:NUM_PACKETS]

    # 拼接所有字节
    all_bytes = b''.join(packet_bytes_list)

    # 确保长度正确
    if len(all_bytes) < TOTAL_BYTES:
        all_bytes = all_bytes + bytes(TOTAL_BYTES - len(all_bytes))
    else:
        all_bytes = all_bytes[:TOTAL_BYTES]

    # 转换为 numpy 数组并 reshape
    arr = np.frombuffer(all_bytes, dtype=np.uint8)
    image = arr.reshape(IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_flows_from_pcap(pcap_path: str) -> List[np.ndarray]:
    """
    从单个 PCAP 提取所有流的 MFR 图像

    返回: List[np.ndarray]，每个元素是 40×40 的 uint8 图像
    """
    pcap_path = Path(pcap_path)

    if pcap_path.stat().st_size < MIN_PCAP_SIZE:
        return []

    # 按五元组聚合流: key -> {"packets": [bytes, ...]}
    flows: Dict[tuple, Dict] = {}

    for ts, buf in _iter_packets(pcap_path):
        l3, raw_buf = _parse_l3(buf)
        if l3 is None:
            continue

        proto_str = None
        sport = dport = None

        if isinstance(l3, dpkt.ip.IP):
            if l3.p == dpkt.ip.IP_PROTO_TCP and isinstance(l3.data, dpkt.tcp.TCP):
                proto_str = "TCP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
            elif l3.p == dpkt.ip.IP_PROTO_UDP and isinstance(l3.data, dpkt.udp.UDP):
                if int(l3.data.sport) in EXCLUDE_UDP_PORTS or int(l3.data.dport) in EXCLUDE_UDP_PORTS:
                    continue
                proto_str = "UDP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
        elif isinstance(l3, dpkt.ip6.IP6):
            if l3.nxt == dpkt.ip.IP_PROTO_TCP and isinstance(l3.data, dpkt.tcp.TCP):
                proto_str = "TCP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
            elif l3.nxt == dpkt.ip.IP_PROTO_UDP and isinstance(l3.data, dpkt.udp.UDP):
                if int(l3.data.sport) in EXCLUDE_UDP_PORTS or int(l3.data.dport) in EXCLUDE_UDP_PORTS:
                    continue
                proto_str = "UDP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)

        if proto_str is None:
            continue

        # 计算无向键
        src_ip = ipaddress.ip_address(l3.src)
        dst_ip = ipaddress.ip_address(l3.dst)
        a = (int(src_ip), sport)
        b = (int(dst_ip), dport)

        if a <= b:
            key = (proto_str, (src_ip.compressed, sport), (dst_ip.compressed, dport))
        else:
            key = (proto_str, (dst_ip.compressed, dport), (src_ip.compressed, sport))

        if key not in flows:
            flows[key] = {"packets": []}

        # 只收集前 NUM_PACKETS 个数据包
        if len(flows[key]["packets"]) < NUM_PACKETS:
            pkt_bytes = extract_packet_bytes(l3, raw_buf)
            flows[key]["packets"].append(pkt_bytes)

    # 生成 MFR 图像
    result = []
    for flow_data in flows.values():
        packets = flow_data["packets"]
        if len(packets) >= MIN_PKTS_PER_FLOW:
            image = bytes_to_mfr_image(packets)
            result.append(image)

    return result


def process_single_pcap(args: Tuple[str, str]) -> Tuple[str, str, List[np.ndarray]]:
    """处理单个 PCAP 文件（用于多进程）"""
    pcap_path, label = args
    try:
        images = extract_flows_from_pcap(pcap_path)
        return (pcap_path, label, images)
    except Exception as e:
        return (pcap_path, label, [])


def find_pcaps(root: Path) -> Dict[str, List[str]]:
    """扫描目录，返回 {label: [pcap_paths]}"""
    label_to_pcaps: Dict[str, List[str]] = defaultdict(list)

    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for pcap_path in sorted(label_dir.rglob("*")):
            if pcap_path.is_file() and pcap_path.suffix.lower() in (".pcap", ".pcapng"):
                label_to_pcaps[label].append(str(pcap_path))

    return dict(label_to_pcaps)


def main():
    print(f"[YaTC VPN Data Processor]")
    print(f"输入目录: {ROOT}")
    print(f"输出目录: {OUT_ROOT}")
    print(f"进程数: {MAX_PROCS}")
    print(f"MFR 配置: {NUM_PACKETS} 包 × {BYTES_PER_PACKET} 字节 = {TOTAL_BYTES} 字节 -> {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("=" * 60)

    # 扫描 PCAP 文件
    label_to_pcaps = find_pcaps(ROOT)
    if not label_to_pcaps:
        print("[错误] 未找到任何 PCAP 文件")
        return

    labels = sorted(label_to_pcaps.keys())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"发现 {len(labels)} 个类别:")
    for label in labels:
        print(f"  - {label}: {len(label_to_pcaps[label])} 个 PCAP 文件")
    print()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 准备任务
    all_tasks = []
    for label, pcaps in label_to_pcaps.items():
        for pcap_path in pcaps:
            all_tasks.append((pcap_path, label))

    print(f"总计 {len(all_tasks)} 个 PCAP 文件待处理")

    # 多进程处理
    label_images: Dict[str, List[np.ndarray]] = defaultdict(list)

    with mp.Pool(processes=MAX_PROCS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_pcap, all_tasks),
            total=len(all_tasks),
            desc="处理 PCAP"
        ))

    for pcap_path, label, images in results:
        if images:
            label_images[label].extend(images)

    # 保存每个 label 的 NPZ
    print("\n保存 NPZ 文件:")
    total_images = 0

    for label in labels:
        images = label_images.get(label, [])
        if not images:
            print(f"  [跳过] {label}: 无有效流")
            continue

        label_id = label2id[label]
        out_path = OUT_ROOT / f"{label}.npz"

        images_arr = np.stack(images, axis=0)
        np.savez_compressed(
            out_path,
            images=images_arr,
            label=label,
            label_id=label_id
        )
        print(f"  [保存] {label}: {len(images)} 个 MFR 图像 (shape: {images_arr.shape}) -> {out_path.name}")
        total_images += len(images)

    print(f"\n[总计] {total_images} 个 MFR 图像")

    # 保存标签映射
    labels_json_path = OUT_ROOT / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "mfr_config": {
                "num_packets": NUM_PACKETS,
                "header_len": HEADER_LEN,
                "payload_len": PAYLOAD_LEN,
                "bytes_per_packet": BYTES_PER_PACKET,
                "total_bytes": TOTAL_BYTES,
                "image_size": IMAGE_SIZE
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"[标签] -> {labels_json_path.name}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  类别数: {len(labels)}")
    print(f"  总图像数: {total_images}")
    print(f"  图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}")
    for label in labels:
        count = len(label_images.get(label, []))
        print(f"  - {label}: {count} 个图像")


if __name__ == "__main__":
    main()
