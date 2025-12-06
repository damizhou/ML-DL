#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpn_build_npz.py - DeepFingerprinting VPN 数据处理

功能：
- 递归扫描 ROOT 目录（按子目录名作为 label）下的 .pcap / .pcapng
- 多进程并行处理 PCAP 文件
- 提取流量方向序列（+1=出站，-1=入站）
- 每个 label 的所有流合并为一个 NPZ 文件
- 输出格式兼容 DeepFingerprinting 训练脚本

输出 NPZ 结构：
    - flows: object 数组，元素为变长 int8 数组（±1 方向序列）
    - labels: object 数组，字符串标签（用于单个 label 文件）
    - 或 labels: int64 数组（用于合并文件）
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
OUT_ROOT = Path(__file__).resolve().parent / "vpn_df_data"  # 输出目录
MAX_PROCS = 16  # 并发进程数
MIN_PKTS_PER_FLOW = 20  # 最小数据包数
MAX_SEQ_LEN = 5000  # 最大序列长度（截断，与模型输入一致）
EXCLUDE_UDP_PORTS = {5353}  # 排除 mDNS
MIN_PCAP_SIZE = 20 * 1024  # 最小 PCAP 文件大小 (20KB)
# ==========================


def _parse_l3(buf: bytes):
    """解析 L3 层（IPv4/IPv6），兼容 Ethernet/SLL/RAW"""
    # Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except Exception:
        pass

    # Linux SLL
    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except Exception:
        pass

    # RAW IP
    try:
        ver = buf[0] >> 4
        if ver == 4:
            return dpkt.ip.IP(buf)
        elif ver == 6:
            return dpkt.ip6.IP6(buf)
    except Exception:
        pass

    return None


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


def extract_flows_from_pcap(pcap_path: str) -> List[np.ndarray]:
    """
    从单个 PCAP 提取所有流的方向序列

    返回: List[np.ndarray]，每个元素是一个流的方向序列（int8，±1）
    """
    pcap_path = Path(pcap_path)

    # 跳过过小的文件
    if pcap_path.stat().st_size < MIN_PCAP_SIZE:
        return []

    # 按五元组聚合流: key -> List[int8]
    flows: Dict[tuple, List[np.int8]] = {}

    for ts, buf in _iter_packets(pcap_path):
        l3 = _parse_l3(buf)
        if l3 is None:
            continue

        # 提取协议和端口
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

        # 计算无向键和方向
        src_ip = ipaddress.ip_address(l3.src)
        dst_ip = ipaddress.ip_address(l3.dst)
        a = (int(src_ip), sport)
        b = (int(dst_ip), dport)

        if a <= b:
            key = (proto_str, (src_ip.compressed, sport), (dst_ip.compressed, dport))
            sign = np.int8(+1)
        else:
            key = (proto_str, (dst_ip.compressed, dport), (src_ip.compressed, sport))
            sign = np.int8(-1)

        # 初始化流
        if key not in flows:
            flows[key] = []

        flows[key].append(sign)

    # 过滤并转换
    result = []
    for directions in flows.values():
        if len(directions) >= MIN_PKTS_PER_FLOW:
            # 截断到最大长度
            directions = directions[:MAX_SEQ_LEN]
            result.append(np.array(directions, dtype=np.int8))

    return result


def process_single_pcap(args: Tuple[str, str]) -> Tuple[str, str, List[np.ndarray]]:
    """处理单个 PCAP 文件（用于多进程）"""
    pcap_path, label = args
    try:
        flows = extract_flows_from_pcap(pcap_path)
        return (pcap_path, label, flows)
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
    print(f"[DeepFingerprinting VPN Data Processor]")
    print(f"输入目录: {ROOT}")
    print(f"输出目录: {OUT_ROOT}")
    print(f"进程数: {MAX_PROCS}")
    print(f"最小流长度: {MIN_PKTS_PER_FLOW} 包")
    print(f"最大序列长度: {MAX_SEQ_LEN}")
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

    # 创建输出目录
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 准备任务
    all_tasks = []
    for label, pcaps in label_to_pcaps.items():
        for pcap_path in pcaps:
            all_tasks.append((pcap_path, label))

    print(f"总计 {len(all_tasks)} 个 PCAP 文件待处理")

    # 多进程处理
    label_flows: Dict[str, List[np.ndarray]] = defaultdict(list)

    with mp.Pool(processes=MAX_PROCS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_pcap, all_tasks),
            total=len(all_tasks),
            desc="处理 PCAP"
        ))

    # 按 label 聚合
    for pcap_path, label, flows in results:
        if flows:
            label_flows[label].extend(flows)

    # 保存每个 label 的 NPZ
    print("\n保存 NPZ 文件:")
    total_flows = 0

    for label in labels:
        flows = label_flows.get(label, [])
        if not flows:
            print(f"  [跳过] {label}: 无有效流")
            continue

        label_id = label2id[label]
        out_path = OUT_ROOT / f"{label}.npz"

        # 保存单个 label 的 NPZ（兼容原有格式）
        np.savez_compressed(
            out_path,
            flows=np.array(flows, dtype=object),
            labels=np.array([label] * len(flows), dtype=object),
            label=label,
            label_id=label_id
        )
        print(f"  [保存] {label}: {len(flows)} 条流 -> {out_path.name}")
        total_flows += len(flows)

    print(f"\n[总计] {total_flows} 条流")

    # 保存标签映射
    labels_json_path = OUT_ROOT / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"[标签] -> {labels_json_path.name}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  类别数: {len(labels)}")
    print(f"  总流数: {total_flows}")
    for label in labels:
        count = len(label_flows.get(label, []))
        print(f"  - {label}: {count} 条流")


if __name__ == "__main__":
    main()
