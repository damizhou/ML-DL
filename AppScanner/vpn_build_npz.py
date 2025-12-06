#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpn_build_npz.py - AppScanner VPN 数据处理

功能：
- 递归扫描 ROOT 目录（按子目录名作为 label）下的 .pcap / .pcapng
- 多进程并行处理 PCAP 文件
- 提取 54 维统计特征（入站/出站/双向各 18 个特征）
- 每个 label 的所有流合并为一个 NPZ 文件
- 输出格式兼容 AppScanner 训练脚本

统计特征 (每个方向 18 维，共 54 维)：
    - 数据包计数 (1)
    - Min, Max, Mean, Std, Var (5)
    - Skewness, Kurtosis (2)
    - MAD - 中位数绝对偏差 (1)
    - Percentiles: 10%, 20%, ..., 90% (9)

输出 NPZ 结构：
    - features: float32 数组 (N, 54)
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
import warnings

import numpy as np
from scipy import stats

try:
    import dpkt
except ImportError:
    raise SystemExit("需要安装 dpkt: pip install dpkt")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# 忽略统计计算警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ========== 配置 ==========
ROOT = Path("/netdisk/dataset/vpn/data")  # 输入：一级子目录名为标签
OUT_ROOT = Path(__file__).resolve().parent / "vpn_appscanner_data"  # 输出目录
MAX_PROCS = 16  # 并发进程数
MIN_PKTS_PER_FLOW = 7  # 最小数据包数（论文参数）
MAX_PKTS_PER_FLOW = 260  # 最大数据包数（论文参数）
EXCLUDE_UDP_PORTS = {5353}  # 排除 mDNS
MIN_PCAP_SIZE = 20 * 1024  # 最小 PCAP 文件大小 (20KB)
PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # 分位数
# ==========================


def _parse_l3(buf: bytes):
    """解析 L3 层（IPv4/IPv6），兼容 Ethernet/SLL/RAW"""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except Exception:
        pass

    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except Exception:
        pass

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


def compute_statistics(lengths: np.ndarray) -> np.ndarray:
    """
    计算 18 维统计特征

    Args:
        lengths: 数据包长度数组

    Returns:
        18 维特征向量
    """
    if len(lengths) == 0:
        return np.zeros(18, dtype=np.float32)

    lengths = lengths.astype(np.float64)

    features = []

    # 1. 数据包计数
    features.append(len(lengths))

    # 2-6. Min, Max, Mean, Std, Var
    features.append(np.min(lengths))
    features.append(np.max(lengths))
    features.append(np.mean(lengths))
    features.append(np.std(lengths))
    features.append(np.var(lengths))

    # 7-8. Skewness, Kurtosis
    if len(lengths) >= 3:
        features.append(stats.skew(lengths))
    else:
        features.append(0.0)

    if len(lengths) >= 4:
        features.append(stats.kurtosis(lengths))
    else:
        features.append(0.0)

    # 9. MAD (Median Absolute Deviation)
    median = np.median(lengths)
    features.append(np.median(np.abs(lengths - median)))

    # 10-18. Percentiles (10%, 20%, ..., 90%)
    for p in PERCENTILES:
        features.append(np.percentile(lengths, p))

    return np.array(features, dtype=np.float32)


def extract_flow_features(incoming: List[int], outgoing: List[int]) -> Optional[np.ndarray]:
    """
    提取单个流的 54 维统计特征

    Args:
        incoming: 入站数据包长度列表
        outgoing: 出站数据包长度列表

    Returns:
        54 维特征向量，或 None（如果流太短）
    """
    total_pkts = len(incoming) + len(outgoing)

    # 过滤过短或过长的流
    if total_pkts < MIN_PKTS_PER_FLOW:
        return None
    if total_pkts > MAX_PKTS_PER_FLOW:
        # 截断
        ratio = len(outgoing) / total_pkts if total_pkts > 0 else 0.5
        max_out = int(MAX_PKTS_PER_FLOW * ratio)
        max_in = MAX_PKTS_PER_FLOW - max_out
        outgoing = outgoing[:max_out]
        incoming = incoming[:max_in]

    # 转换为数组
    incoming_arr = np.array(incoming, dtype=np.float64)
    outgoing_arr = np.array(outgoing, dtype=np.float64)
    bidirectional_arr = np.concatenate([incoming_arr, outgoing_arr])

    # 计算三个方向的特征
    incoming_features = compute_statistics(incoming_arr)
    outgoing_features = compute_statistics(outgoing_arr)
    bidirectional_features = compute_statistics(bidirectional_arr)

    # 拼接为 54 维
    return np.concatenate([incoming_features, outgoing_features, bidirectional_features])


def extract_flows_from_pcap(pcap_path: str) -> List[np.ndarray]:
    """
    从单个 PCAP 提取所有流的特征

    返回: List[np.ndarray]，每个元素是 54 维特征向量
    """
    pcap_path = Path(pcap_path)

    if pcap_path.stat().st_size < MIN_PCAP_SIZE:
        return []

    # 按五元组聚合流: key -> {"incoming": [], "outgoing": [], "origin": ...}
    flows: Dict[tuple, Dict] = {}

    for ts, buf in _iter_packets(pcap_path):
        l3 = _parse_l3(buf)
        if l3 is None:
            continue

        proto_str = None
        sport = dport = None
        pkt_len = len(buf)

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
            is_outgoing = True
        else:
            key = (proto_str, (dst_ip.compressed, dport), (src_ip.compressed, sport))
            is_outgoing = False

        if key not in flows:
            flows[key] = {"incoming": [], "outgoing": []}

        if is_outgoing:
            flows[key]["outgoing"].append(pkt_len)
        else:
            flows[key]["incoming"].append(pkt_len)

    # 提取特征
    result = []
    for flow_data in flows.values():
        features = extract_flow_features(flow_data["incoming"], flow_data["outgoing"])
        if features is not None:
            result.append(features)

    return result


def process_single_pcap(args: Tuple[str, str]) -> Tuple[str, str, List[np.ndarray]]:
    """处理单个 PCAP 文件（用于多进程）"""
    pcap_path, label = args
    try:
        features = extract_flows_from_pcap(pcap_path)
        return (pcap_path, label, features)
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
    print(f"[AppScanner VPN Data Processor]")
    print(f"输入目录: {ROOT}")
    print(f"输出目录: {OUT_ROOT}")
    print(f"进程数: {MAX_PROCS}")
    print(f"流长度范围: {MIN_PKTS_PER_FLOW} - {MAX_PKTS_PER_FLOW} 包")
    print(f"特征维度: 54 (18 × 3 方向)")
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
    label_features: Dict[str, List[np.ndarray]] = defaultdict(list)

    with mp.Pool(processes=MAX_PROCS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_pcap, all_tasks),
            total=len(all_tasks),
            desc="处理 PCAP"
        ))

    for pcap_path, label, features in results:
        if features:
            label_features[label].extend(features)

    # 保存每个 label 的 NPZ
    print("\n保存 NPZ 文件:")
    total_flows = 0

    for label in labels:
        features = label_features.get(label, [])
        if not features:
            print(f"  [跳过] {label}: 无有效流")
            continue

        label_id = label2id[label]
        out_path = OUT_ROOT / f"{label}.npz"

        features_arr = np.stack(features, axis=0)
        np.savez_compressed(
            out_path,
            features=features_arr,
            label=label,
            label_id=label_id
        )
        print(f"  [保存] {label}: {len(features)} 条流 (shape: {features_arr.shape}) -> {out_path.name}")
        total_flows += len(features)

    print(f"\n[总计] {total_flows} 条流")

    # 保存标签映射
    labels_json_path = OUT_ROOT / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "feature_names": [
                # Incoming features
                "in_count", "in_min", "in_max", "in_mean", "in_std", "in_var",
                "in_skew", "in_kurt", "in_mad",
                "in_p10", "in_p20", "in_p30", "in_p40", "in_p50", "in_p60", "in_p70", "in_p80", "in_p90",
                # Outgoing features
                "out_count", "out_min", "out_max", "out_mean", "out_std", "out_var",
                "out_skew", "out_kurt", "out_mad",
                "out_p10", "out_p20", "out_p30", "out_p40", "out_p50", "out_p60", "out_p70", "out_p80", "out_p90",
                # Bidirectional features
                "bi_count", "bi_min", "bi_max", "bi_mean", "bi_std", "bi_var",
                "bi_skew", "bi_kurt", "bi_mad",
                "bi_p10", "bi_p20", "bi_p30", "bi_p40", "bi_p50", "bi_p60", "bi_p70", "bi_p80", "bi_p90",
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"[标签] -> {labels_json_path.name}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  类别数: {len(labels)}")
    print(f"  总流数: {total_flows}")
    print(f"  特征维度: 54")
    for label in labels:
        count = len(label_features.get(label, []))
        print(f"  - {label}: {count} 条流")


if __name__ == "__main__":
    main()
