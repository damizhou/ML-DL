#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_vpn_processor.py - 统一 VPN 数据处理脚本

功能：
- 只读取一次 PCAP 文件，同时为四个模型生成数据
- 大幅减少网络数据传输时间（适用于远程数据服务器）
- 支持多进程并行处理

输出：
- FS-Net:            数据包长度序列 (±int16)
- DeepFingerprinting: 方向序列 (±1 int8)
- AppScanner:         54维统计特征 (float32)
- YaTC:               MFR图像 (40×40 uint8)

使用方法：
    python unified_vpn_processor.py --root /path/to/vpn/data --output ./output
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import ipaddress
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, NamedTuple
from collections import defaultdict
from dataclasses import dataclass, field
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

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class Config:
    """统一配置"""
    # 输入输出
    root: Path = Path("/netdisk/dataset/vpn/data")
    output: Path = Path(".")  # 项目根目录，数据将放到各模型的 vpn_data/ 子目录

    # 并行处理
    max_procs: int = 16

    # 通用过滤
    min_pcap_size: int = 20 * 1024  # 20KB
    exclude_udp_ports: set = field(default_factory=lambda: {5353})

    # FS-Net 参数
    fsnet_min_pkts: int = 10
    fsnet_max_seq_len: int = 100
    fsnet_max_pkt_len: int = 1500

    # DeepFingerprinting 参数
    df_min_pkts: int = 20
    df_max_seq_len: int = 5000

    # AppScanner 参数
    appscanner_min_pkts: int = 7
    appscanner_max_pkts: int = 260
    appscanner_percentiles: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])

    # YaTC 参数 (5包 × 320字节 = 1600字节 = 40×40)
    yatc_num_packets: int = 5
    yatc_header_len: int = 40      # IP + TCP/UDP 头部
    yatc_payload_len: int = 280    # 载荷
    yatc_image_size: int = 40


# ============================================================================
# 数据包解析（只读取一次）
# ============================================================================

@dataclass
class PacketInfo:
    """解析后的数据包信息"""
    timestamp: float
    length: int           # 数据包总长度
    direction: int        # +1 出站, -1 入站
    ip_header: bytes      # IP 头部
    l4_header: bytes      # L4 头部
    payload: bytes        # 载荷
    flow_key: tuple       # 五元组 key


def parse_l3(buf: bytes):
    """解析 L3 层"""
    # Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except:
        pass

    # Linux SLL
    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except:
        pass

    # RAW IP
    try:
        ver = buf[0] >> 4
        if ver == 4:
            return dpkt.ip.IP(buf)
        elif ver == 6:
            return dpkt.ip6.IP6(buf)
    except:
        pass

    return None


def iter_packets(pcap_path: Path) -> Iterable[Tuple[float, bytes]]:
    """迭代 PCAP 文件"""
    with pcap_path.open("rb") as f:
        try:
            f.seek(0)
            reader = dpkt.pcap.Reader(f)
            for ts, buf in reader:
                yield ts, buf
            return
        except:
            pass

        f.seek(0)
        try:
            reader = dpkt.pcapng.Reader(f)
            for rec in reader:
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    yield rec[0], rec[1]
        except:
            pass


def parse_pcap_once(pcap_path: Path, config: Config) -> Dict[tuple, List[PacketInfo]]:
    """
    一次性解析 PCAP，返回按流聚合的数据包信息

    Returns:
        Dict[flow_key, List[PacketInfo]]
    """
    flows: Dict[tuple, List[PacketInfo]] = defaultdict(list)

    for ts, buf in iter_packets(pcap_path):
        l3 = parse_l3(buf)
        if l3 is None:
            continue

        # 提取协议和端口
        proto_str = None
        sport = dport = None
        ip_header = b''
        l4_header = b''
        payload = b''

        if isinstance(l3, dpkt.ip.IP):
            ip_header = bytes(l3.pack_hdr())
            if l3.p == dpkt.ip.IP_PROTO_TCP and isinstance(l3.data, dpkt.tcp.TCP):
                proto_str = "TCP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
                l4_header = bytes(l3.data.pack_hdr())
                payload = bytes(l3.data.data) if l3.data.data else b''
            elif l3.p == dpkt.ip.IP_PROTO_UDP and isinstance(l3.data, dpkt.udp.UDP):
                if int(l3.data.sport) in config.exclude_udp_ports or int(l3.data.dport) in config.exclude_udp_ports:
                    continue
                proto_str = "UDP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
                l4_header = bytes(l3.data.pack_hdr())
                payload = bytes(l3.data.data) if l3.data.data else b''

        elif isinstance(l3, dpkt.ip6.IP6):
            ip_header = bytes(l3.pack_hdr())
            if l3.nxt == dpkt.ip.IP_PROTO_TCP and isinstance(l3.data, dpkt.tcp.TCP):
                proto_str = "TCP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
                l4_header = bytes(l3.data.pack_hdr())
                payload = bytes(l3.data.data) if l3.data.data else b''
            elif l3.nxt == dpkt.ip.IP_PROTO_UDP and isinstance(l3.data, dpkt.udp.UDP):
                if int(l3.data.sport) in config.exclude_udp_ports or int(l3.data.dport) in config.exclude_udp_ports:
                    continue
                proto_str = "UDP"
                sport, dport = int(l3.data.sport), int(l3.data.dport)
                l4_header = bytes(l3.data.pack_hdr())
                payload = bytes(l3.data.data) if l3.data.data else b''

        if proto_str is None:
            continue

        # 计算无向键和方向
        src_ip = ipaddress.ip_address(l3.src)
        dst_ip = ipaddress.ip_address(l3.dst)
        a = (int(src_ip), sport)
        b = (int(dst_ip), dport)

        if a <= b:
            key = (proto_str, (src_ip.compressed, sport), (dst_ip.compressed, dport))
            direction = 1
        else:
            key = (proto_str, (dst_ip.compressed, dport), (src_ip.compressed, sport))
            direction = -1

        pkt_info = PacketInfo(
            timestamp=ts,
            length=len(buf),
            direction=direction,
            ip_header=ip_header,
            l4_header=l4_header,
            payload=payload,
            flow_key=key
        )

        flows[key].append(pkt_info)

    return dict(flows)


# ============================================================================
# 四种特征提取器
# ============================================================================

def extract_fsnet_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 FS-Net 特征：数据包长度序列（带方向）"""
    result = []

    for packets in flows.values():
        if len(packets) < config.fsnet_min_pkts:
            continue

        lengths = []
        for pkt in packets[:config.fsnet_max_seq_len]:
            pkt_len = min(pkt.length, config.fsnet_max_pkt_len)
            lengths.append(pkt.direction * pkt_len)

        result.append(np.array(lengths, dtype=np.int16))

    return result


def extract_df_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 DeepFingerprinting 特征：方向序列"""
    result = []

    for packets in flows.values():
        if len(packets) < config.df_min_pkts:
            continue

        directions = [np.int8(pkt.direction) for pkt in packets[:config.df_max_seq_len]]
        result.append(np.array(directions, dtype=np.int8))

    return result


def compute_statistics(lengths: np.ndarray, percentiles: List[int]) -> np.ndarray:
    """计算 18 维统计特征"""
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
    features.append(stats.skew(lengths) if len(lengths) >= 3 else 0.0)
    features.append(stats.kurtosis(lengths) if len(lengths) >= 4 else 0.0)

    # 9. MAD
    median = np.median(lengths)
    features.append(np.median(np.abs(lengths - median)))

    # 10-18. Percentiles
    for p in percentiles:
        features.append(np.percentile(lengths, p))

    return np.array(features, dtype=np.float32)


def extract_appscanner_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 AppScanner 特征：54维统计特征"""
    result = []

    for packets in flows.values():
        total_pkts = len(packets)
        if total_pkts < config.appscanner_min_pkts:
            continue

        # 分离入站和出站
        incoming = [pkt.length for pkt in packets if pkt.direction == -1]
        outgoing = [pkt.length for pkt in packets if pkt.direction == 1]

        # 截断
        if total_pkts > config.appscanner_max_pkts:
            ratio = len(outgoing) / total_pkts if total_pkts > 0 else 0.5
            max_out = int(config.appscanner_max_pkts * ratio)
            max_in = config.appscanner_max_pkts - max_out
            outgoing = outgoing[:max_out]
            incoming = incoming[:max_in]

        incoming_arr = np.array(incoming, dtype=np.float64)
        outgoing_arr = np.array(outgoing, dtype=np.float64)
        bidirectional_arr = np.concatenate([incoming_arr, outgoing_arr])

        # 计算三个方向的特征
        in_features = compute_statistics(incoming_arr, config.appscanner_percentiles)
        out_features = compute_statistics(outgoing_arr, config.appscanner_percentiles)
        bi_features = compute_statistics(bidirectional_arr, config.appscanner_percentiles)

        result.append(np.concatenate([in_features, out_features, bi_features]))

    return result


def extract_yatc_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 YaTC 特征：MFR 图像"""
    result = []
    bytes_per_packet = config.yatc_header_len + config.yatc_payload_len
    total_bytes = config.yatc_num_packets * bytes_per_packet

    for packets in flows.values():
        if len(packets) < config.yatc_num_packets:
            continue

        # 提取前 N 个数据包的字节
        all_bytes = bytearray(total_bytes)
        offset = 0

        for pkt in packets[:config.yatc_num_packets]:
            # 头部
            header = pkt.ip_header + pkt.l4_header
            header = header[:config.yatc_header_len]
            all_bytes[offset:offset + len(header)] = header

            # 载荷
            payload = pkt.payload[:config.yatc_payload_len]
            payload_offset = offset + config.yatc_header_len
            all_bytes[payload_offset:payload_offset + len(payload)] = payload

            offset += bytes_per_packet

        # 转换为图像
        arr = np.frombuffer(bytes(all_bytes), dtype=np.uint8)
        image = arr.reshape(config.yatc_image_size, config.yatc_image_size)
        result.append(image)

    return result


# ============================================================================
# 处理结果容器
# ============================================================================

@dataclass
class FlowFeatures:
    """单个 PCAP 提取的四种特征"""
    fsnet: List[np.ndarray] = field(default_factory=list)
    df: List[np.ndarray] = field(default_factory=list)
    appscanner: List[np.ndarray] = field(default_factory=list)
    yatc: List[np.ndarray] = field(default_factory=list)


def process_single_pcap(args: Tuple[str, str, Config]) -> Tuple[str, str, FlowFeatures]:
    """处理单个 PCAP 文件，一次读取提取四种特征"""
    pcap_path, label, config = args

    features = FlowFeatures()

    try:
        pcap_path = Path(pcap_path)

        # 跳过过小的文件
        if pcap_path.stat().st_size < config.min_pcap_size:
            return (str(pcap_path), label, features)

        # 一次性解析 PCAP
        flows = parse_pcap_once(pcap_path, config)

        if not flows:
            return (str(pcap_path), label, features)

        # 提取四种特征
        features.fsnet = extract_fsnet_features(flows, config)
        features.df = extract_df_features(flows, config)
        features.appscanner = extract_appscanner_features(flows, config)
        features.yatc = extract_yatc_features(flows, config)

    except Exception as e:
        pass

    return (str(pcap_path), label, features)


# ============================================================================
# 主程序
# ============================================================================

def find_pcaps(root: Path) -> Dict[str, List[str]]:
    """扫描目录，返回 {label: [pcap_paths]}"""
    label_to_pcaps: Dict[str, List[str]] = defaultdict(list)

    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for pcap_path in sorted(label_dir.rglob("*")):
            if pcap_path.is_file() and pcap_path.suffix.lower() in (".pcap", ".pcapng"):
                label_to_pcaps[label].append(str(pcap_path))

    return dict(label_to_pcaps)


def save_model_data(
    model_name: str,
    output_dir: Path,
    label_data: Dict[str, List[np.ndarray]],
    labels: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    data_key: str = "data",
    extra_meta: Dict = None
):
    """保存单个模型的数据（每个类别一个 NPZ）"""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0

    print(f"\n[{model_name}] 保存数据:")

    for label in labels:
        data = label_data.get(label, [])
        if not data:
            print(f"  [跳过] {label}: 无数据")
            continue

        label_id = label2id[label]
        out_path = model_dir / f"{label}.npz"

        if data_key == "features":
            # AppScanner: 堆叠为矩阵
            data_arr = np.stack(data, axis=0)
            np.savez_compressed(out_path, features=data_arr, label=label, label_id=label_id)
            print(f"  [保存] {label}: {len(data)} 条 (shape: {data_arr.shape}) -> {out_path.name}")
        elif data_key == "images":
            # YaTC: 堆叠为图像
            data_arr = np.stack(data, axis=0)
            np.savez_compressed(out_path, images=data_arr, label=label, label_id=label_id)
            print(f"  [保存] {label}: {len(data)} 个图像 (shape: {data_arr.shape}) -> {out_path.name}")
        else:
            # FS-Net / DF: object 数组
            np.savez_compressed(
                out_path,
                **{data_key: np.array(data, dtype=object)},
                label=label,
                label_id=label_id
            )
            print(f"  [保存] {label}: {len(data)} 条 -> {out_path.name}")

        total_count += len(data)

    # 保存标签映射
    labels_json = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()}
    }
    if extra_meta:
        labels_json.update(extra_meta)

    with open(model_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    print(f"  [总计] {total_count} 条")

    return total_count


def main():
    parser = argparse.ArgumentParser(description="统一 VPN 数据处理脚本")
    parser.add_argument("--root", type=str, default="/netdisk/dataset/vpn/data",
                        help="VPN 数据集根目录")
    parser.add_argument("--output", type=str, default="./vpn_unified_output",
                        help="输出目录")
    parser.add_argument("--procs", type=int, default=16,
                        help="并行进程数")
    parser.add_argument("--max_labels", type=int, default=10,
                        help="最大处理标签数（0=不限制）")
    args = parser.parse_args()

    config = Config()
    config.root = Path(args.root)
    config.output = Path(args.output)
    config.max_procs = args.procs

    print("=" * 70)
    print("统一 VPN 数据处理脚本")
    print("=" * 70)
    print(f"输入目录: {config.root}")
    print(f"输出目录: {config.output}")
    print(f"并行进程数: {config.max_procs}")
    if args.max_labels > 0:
        print(f"标签数限制: {args.max_labels}")
    print()
    print("将为以下模型生成数据:")
    print("  1. FS-Net:            数据包长度序列 (±int16)")
    print("  2. DeepFingerprinting: 方向序列 (±1 int8)")
    print("  3. AppScanner:         54维统计特征 (float32)")
    print("  4. YaTC:               MFR图像 (40×40 uint8)")
    print("=" * 70)

    # 扫描 PCAP 文件
    label_to_pcaps = find_pcaps(config.root)
    if not label_to_pcaps:
        print("[错误] 未找到任何 PCAP 文件")
        return

    labels = sorted(label_to_pcaps.keys())

    # 限制标签数量
    if args.max_labels > 0 and len(labels) > args.max_labels:
        print(f"\n[限制] 只处理前 {args.max_labels} 个标签（共 {len(labels)} 个）")
        labels = labels[:args.max_labels]
        label_to_pcaps = {k: v for k, v in label_to_pcaps.items() if k in labels}

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"\n发现 {len(labels)} 个类别:")
    total_pcaps = 0
    for label in labels:
        count = len(label_to_pcaps[label])
        total_pcaps += count
        print(f"  - {label}: {count} 个 PCAP 文件")

    print(f"\n总计 {total_pcaps} 个 PCAP 文件待处理")

    # 各模型的输出目录（输出到各模型项目目录下）
    # 格式: <project_root>/<ModelName>/<output_subdir>/
    project_root = config.output.parent if config.output.name != "." else config.output
    output_subdir = config.output.name if config.output.name != "." else "vpn_unified_output"
    OUTPUT_DIRS = {
        "fsnet": project_root / "FS-Net" / output_subdir,
        "deepfingerprinting": project_root / "DeepFingerprinting" / output_subdir,
        "appscanner": project_root / "AppScanner" / output_subdir,
        "yatc": project_root / "YaTC" / output_subdir,
    }

    # 创建各模型输出目录
    for out_dir in OUTPUT_DIRS.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    fsnet_count = 0
    df_count = 0
    appscanner_count = 0
    yatc_count = 0

    print("\n开始处理（按 label 分批，节省内存）...")
    print("=" * 70)

    # 按 label 逐个处理，处理完立即保存并释放内存
    for label in tqdm(labels, desc="处理 Label"):
        pcaps = label_to_pcaps[label]
        label_id = label2id[label]

        # 准备该 label 的任务
        tasks = [(pcap_path, label, config) for pcap_path in pcaps]

        # 该 label 的特征
        fsnet_data: List[np.ndarray] = []
        df_data: List[np.ndarray] = []
        appscanner_data: List[np.ndarray] = []
        yatc_data: List[np.ndarray] = []

        # 多进程处理该 label 的所有 PCAP
        with mp.Pool(processes=config.max_procs) as pool:
            results = list(pool.imap_unordered(process_single_pcap, tasks))

        # 聚合该 label 的结果
        for pcap_path, _, features in results:
            if features.fsnet:
                fsnet_data.extend(features.fsnet)
            if features.df:
                df_data.extend(features.df)
            if features.appscanner:
                appscanner_data.extend(features.appscanner)
            if features.yatc:
                yatc_data.extend(features.yatc)

        # 立即保存该 label 的数据
        # FS-Net
        if fsnet_data:
            out_path = OUTPUT_DIRS["fsnet"] / f"{label}.npz"
            np.savez_compressed(out_path, sequences=np.array(fsnet_data, dtype=object), label=label, label_id=label_id)
            fsnet_count += len(fsnet_data)

        # DeepFingerprinting
        if df_data:
            out_path = OUTPUT_DIRS["deepfingerprinting"] / f"{label}.npz"
            np.savez_compressed(out_path, flows=np.array(df_data, dtype=object), label=label, label_id=label_id)
            df_count += len(df_data)

        # AppScanner
        if appscanner_data:
            out_path = OUTPUT_DIRS["appscanner"] / f"{label}.npz"
            np.savez_compressed(out_path, features=np.stack(appscanner_data, axis=0), label=label, label_id=label_id)
            appscanner_count += len(appscanner_data)

        # YaTC
        if yatc_data:
            out_path = OUTPUT_DIRS["yatc"] / f"{label}.npz"
            np.savez_compressed(out_path, images=np.stack(yatc_data, axis=0), label=label, label_id=label_id)
            yatc_count += len(yatc_data)

        # 打印该 label 的统计
        tqdm.write(f"  {label}: FS-Net={len(fsnet_data)}, DF={len(df_data)}, AppScanner={len(appscanner_data)}, YaTC={len(yatc_data)}")

        # 释放内存
        del fsnet_data, df_data, appscanner_data, yatc_data, results

    # 保存标签映射
    print("\n保存标签映射...")
    extra_metas = {
        "fsnet": None,
        "deepfingerprinting": None,
        "appscanner": {
            "feature_names": [
                "in_count", "in_min", "in_max", "in_mean", "in_std", "in_var",
                "in_skew", "in_kurt", "in_mad",
                "in_p10", "in_p20", "in_p30", "in_p40", "in_p50", "in_p60", "in_p70", "in_p80", "in_p90",
                "out_count", "out_min", "out_max", "out_mean", "out_std", "out_var",
                "out_skew", "out_kurt", "out_mad",
                "out_p10", "out_p20", "out_p30", "out_p40", "out_p50", "out_p60", "out_p70", "out_p80", "out_p90",
                "bi_count", "bi_min", "bi_max", "bi_mean", "bi_std", "bi_var",
                "bi_skew", "bi_kurt", "bi_mad",
                "bi_p10", "bi_p20", "bi_p30", "bi_p40", "bi_p50", "bi_p60", "bi_p70", "bi_p80", "bi_p90",
            ]
        },
        "yatc": {
            "mfr_config": {
                "num_packets": config.yatc_num_packets,
                "header_len": config.yatc_header_len,
                "payload_len": config.yatc_payload_len,
                "image_size": config.yatc_image_size
            }
        }
    }
    for model_name, out_dir in OUTPUT_DIRS.items():
        labels_json = {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()}
        }
        extra_meta = extra_metas.get(model_name)
        if extra_meta:
            labels_json.update(extra_meta)
        with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels_json, f, ensure_ascii=False, indent=2)

    # 总结
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"输出目录: {project_root}/<ModelName>/{output_subdir}/")
    print(f"\n各模型数据统计:")
    print(f"  FS-Net:            {fsnet_count:>8} 条流")
    print(f"  DeepFingerprinting: {df_count:>8} 条流")
    print(f"  AppScanner:         {appscanner_count:>8} 条流")
    print(f"  YaTC:               {yatc_count:>8} 个图像")
    print(f"\n目录结构:")
    print(f"  {project_root}/")
    print(f"  ├── FS-Net/{output_subdir}/")
    print(f"  │   ├── <label>.npz")
    print(f"  │   └── labels.json")
    print(f"  ├── DeepFingerprinting/{output_subdir}/")
    print(f"  │   └── ...")
    print(f"  ├── AppScanner/{output_subdir}/")
    print(f"  │   └── ...")
    print(f"  └── YaTC/{output_subdir}/")
    print(f"      └── ...")


if __name__ == "__main__":
    main()
