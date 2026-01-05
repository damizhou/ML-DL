#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_ablation_processor.py - 统一消融实验数据处理脚本

功能：
- 只读取一次 PCAP 文件，同时为四个模型生成数据
- 大幅减少网络数据传输时间（适用于远程数据服务器）
- 支持多进程并行处理
- 区分首页(homepage)和子页面(subpage)用于消融实验
- 按 label 分组处理，完成后立即保存释放内存

输出：
- FS-Net:            数据包长度序列 (±int16)
- DeepFingerprinting: 方向序列 (±1 int8)
- AppScanner:         54维统计特征 (float32)
- YaTC:               MFR图像 (40×40 uint8)

目录结构：
/netdisk/dataset/ablation_study/
├── batch/           # 数据集A：连续访问
│   ├── website1/
│   │   └── pcap/
│   │       ├── batch_1.pcap
│   │       └── ...
│   └── website2/
│       └── pcap/
│           └── ...
└── single/          # 数据集B：单独访问
    ├── website1/
    │   └── pcap/
    │       ├── 1_*.pcap       # 首页 (以1_开头)
    │       ├── 2_*.pcap       # 子页面
    │       └── ...
    └── website2/
        └── pcap/
            └── ...

使用方法：
    python unified_ablation_processor.py --root /netdisk/dataset/ablation_study --output .
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import ipaddress
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing as mp
import warnings
import time
import gc
from datetime import datetime

import numpy as np
import psutil
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
    root: Path = Path("/netdisk/dataset/ablation_study")
    output: Path = Path("..")

    # 并行处理
    max_procs: int = 8
    memory_limit: float = 0.8

    # 通用过滤
    min_pcap_size: int = 20 * 1024  # 20KB
    exclude_udp_ports: set = field(default_factory=lambda: {5353})

    # FS-Net 参数
    fsnet_min_pkts: int = 10
    fsnet_max_seq_len: int = 100
    fsnet_max_pkt_len: int = 1500

    # DeepFingerprinting 参数
    df_min_pkts: int = 50
    df_max_seq_len: int = 5000

    # AppScanner 参数
    appscanner_min_pkts: int = 7
    appscanner_max_pkts: int = 260
    appscanner_percentiles: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])

    # YaTC 参数
    yatc_num_packets: int = 5
    yatc_header_len: int = 80
    yatc_payload_len: int = 240
    yatc_image_size: int = 40

    # 首页标识
    homepage_prefix: str = '1_'


# ============================================================================
# 数据包解析
# ============================================================================

@dataclass
class PacketInfo:
    """解析后的数据包信息"""
    timestamp: float
    length: int
    direction: int
    ip_header: bytes
    l4_header: bytes
    payload: bytes
    flow_key: tuple


def parse_l3(buf: bytes):
    """解析 L3 层"""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except:
        pass

    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except:
        pass

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
    """一次性解析 PCAP，返回按流聚合的数据包信息"""
    flows: Dict[tuple, List[PacketInfo]] = defaultdict(list)

    for ts, buf in iter_packets(pcap_path):
        l3 = parse_l3(buf)
        if l3 is None:
            continue

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

        if proto_str is None or len(payload) == 0:
            continue

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
    """提取 FS-Net 特征"""
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
    """提取 DeepFingerprinting 特征"""
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
    features.append(len(lengths))
    features.append(np.min(lengths))
    features.append(np.max(lengths))
    features.append(np.mean(lengths))
    features.append(np.std(lengths))
    features.append(np.var(lengths))
    features.append(stats.skew(lengths) if len(lengths) >= 3 else 0.0)
    features.append(stats.kurtosis(lengths) if len(lengths) >= 4 else 0.0)
    median = np.median(lengths)
    features.append(np.median(np.abs(lengths - median)))
    for p in percentiles:
        features.append(np.percentile(lengths, p))
    return np.array(features, dtype=np.float32)


def extract_appscanner_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 AppScanner 特征"""
    result = []
    for packets in flows.values():
        total_pkts = len(packets)
        if total_pkts < config.appscanner_min_pkts:
            continue

        incoming = [pkt.length for pkt in packets if pkt.direction == -1]
        outgoing = [pkt.length for pkt in packets if pkt.direction == 1]

        if total_pkts > config.appscanner_max_pkts:
            ratio = len(outgoing) / total_pkts if total_pkts > 0 else 0.5
            max_out = int(config.appscanner_max_pkts * ratio)
            max_in = config.appscanner_max_pkts - max_out
            outgoing = outgoing[:max_out]
            incoming = incoming[:max_in]

        incoming_arr = np.array(incoming, dtype=np.float64)
        outgoing_arr = np.array(outgoing, dtype=np.float64)
        bidirectional_arr = np.concatenate([incoming_arr, outgoing_arr])

        in_features = compute_statistics(incoming_arr, config.appscanner_percentiles)
        out_features = compute_statistics(outgoing_arr, config.appscanner_percentiles)
        bi_features = compute_statistics(bidirectional_arr, config.appscanner_percentiles)

        result.append(np.concatenate([in_features, out_features, bi_features]))
    return result


def extract_yatc_features(flows: Dict[tuple, List[PacketInfo]], config: Config) -> List[np.ndarray]:
    """提取 YaTC 特征"""
    result = []
    bytes_per_packet = config.yatc_header_len + config.yatc_payload_len
    total_bytes = config.yatc_num_packets * bytes_per_packet

    for packets in flows.values():
        if len(packets) < config.yatc_num_packets:
            continue

        all_bytes = bytearray(total_bytes)
        offset = 0

        for pkt in packets[:config.yatc_num_packets]:
            header = pkt.ip_header + pkt.l4_header
            header = header[:config.yatc_header_len]
            all_bytes[offset:offset + len(header)] = header

            payload = pkt.payload[:config.yatc_payload_len]
            payload_offset = offset + config.yatc_header_len
            all_bytes[payload_offset:payload_offset + len(payload)] = payload

            offset += bytes_per_packet

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


_worker_status = None


def init_worker(worker_status):
    """初始化 worker 进程"""
    global _worker_status
    _worker_status = worker_status


def process_single_pcap(args: Tuple[str, str, str, Config]) -> Tuple[str, str, str, FlowFeatures, float, int, int]:
    """
    处理单个 PCAP 文件

    Args:
        args: (pcap_path, label, page_type, config)

    Returns:
        (pcap_path, label, page_type, features, memory_mb, flow_count, packet_count)
    """
    pcap_path, label, page_type, config = args

    features = FlowFeatures()
    memory_mb = 0.0
    flow_count = 0
    packet_count = 0

    try:
        pcap_path = Path(pcap_path)

        if _worker_status is not None:
            pid = os.getpid()
            _worker_status[pid] = pcap_path.name

        proc = psutil.Process()

        if pcap_path.stat().st_size < config.min_pcap_size:
            return (str(pcap_path), label, page_type, features, memory_mb, flow_count, packet_count)

        flows = parse_pcap_once(pcap_path, config)

        if not flows:
            return (str(pcap_path), label, page_type, features, memory_mb, flow_count, packet_count)

        flow_count = len(flows)
        packet_count = sum(len(pkts) for pkts in flows.values())

        # 提取四种特征
        features.fsnet = extract_fsnet_features(flows, config)
        features.df = extract_df_features(flows, config)
        features.appscanner = extract_appscanner_features(flows, config)
        features.yatc = extract_yatc_features(flows, config)

        memory_mb = proc.memory_info().rss / (1024 * 1024)

    except Exception as e:
        pass

    return (str(pcap_path), label, page_type, features, memory_mb, flow_count, packet_count)


# ============================================================================
# 主程序
# ============================================================================

def is_homepage(filename: str, config: Config) -> bool:
    """判断是否为首页文件"""
    return filename.startswith(config.homepage_prefix)


def collect_pcap_files(root: Path, config: Config) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[Tuple[str, str]]]]:
    """收集 PCAP 文件"""
    batch_dir = root / "batch"
    single_dir = root / "single"

    batch_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    single_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    if batch_dir.exists():
        for website_dir in sorted(batch_dir.iterdir()):
            if not website_dir.is_dir():
                continue
            pcap_dir = website_dir / "pcap"
            if not pcap_dir.exists():
                continue
            label = website_dir.name
            for pcap_file in pcap_dir.glob("*.pcap"):
                batch_files[label].append((str(pcap_file), 'aggregate'))

    if single_dir.exists():
        for website_dir in sorted(single_dir.iterdir()):
            if not website_dir.is_dir():
                continue
            pcap_dir = website_dir / "pcap"
            if not pcap_dir.exists():
                continue
            label = website_dir.name
            for pcap_file in pcap_dir.glob("*.pcap"):
                page_type = 'homepage' if is_homepage(pcap_file.name, config) else 'subpage'
                single_files[label].append((str(pcap_file), page_type))

    return dict(batch_files), dict(single_files)


def main():
    max_sys_mem_percent = 0
    parser = argparse.ArgumentParser(description="统一消融实验数据处理脚本")
    parser.add_argument("--root", type=str, default="/mnt/netdisk/dataset/ablation_study",
                        help="消融实验数据集根目录")
    parser.add_argument("--output", type=str, default=".",
                        help="输出目录（项目根目录）")
    parser.add_argument("--procs", type=int, default=64,
                        help="并行进程数")
    parser.add_argument("--memory_limit", type=float, default=0.8,
                        help="内存使用率阈值")
    args = parser.parse_args()

    config = Config()
    config.root = Path(args.root)
    config.output = Path(args.output)
    config.max_procs = args.procs
    config.memory_limit = args.memory_limit

    print("=" * 70)
    print("统一消融实验数据处理脚本")
    print("=" * 70)
    print(f"输入目录: {config.root}")
    print(f"输出目录: {config.output}")
    print(f"并行进程数: {config.max_procs}")
    print(f"内存限制: {config.memory_limit * 100:.0f}%")
    print()
    print("将为以下模型生成消融实验数据:")
    print("  1. FS-Net:            数据包长度序列 (±int16)")
    print("  2. DeepFingerprinting: 方向序列 (±1 int8)")
    print("  3. AppScanner:         54维统计特征 (float32)")
    print("  4. YaTC:               MFR图像 (40×40 uint8)")
    print("=" * 70)

    # 收集 PCAP 文件
    batch_files, single_files = collect_pcap_files(config.root, config)

    # 获取所有标签
    all_labels = sorted(set(batch_files.keys()) | set(single_files.keys()))
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for i, label in enumerate(all_labels)}

    print(f"\n发现 {len(all_labels)} 个网站类别")

    # 统计
    batch_count = sum(len(v) for v in batch_files.values())
    single_count = sum(len(v) for v in single_files.values())
    homepage_count = sum(1 for files in single_files.values() for _, pt in files if pt == 'homepage')
    subpage_count = single_count - homepage_count

    print(f"  Batch 数据集: {batch_count} 个 PCAP")
    print(f"  Single 数据集: {single_count} 个 PCAP (首页: {homepage_count}, 子页面: {subpage_count})")
    print(f"  总计: {batch_count + single_count} 个 PCAP")

    # 输出目录
    OUTPUT_DIRS = {
        "fsnet": config.output / "FS-Net" / "data" / "ablation_study",
        "df": config.output / "DeepFingerprinting" / "data" / "ablation_study",
        "appscanner": config.output / "AppScanner" / "data" / "ablation_study",
        "yatc": config.output / "YaTC" / "data" / "ablation_study",
    }

    for out_dir in OUTPUT_DIRS.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    # 准备任务
    all_tasks = []
    label_task_counts = {}

    for label in all_labels:
        tasks_for_label = []
        if label in batch_files:
            for pcap_path, page_type in batch_files[label]:
                tasks_for_label.append((pcap_path, label, page_type, config))
        if label in single_files:
            for pcap_path, page_type in single_files[label]:
                tasks_for_label.append((pcap_path, label, page_type, config))

        label_task_counts[label] = len(tasks_for_label)
        all_tasks.extend(tasks_for_label)

    print(f"\n总计 {len(all_tasks)} 个 PCAP 文件待处理")
    print("=" * 70)

    # 数据收集器（按 page_type 分类）
    # batch 数据 (Dataset A)
    fsnet_batch: Dict[str, List] = defaultdict(list)
    df_batch: Dict[str, List] = defaultdict(list)
    appscanner_batch: Dict[str, List] = defaultdict(list)
    yatc_batch: Dict[str, List] = defaultdict(list)

    # single 首页数据 (Dataset B - homepage)
    fsnet_homepage: Dict[str, List] = defaultdict(list)
    df_homepage: Dict[str, List] = defaultdict(list)
    appscanner_homepage: Dict[str, List] = defaultdict(list)
    yatc_homepage: Dict[str, List] = defaultdict(list)

    # single 子页面数据 (Dataset B - subpage)
    fsnet_subpage: Dict[str, List] = defaultdict(list)
    df_subpage: Dict[str, List] = defaultdict(list)
    appscanner_subpage: Dict[str, List] = defaultdict(list)
    yatc_subpage: Dict[str, List] = defaultdict(list)

    memory_wait_count = 0

    # 创建共享状态
    manager = mp.Manager()
    worker_status = manager.dict()

    MAX_PENDING = config.max_procs
    pending_results = []
    task_index = 0
    last_monitor_time = 0
    MONITOR_INTERVAL = 30.0

    def print_worker_status(pool):
        nonlocal max_sys_mem_percent
        try:
            workers = pool._pool
            lines = []
            total_mem_gb = 0
            for i, worker in enumerate(workers):
                if worker.is_alive():
                    try:
                        proc = psutil.Process(worker.pid)
                        mem_mb = proc.memory_info().rss / (1024 * 1024)
                        mem_gb = mem_mb / 1024
                        total_mem_gb += mem_gb
                        current_file = worker_status.get(worker.pid, "-")
                        if mem_gb >= 1:
                            lines.append(f"  Worker {i:2d} [PID {worker.pid}]: {mem_gb:6.2f} GB | {current_file}")
                        else:
                            lines.append(f"  Worker {i:2d} [PID {worker.pid}]: {mem_mb:6.0f} MB | {current_file}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        lines.append(f"  Worker {i:2d} [PID {worker.pid}]: N/A")
                else:
                    lines.append(f"  Worker {i:2d}: dead")
            if lines:
                main_mem_gb = psutil.Process().memory_info().rss / (1024**3)
                sys_mem = psutil.virtual_memory()
                if sys_mem.percent > max_sys_mem_percent:
                    max_sys_mem_percent = sys_mem.percent
                tqdm.write("\n" + "=" * 70)
                tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] 主进程: {main_mem_gb:.2f} GB | Worker总计: {total_mem_gb:.2f} GB | 系统: {sys_mem.percent:.1f}% (峰值: {max_sys_mem_percent:.1f}%)")
                tqdm.write("\n".join(lines))
                tqdm.write("=" * 70)
        except Exception:
            pass

    def collect_completed():
        nonlocal pending_results
        still_pending = []
        collected = 0

        for async_result, task_info in pending_results:
            if async_result.ready():
                try:
                    pcap_path, label, page_type, features, memory_mb, flow_count, packet_count = async_result.get(timeout=1)

                    pcap_name = Path(pcap_path).name
                    mem_str = f"{memory_mb:.0f}MB" if memory_mb > 0 else "-"
                    tqdm.write(f"  [{label}] {pcap_name} | {page_type} | 流:{flow_count} 包:{packet_count} 内存:{mem_str}")

                    # 根据 page_type 分配数据到对应的收集器
                    if page_type == "aggregate":
                        if features.fsnet:
                            fsnet_batch[label].extend(features.fsnet)
                        if features.df:
                            df_batch[label].extend(features.df)
                        if features.appscanner:
                            appscanner_batch[label].extend(features.appscanner)
                        if features.yatc:
                            yatc_batch[label].extend(features.yatc)
                    elif page_type == "homepage":
                        if features.fsnet:
                            fsnet_homepage[label].extend(features.fsnet)
                        if features.df:
                            df_homepage[label].extend(features.df)
                        if features.appscanner:
                            appscanner_homepage[label].extend(features.appscanner)
                        if features.yatc:
                            yatc_homepage[label].extend(features.yatc)
                    elif page_type == "subpage":
                        if features.fsnet:
                            fsnet_subpage[label].extend(features.fsnet)
                        if features.df:
                            df_subpage[label].extend(features.df)
                        if features.appscanner:
                            appscanner_subpage[label].extend(features.appscanner)
                        if features.yatc:
                            yatc_subpage[label].extend(features.yatc)

                    collected += 1
                    pbar.update(1)
                except Exception as e:
                    tqdm.write(f"  [错误] {task_info}: {e}")
                    pbar.update(1)
            else:
                still_pending.append((async_result, task_info))

        pending_results = still_pending
        return collected

    pbar = tqdm(total=len(all_tasks), desc="处理 PCAP", position=0)

    with mp.Pool(processes=config.max_procs, initializer=init_worker, initargs=(worker_status,), maxtasksperchild=2) as pool:
        while task_index < len(all_tasks) or pending_results:
            current_time = time.time()
            if current_time - last_monitor_time >= MONITOR_INTERVAL:
                print_worker_status(pool)
                last_monitor_time = current_time

            collect_completed()

            while len(pending_results) < MAX_PENDING and task_index < len(all_tasks):
                task = all_tasks[task_index]
                async_result = pool.apply_async(process_single_pcap, (task,))
                pending_results.append((async_result, task[0]))
                task_index += 1

            if pending_results and task_index >= len(all_tasks):
                time.sleep(0.1)
            elif not pending_results and task_index >= len(all_tasks):
                break
            elif len(pending_results) >= MAX_PENDING:
                time.sleep(0.1)

    pbar.close()

    if memory_wait_count > 0:
        print(f"\n[内存监控] 共触发 {memory_wait_count} 次内存等待")

    # =========================================================================
    # 保存数据 - 与 AppScanner/ablation_processor.py 格式一致
    # =========================================================================
    print("\n保存数据...")
    print("=" * 70)

    def save_model_data(
        model_name: str,
        out_dir: Path,
        batch_data: Dict[str, List],
        homepage_data: Dict[str, List],
        subpage_data: Dict[str, List],
        data_type: str  # 'sequences', 'features', 'images'
    ):
        """
        保存模型数据，格式与 AppScanner/ablation_processor.py 一致：
        - dataset_a_batch.pkl
        - dataset_b_single.pkl
        """
        # Dataset A (batch)
        batch_samples = []
        batch_labels = []
        for label, samples in batch_data.items():
            label_id = label2id[label]
            for sample in samples:
                batch_samples.append(sample)
                batch_labels.append(label_id)

        if batch_samples:
            if data_type == 'features':
                data_a = {
                    'features': np.stack(batch_samples, axis=0).astype(np.float32),
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                    'num_features': 54,
                }
            elif data_type == 'images':
                data_a = {
                    'images': np.stack(batch_samples, axis=0).astype(np.uint8),
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                }
            else:  # sequences
                data_a = {
                    'sequences': batch_samples,
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                }

            with open(out_dir / 'dataset_a_batch.pkl', 'wb') as f:
                pickle.dump(data_a, f)

            print(f"  [{model_name}] Dataset A (batch): {len(batch_labels)} 条")

        # Dataset B (single)
        homepage_samples = []
        homepage_labels = []
        for label, samples in homepage_data.items():
            label_id = label2id[label]
            for sample in samples:
                homepage_samples.append(sample)
                homepage_labels.append(label_id)

        subpage_samples = []
        subpage_labels = []
        for label, samples in subpage_data.items():
            label_id = label2id[label]
            for sample in samples:
                subpage_samples.append(sample)
                subpage_labels.append(label_id)

        all_samples = homepage_samples + subpage_samples
        all_sample_labels = homepage_labels + subpage_labels

        if all_samples:
            if data_type == 'features':
                data_b = {
                    'all_features': np.stack(all_samples, axis=0).astype(np.float32),
                    'all_labels': np.array(all_sample_labels, dtype=np.int64),
                    'homepage_features': np.stack(homepage_samples, axis=0).astype(np.float32) if homepage_samples else np.array([], dtype=np.float32),
                    'homepage_labels': np.array(homepage_labels, dtype=np.int64),
                    'subpage_features': np.stack(subpage_samples, axis=0).astype(np.float32) if subpage_samples else np.array([], dtype=np.float32),
                    'subpage_labels': np.array(subpage_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                    'num_features': 54,
                }
            elif data_type == 'images':
                data_b = {
                    'all_images': np.stack(all_samples, axis=0).astype(np.uint8),
                    'all_labels': np.array(all_sample_labels, dtype=np.int64),
                    'homepage_images': np.stack(homepage_samples, axis=0).astype(np.uint8) if homepage_samples else np.array([], dtype=np.uint8),
                    'homepage_labels': np.array(homepage_labels, dtype=np.int64),
                    'subpage_images': np.stack(subpage_samples, axis=0).astype(np.uint8) if subpage_samples else np.array([], dtype=np.uint8),
                    'subpage_labels': np.array(subpage_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                }
            else:  # sequences
                data_b = {
                    'all_sequences': all_samples,
                    'all_labels': np.array(all_sample_labels, dtype=np.int64),
                    'homepage_sequences': homepage_samples,
                    'homepage_labels': np.array(homepage_labels, dtype=np.int64),
                    'subpage_sequences': subpage_samples,
                    'subpage_labels': np.array(subpage_labels, dtype=np.int64),
                    'label_map': id2label,
                    'num_classes': len(all_labels),
                }

            with open(out_dir / 'dataset_b_single.pkl', 'wb') as f:
                pickle.dump(data_b, f)

            print(f"  [{model_name}] Dataset B (single): homepage={len(homepage_labels)}, subpage={len(subpage_labels)}")

        return len(batch_labels), len(homepage_labels), len(subpage_labels)

    # 保存各模型数据
    total_counts = {}

    # FS-Net
    b, h, s = save_model_data(
        "FS-Net", OUTPUT_DIRS["fsnet"],
        fsnet_batch, fsnet_homepage, fsnet_subpage,
        data_type='sequences'
    )
    total_counts['fsnet'] = b + h + s

    # DeepFingerprinting
    b, h, s = save_model_data(
        "DeepFingerprinting", OUTPUT_DIRS["df"],
        df_batch, df_homepage, df_subpage,
        data_type='sequences'
    )
    total_counts['df'] = b + h + s

    # AppScanner
    b, h, s = save_model_data(
        "AppScanner", OUTPUT_DIRS["appscanner"],
        appscanner_batch, appscanner_homepage, appscanner_subpage,
        data_type='features'
    )
    total_counts['appscanner'] = b + h + s

    # YaTC
    b, h, s = save_model_data(
        "YaTC", OUTPUT_DIRS["yatc"],
        yatc_batch, yatc_homepage, yatc_subpage,
        data_type='images'
    )
    total_counts['yatc'] = b + h + s

    # 总结
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)

    print(f"\n各模型数据统计:")
    print(f"  FS-Net:             {total_counts['fsnet']:>8} 条流")
    print(f"  DeepFingerprinting: {total_counts['df']:>8} 条流")
    print(f"  AppScanner:         {total_counts['appscanner']:>8} 条流")
    print(f"  YaTC:               {total_counts['yatc']:>8} 个图像")

    print(f"\n系统内存峰值: {max_sys_mem_percent:.1f}%")

    print(f"\n输出目录结构:")
    print(f"  {config.output}/")
    print(f"  ├── FS-Net/data/ablation_study/")
    print(f"  │   ├── dataset_a_batch.pkl")
    print(f"  │   └── dataset_b_single.pkl")
    print(f"  ├── DeepFingerprinting/data/ablation_study/")
    print(f"  │   ├── dataset_a_batch.pkl")
    print(f"  │   └── dataset_b_single.pkl")
    print(f"  ├── AppScanner/data/ablation_study/")
    print(f"  │   ├── dataset_a_batch.pkl")
    print(f"  │   └── dataset_b_single.pkl")
    print(f"  └── YaTC/data/ablation_study/")
    print(f"      ├── dataset_a_batch.pkl")
    print(f"      └── dataset_b_single.pkl")

    print(f"\n数据格式说明:")
    print(f"  dataset_a_batch.pkl:")
    print(f"    - features/sequences/images: 所有样本")
    print(f"    - labels: 标签数组")
    print(f"    - label_map: {{id: website_name}}")
    print(f"  dataset_b_single.pkl:")
    print(f"    - all_*: 所有样本")
    print(f"    - homepage_*: 首页样本")
    print(f"    - subpage_*: 子页面样本")

    print(f"\n下一步:")
    print(f"  cd FS-Net && python train_ablation.py --experiment 1")
    print(f"  cd DeepFingerprinting && python train_ablation.py --experiment 1")
    print(f"  cd AppScanner && python train_ablation.py --experiment 1")
    print(f"  cd YaTC && python train_ablation.py --experiment 1")


if __name__ == "__main__":
    main()
