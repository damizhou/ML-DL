#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
df_build_npz.py

功能：
- 递归扫描 ROOT 目录（按子目录名作为 label）下的 .pcap / .pcapng；
- 多进程并行处理：每个 PCAP 输出一个 NPZ 文件；
- 方向不敏感五元组聚合 (ip1,port1,ip2,port2,PROTO)；
- “处理所有 TCP/UDP 报文”：对每个报文使用 L4 完整字节（头部+负载），
  将其顺序拼接到对应会话的字节序列中；
- 剔除 mDNS（UDP 5353），兼容 IPv4 / IPv6，兼容 Ethernet/SLL/原始IP；
- NPZ 结构：flows(keys 对应的变长 uint8 数组，object dtype)、keys、lengths、label、labels。
"""

from __future__ import annotations

import os
import ipaddress
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Iterable, List
import multiprocessing as mp

import numpy as np
import dpkt


# ========== 基本配置（按需修改） ==========
# 输入根目录：一级子目录名为标签；其下递归查找 pcap/pcapng
ROOT        = Path("/netdisk/dataset/vpn/data")
# 输出根目录：结构为  OUT_ROOT/<label>/<pcap_stem>.npz
OUT_ROOT    = Path(__file__).resolve().parent / "npz_longflows"
# 已存在是否覆盖
OVERWRITE   = False
# 并发进程数
MAX_PROCS   = 32
# 剔除的 L4 端口集合（仅对 UDP 生效，例：mDNS）
EXCLUDE_UDP_PORTS = {5353}
# ========================================


# --------- 工具与类型 ---------
FiveTuple = Tuple[str, int, str, int, str]  # (ip1, port1, ip2, port2, "TCP"/"UDP")


def _inet_to_str(raw: bytes) -> str:
    # 兼容 v4/v6，压缩输出
    return ipaddress.ip_address(raw).compressed


def _canon_key(src_raw: bytes, sport: int, dst_raw: bytes, dport: int, proto: str) -> FiveTuple:
    """
    方向不敏感键：按 (ip整数值, 端口) 排序，较小端在前。
    """
    a = ipaddress.ip_address(src_raw)
    b = ipaddress.ip_address(dst_raw)
    ka = (int(a), sport)
    kb = (int(b), dport)
    if ka <= kb:
        return (a.compressed, sport, b.compressed, dport, proto)
    else:
        return (b.compressed, dport, a.compressed, sport, proto)


def _parse_l3(buf: bytes):
    """
    最稳妥地拿到 L3：
    - 优先 Ethernet，再 SLL，再原始 IPv4/IPv6 兜底
    返回 dpkt.ip.IP 或 dpkt.ip6.IP6，失败返回 None
    """
    # Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        l3 = eth.data
        if isinstance(l3, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return l3
    except Exception:
        pass

    # Linux SLL (cooked)
    try:
        sll = dpkt.sll.SLL(buf)
        l3 = sll.data
        if isinstance(l3, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return l3
    except Exception:
        pass

    # RAW IPv4/IPv6
    try:
        ver = buf[0] >> 4
        if ver == 4:
            return dpkt.ip.IP(buf)
        elif ver == 6:
            return dpkt.ip6.IP6(buf)
    except Exception:
        pass

    return None


def _iter_packets_any(pcap_path: Path) -> Iterable[Tuple[float, bytes]]:
    """
    统一兼容 pcap / pcapng，生成 (ts, buf)
    """
    with pcap_path.open("rb") as f:
        # pcap
        try:
            f.seek(0)
            r = dpkt.pcap.Reader(f)
            for ts, buf in r:
                yield ts, buf
            return
        except Exception:
            pass
        # pcapng
        f.seek(0)
        r = dpkt.pcapng.Reader(f)
        for rec in r:
            # dpkt pcapng Reader 迭代返回 (ts, buf) 或更复杂结构，这里做宽松兼容
            try:
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    yield rec[0], rec[1]
            except Exception:
                continue


# --------- PCAP -> NPZ（单个文件）---------
def pcap_to_npz_all_packets(
    pcap_path: str,
    label: Union[int, str],
    out_path: Optional[str] = None,
    min_pkts_per_flow: int = 200,  # 只保留包数 > 200 的流
) -> str:
    """
    将一个 PCAP 写成一个 NPZ（方向符号序列）：
      - flows: object 数组，元素为各流的方向序列（np.int8，±1）
      - labels: object 数组，对应每条流的字符串标签
    仅保留“包数 > min_pkts_per_flow”的流；若无符合条件的流则跳过该文件。
    其他保持不变：跳过 <20KB 的 pcap；过滤 mDNS(UDP:5353)；仅在写入前创建输出目录。
    """
    # --- 文件大小校验：小于 20KB 直接跳过 ---
    size_bytes = Path(pcap_path).stat().st_size
    if size_bytes < 20 * 1024:
        raise RuntimeError(f"skip-small: {size_bytes} bytes < 20KB")

    if out_path is None:
        raise RuntimeError("out_path must be specified")

    # key: (proto_str, (ip_str,port)_min, (ip_str,port)_max) -> List[np.int8]
    flows: Dict[
        Tuple[str, Tuple[str, int], Tuple[str, int]],
        List[np.int8]
    ] = {}

    for _ts, buf in _iter_packets_any(Path(pcap_path)):
        l3 = _parse_l3(buf)
        if l3 is None:
            continue

        # 取协议/端口
        proto_str: Optional[str] = None
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

        if proto_str is None or sport is None or dport is None:
            continue

        # 计算“无向”键与当前包的方向符号
        a_ip = ipaddress.ip_address(l3.src)
        b_ip = ipaddress.ip_address(l3.dst)
        a = (int(a_ip), sport)
        b = (int(b_ip), dport)
        if a <= b:
            key = (proto_str, (a_ip.compressed, sport), (b_ip.compressed, dport))
            sign = np.int8(+1)
        else:
            key = (proto_str, (b_ip.compressed, dport), (a_ip.compressed, sport))
            sign = np.int8(-1)

        flows.setdefault(key, []).append(sign)

    # —— 仅保留“包数 > min_pkts_per_flow”的流 ——
    kept = [np.asarray(v, dtype=np.int8) for v in flows.values() if len(v) > min_pkts_per_flow]
    if not kept:
        raise RuntimeError(f"skip-few-pkts: no flows with > {min_pkts_per_flow} packets")

    labels = [label] * len(kept)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        flows=np.asarray(kept, dtype=object),
        labels=np.asarray(labels, dtype=object),
    )
    return out_path

# --------- 目录扫描 + 多进程驱动 ---------
def _find_pcaps(root: Path) -> List[Tuple[str, str, str]]:
    """
    返回 (pcap_abs_path, label, out_abs_path) 列表
    注意：不在此处创建输出目录，改为在写入 NPZ 前再创建。
    """
    index = 0
    tasks: List[Tuple[str, str, str]] = []
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if index == 100:
            break
        label = label_dir.name
        for p in sorted(label_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".pcap", ".pcapng"):
                continue
            out_dir = OUT_ROOT / label
            out_path = out_dir / (p.stem + ".npz")
            tasks.append((str(p.resolve()), label, str(out_path.resolve())))
        index += 1
    return tasks


def _worker(args: Tuple[str, str, str]) -> Tuple[bool, str, str]:
    """
    单个 PCAP 处理
    return: (ok, pcap_path, msg_or_out)
    """
    pcap_path, label, out_path = args
    try:
        if (not OVERWRITE) and os.path.exists(out_path):
            return True, pcap_path, f"skip-exist -> {out_path}"
        out_file = pcap_to_npz_all_packets(pcap_path, label=label, out_path=out_path)
        return True, pcap_path, f"ok -> {out_file}"
    except Exception as e:
        return False, pcap_path, f"error: {e}"


def main() -> None:
    print(f"[info] scan root: {ROOT}")
    print(f"[info] out root:  {OUT_ROOT}")
    tasks = _find_pcaps(ROOT)
    print(f"[info] total pcaps: {len(tasks)}")

    if not tasks:
        return

    procs = MAX_PROCS
    print(f"[info] use processes: {procs}")

    ok, skip, err = 0, 0, 0
    with mp.Pool(processes=procs) as pool:
        for success, pcap_path, msg in pool.imap_unordered(_worker, tasks, chunksize=1):
            if success:
                if msg.startswith("skip-exist"):
                    skip += 1
                else:
                    ok += 1
                print(f"[OK] {pcap_path} | {msg}")
            else:
                err += 1
                print(f"[ERR] {pcap_path} | {msg}")
    print(f"[done] ok={ok} skip={skip} err={err}")

if __name__ == "__main__":
    main()
