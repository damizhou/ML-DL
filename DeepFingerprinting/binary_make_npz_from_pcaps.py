#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
binary_make_npz_from_pcaps.py
将两个 pcap 转换为二分类训练数据（方向序列），零参数开箱即用。
输出到 OUTDIR：binary.npz / labels.json / class_count.csv
"""
from __future__ import annotations
import os, io, gzip, json, struct, csv
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import dpkt

# ========= 配置（仅需改路径/标签） =========
PCAPS = [
    # ("/home/pcz/DL/ML&DL/Dataset/novpn/unknow_ubuntu24.04_novpn_20240824_170450_bbc.com.pcap", "bbc.com"),
    ("/home/pcz/DL/ML&DL/Dataset/novpn/hz_ubuntu24.04_novpn_20241119_151725_abc.com.pcap", "abc.com"),
]
OUTDIR            = "outputs_binary"
NPZ_NAME          = "binary.npz"
LABELS_NAME       = "labels.json"
CLASSCOUNT_NAME   = "class_count.csv"

# 序列抽取策略
PAYLOAD_ONLY      = False   # 只统计有负载的 TCP/UDP 包
MIN_LEN           = 1     # 丢弃短于该长度的流
SEGMENT_LONG      = False  # 是否对超长流滑窗切段
WINDOW            = 5000
STRIDE            = 2500

# 过滤项
EXCLUDE_MDNS      = True   # 过滤 mDNS
# ========================================

# ---------- pcap 读取（兼容 pcap/pcapng/gz） ----------
PCAP_MAGIC = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
PCAPNG_MAGIC = 0x0A0D0D0A
GZIP_MAGIC = b"\x1f\x8b"

MDNS_UDP_PORT = 5353
MDNS_MCAST_IPS = {"224.0.0.251", "ff02::fb", "ff05::fb"}  # 常见 mDNS 组播地址

def is_mdns(src_ip: str, sport: int, dst_ip: str, dport: int, proto: int) -> bool:
    """mDNS 判断：UDP 且端口 5353，或命中常见组播地址"""
    if proto != 17:  # 17 = UDP
        return False
    if sport == MDNS_UDP_PORT or dport == MDNS_UDP_PORT:
        return True
    if (src_ip in MDNS_MCAST_IPS) or (dst_ip in MDNS_MCAST_IPS):
        return True
    return False

def _open_maybe_gzip(path: str):
    f = open(path, "rb")
    head = f.read(2); f.seek(0)
    if head == GZIP_MAGIC:
        data = gzip.open(f).read()
        f.close()
        return io.BytesIO(data)
    return f

def _iter_raw_packets(path: str):
    f = _open_maybe_gzip(path)
    head4 = f.read(4); f.seek(0)
    magic = struct.unpack(">I", head4)[0] if len(head4) == 4 else None
    try:
        if magic in PCAP_MAGIC:
            r = dpkt.pcap.Reader(f)
            for _, buf in r: yield buf
        elif magic == PCAPNG_MAGIC:
            r = dpkt.pcapng.Reader(f)
            for item in r:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    yield item[1]
        else:
            try:
                r = dpkt.pcap.Reader(f)
                for _, buf in r: yield buf
            except Exception:
                f.seek(0); r = dpkt.pcapng.Reader(f)
                for item in r:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        yield item[1]
    finally:
        f.close()

def _inet_to_str(x: bytes, af: int) -> str:
    import socket
    return socket.inet_ntop(af, x)

def _parse_ip_l4_auto(pkt_bytes: bytes):
    """Ethernet 优先，失败回退 SLL；返回 (src, sport, dst, dport, proto, payload_len) 或 None。"""
    try:
        eth = dpkt.ethernet.Ethernet(pkt_bytes); ip_pkt = eth.data
    except Exception:
        ip_pkt = None
    if not isinstance(ip_pkt, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            sll = dpkt.sll.SLL(pkt_bytes); ip_pkt = sll.data
        except Exception:
            return None
    import socket
    if isinstance(ip_pkt, dpkt.ip.IP):
        af = socket.AF_INET; src, dst, proto, l4 = _inet_to_str(ip_pkt.src, af), _inet_to_str(ip_pkt.dst, af), ip_pkt.p, ip_pkt.data
    elif isinstance(ip_pkt, dpkt.ip6.IP6):
        af = socket.AF_INET6; src, dst, proto, l4 = _inet_to_str(ip_pkt.src, af), _inet_to_str(ip_pkt.dst, af), ip_pkt.nxt, ip_pkt.data
    else:
        return None
    if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        sport, dport = l4.sport, l4.dport
        if sport is None or dport is None: return None
        payload_len = len(l4.data or b"")
        return src, sport, dst, dport, int(proto), int(payload_len)
    return None

def pcap_to_flow_dirs(pcap_path: str, payload_only: bool, min_len: int) -> List[np.ndarray]:
    """一个 pcap -> 多条方向序列（±1）"""
    flows: Dict[tuple, Dict[str, object]] = {}
    for buf in _iter_raw_packets(pcap_path):
        rec = _parse_ip_l4_auto(buf)
        if rec is None: continue
        src, sport, dst, dport, proto, payload_len = rec
        if EXCLUDE_MDNS and is_mdns(src, sport, dst, dport, proto):
            continue
        if payload_only and payload_len <= 0:  # 仅统计有负载的 L4 包
            continue
        a, b = (src, sport), (dst, dport)
        key = (a, b, proto) if a <= b else (b, a, proto)
        if key not in flows:
            flows[key] = {"origin": (src, sport, dst, dport), "seq": []}
        origin = flows[key]["origin"]
        s: List[int] = flows[key]["seq"]  # type: ignore
        s.append(+1 if (src, sport, dst, dport) == origin else -1)

    seqs: List[np.ndarray] = []
    for v in flows.values():
        arr = np.asarray(v["seq"], dtype=np.int8)
        if len(arr) >= min_len:
            seqs.append(arr)

    # 可选：对超长流切段扩增
    if SEGMENT_LONG:
        out: List[np.ndarray] = []
        for s in seqs:
            if len(s) > WINDOW:
                for st in range(0, len(s), STRIDE):
                    seg = s[st:st+WINDOW]
                    if len(seg) >= min_len: out.append(seg)
            else:
                out.append(s)
        return out
    return seqs

# ---------- 主流程 ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    class_to_samples: Dict[str, List[np.ndarray]] = {}
    for p, lab in PCAPS:
        print(f"[pcap] {p} -> {lab}")
        try:
            seqs = pcap_to_flow_dirs(p, PAYLOAD_ONLY, MIN_LEN)
        except Exception as e:
            print(f"  [解析失败] {e}")
            continue
        class_to_samples.setdefault(lab, []).extend(seqs)
        print(f"  +{len(seqs)} 序列")

    labels = sorted(class_to_samples.keys())
    label2id = {lab:i for i,lab in enumerate(labels)}
    id2label = {str(i):lab for lab,i in label2id.items()}

    X: List[np.ndarray] = []
    y: List[int] = []
    for lab, seqs in class_to_samples.items():
        for s in seqs:
            X.append(s); y.append(label2id[lab])
    y = np.asarray(y, dtype=np.int64)

    print("\n== 汇总 ==")
    for lab in labels:
        print(f"  {lab:12s} -> {len(class_to_samples[lab])}")
    print(f"total={len(X)}  classes={len(labels)}")

    # 写 npz / labels.json / class_count.csv
    np.savez(os.path.join(OUTDIR, NPZ_NAME), X=np.array(X, dtype=object), y=y, allow_pickle=True)
    with open(os.path.join(OUTDIR, LABELS_NAME), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTDIR, CLASSCOUNT_NAME), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["class","class_id","count"])
        for lab in labels:
            wr.writerow([lab, label2id[lab], len(class_to_samples[lab])])
    print(f"\n[write] {os.path.join(OUTDIR, NPZ_NAME)} / {LABELS_NAME} / {CLASSCOUNT_NAME}")

if __name__ == "__main__":
    main()
