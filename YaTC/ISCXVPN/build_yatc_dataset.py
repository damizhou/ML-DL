#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_yatc_dataset.py

在已有 artifacts/iscx/label_map.csv 的基础上，从 pcap 中抽取简化版 YaTc 特征：

- 读取 label_map.csv，遍历所有 pcap 文件
- 按五元组聚合为流（方向无关）
- 每条流取前 MAX_PKTS 个包，每包取至多 PKT_BYTES 字节的 L4 负载
- 共 MAX_PKTS * PKT_BYTES 字节，reshape 为 (MFR_H, MFR_W) 的 uint8 矩阵
- 输出全集数据，不做 train/val/test 切分：
    artifacts/iscx/yatc/
      - data.npz    # X: (N, MFR_H, MFR_W), y: (N,)
      - labels.json
      - class_count.csv
      - meta.json

注意：
  - 要求已存在 artifacts/iscx/label_map.csv 和 service_vocab.csv
  - 训练时请自行在脚本内随机 8:1:1 划分
"""

from __future__ import annotations
import os
import csv
import json
import gzip
import struct
import socket
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ===== 基本配置 =====
DATASET_KEY   = "iscx"
SCHEMA_KEY    = "yatc"
ART_ROOT      = "artifacts"

# 流 / 特征配置
MAX_PKTS      = 5      # 每条流最多取多少个包
PKT_BYTES     = 320    # 每个包最多使用多少字节（L4 负载）
MFR_H, MFR_W  = 40, 40 # MFR 矩阵尺寸，需满足 MFR_H * MFR_W == MAX_PKTS * PKT_BYTES
MIN_PKTS      = 1      # 流中至少包含多少个有效包才保留

# 解析 / 过滤配置（与 dirseq 保持风格一致）
PAYLOAD_ONLY      = True
EXCLUDE_L4        = {("udp", 5353)}  # 丢弃整条流
EXCLUDE_MCAST_IPS = {"224.0.0.251", "ff02::fb", "ff05::fb"}

# pcap 相关
USE_GZIP_EXTS = (".gz",)
PCAP_EXTS     = (".pcap", ".pcapng", ".pcap.gz", ".pcapng.gz")
PCAP_MAGIC    = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
PCAPNG_MAGIC  = 0x0A0D0D0A
GZIP_MAGIC    = b"\x1f\x8b"

try:
    import dpkt
except Exception as e:
    raise SystemExit("需要安装 dpkt：pip install dpkt") from e


# ===== 通用工具 =====

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    """返回 2025-11-28T08:30:00Z 形式的时间字符串（UTC，无微秒）。"""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# ===== pcap 相关工具 =====

def open_maybe_gzip(path: str):
    """支持普通文件和 .gz 文件的统一读取。"""
    f = open(path, "rb")
    head2 = f.read(2)
    f.seek(0)
    if head2 == GZIP_MAGIC:
        return gzip.GzipFile(fileobj=f)
    return f


def iter_packets(path: str):
    """从 pcap / pcapng 文件中迭代原始报文字节。"""
    f = open_maybe_gzip(path)
    head4 = f.read(4)
    f.seek(0)
    if len(head4) < 4:
        f.close()
        return

    magic = struct.unpack(">I", head4)[0]
    try:
        if magic in PCAP_MAGIC:
            r = dpkt.pcap.Reader(f)
            for _, buf in r:
                yield buf
        elif magic == PCAPNG_MAGIC:
            r = dpkt.pcapng.Reader(f)
            for item in r:
                # dpkt.pcapng.Reader 返回 (ts, buf) 或复杂结构，这里只取 buf
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    yield item[1]
        else:
            # 魔数不对时，用“试一下”的方式兜底
            try:
                r = dpkt.pcap.Reader(f)
                for _, buf in r:
                    yield buf
            except Exception:
                f.seek(0)
                r = dpkt.pcapng.Reader(f)
                for item in r:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        yield item[1]
    finally:
        f.close()


def parse_ip_l4(pkt: bytes) -> Optional[Tuple[str, int, str, int, int, bytes]]:
    """
    解析以太网 / SLL / Loopback / RAW 等封装，提取：
      src_ip(str), sport(int), dst_ip(str), dport(int), proto(int), payload(bytes)

    仅保留 TCP / UDP 报文。
    """
    ip = None

    # Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data
    except Exception:
        ip = None

    # Linux SLL
    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            sll = dpkt.sll.SLL(pkt)
            ip = sll.data
        except Exception:
            ip = None

    # Loopback
    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            lo = dpkt.loopback.Loopback(pkt)
            ip = lo.data
        except Exception:
            ip = None

    # RAW IPv4/IPv6
    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            ip = dpkt.ip.IP(pkt)
        except Exception:
            try:
                ip = dpkt.ip6.IP6(pkt)
            except Exception:
                return None  # 仍然解不出来

    # 走到这里一定是 IPv4/IPv6
    if isinstance(ip, dpkt.ip.IP):
        af = socket.AF_INET
        proto = ip.p
        l4 = ip.data
        src = socket.inet_ntop(af, ip.src)
        dst = socket.inet_ntop(af, ip.dst)
    else:
        af = socket.AF_INET6
        proto = ip.nxt
        l4 = ip.data
        src = socket.inet_ntop(af, ip.src)
        dst = socket.inet_ntop(af, ip.dst)

    if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        sport = l4.sport
        dport = l4.dport
        if sport is None or dport is None:
            return None
        payload = bytes(l4.data or b"")
        return src, int(sport), dst, int(dport), int(proto), payload

    return None


PROTO_NAME = {6: "tcp", 17: "udp"}


def is_excluded(proto: int, sport: int, dport: int, src: str, dst: str) -> bool:
    """黑名单端口 + 组播地址过滤。"""
    pn = PROTO_NAME.get(proto)
    if pn and ((pn, sport) in EXCLUDE_L4 or (pn, dport) in EXCLUDE_L4):
        return True
    if src in EXCLUDE_MCAST_IPS or dst in EXCLUDE_MCAST_IPS:
        return True
    return False


# ===== MFR 构造 =====

def flow_payloads_to_mfr(payloads: List[bytes]) -> np.ndarray:
    """
    将一条流的若干 L4 负载拼成固定长度字节序列，再 reshape 为 (MFR_H, MFR_W)。

    - 最多使用 MAX_PKTS 个包
    - 每个包最多使用 PKT_BYTES 字节
    - 不足补 0，超出截断
    """
    total_bytes = MAX_PKTS * PKT_BYTES
    if MFR_H * MFR_W != total_bytes:
        raise ValueError(
            f"MFR_H * MFR_W 必须等于 MAX_PKTS * PKT_BYTES，"
            f"当前 {MFR_H}*{MFR_W} != {MAX_PKTS}*{PKT_BYTES}"
        )

    buf = np.zeros(total_bytes, dtype=np.uint8)
    for i in range(MAX_PKTS):
        if i >= len(payloads):
            break
        data = payloads[i] or b""
        if not data:
            continue
        arr = np.frombuffer(data, dtype=np.uint8)
        n = min(arr.size, PKT_BYTES)
        start = i * PKT_BYTES
        buf[start:start + n] = arr[:n]

    return buf.reshape(MFR_H, MFR_W)


def pcap_to_mfrs(path: str) -> List[np.ndarray]:
    """
    将一个 pcap 文件解析为若干条流的 MFR 矩阵列表。
    """
    flows: Dict[Tuple[Tuple[str, int], Tuple[str, int], int], Dict[str, object]] = {}

    for raw in iter_packets(path):
        rec = parse_ip_l4(raw)
        if rec is None:
            continue
        src, sport, dst, dport, proto, payload = rec

        if is_excluded(proto, sport, dport, src, dst):
            continue
        if PAYLOAD_ONLY and len(payload) == 0:
            continue

        a = (src, sport)
        b = (dst, dport)
        key = (a, b, proto) if a <= b else (b, a, proto)

        ent = flows.get(key)
        if ent is None:
            ent = {"pkts": []}
            flows[key] = ent
        ent["pkts"].append(payload)

    mats: List[np.ndarray] = []
    for ent in flows.values():
        pkts: List[bytes] = ent["pkts"]  # type: ignore
        if len(pkts) < MIN_PKTS:
            continue
        mfr = flow_payloads_to_mfr(pkts)
        mats.append(mfr)

    return mats


# ===== label_map / vocab 读取 =====

def load_label_map(map_csv: Path) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    with map_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            p = row.get("path") or row.get("pcap_path") or ""
            lab = row.get("label") or row.get("service_id") or ""
            p = p.strip()
            lab = lab.strip()
            if not p or not lab:
                continue
            if not any(p.lower().endswith(ext) for ext in PCAP_EXTS):
                continue
            try:
                sid = int(lab)
            except ValueError:
                continue
            if os.path.isfile(p):
                pairs.append((p, sid))
    return pairs


def load_service_vocab(vocab_csv: Path) -> Dict[int, str]:
    """
    service_vocab.csv:
        service_id,service
        0,chat
        1,email
        ...
    """
    mapping: Dict[int, str] = {}
    if not vocab_csv.is_file():
        return mapping
    with vocab_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = row.get("service_id")
            name = row.get("service")
            if sid is None or name is None:
                continue
            try:
                i = int(sid)
            except ValueError:
                continue
            mapping[i] = name
    return mapping


# ===== 主流程 =====

def main() -> None:
    ds_dir = ensure_dir(Path(ART_ROOT) / DATASET_KEY)
    map_csv = ds_dir / "label_map.csv"
    vocab_csv = ds_dir / "service_vocab.csv"

    if not map_csv.is_file():
        raise SystemExit(f"未找到映射表: {map_csv}（请先运行 deal_iscxvpn_labels.py）")

    out_dir = ensure_dir(ds_dir / SCHEMA_KEY)

    pairs = load_label_map(map_csv)
    if not pairs:
        raise SystemExit(f"{map_csv} 中没有有效条目")

    print(f"[info] label_map entries = {len(pairs)}")

    svc_vocab = load_service_vocab(vocab_csv)
    if svc_vocab:
        print(f"[info] loaded service_vocab: {len(svc_vocab)} 条")
    else:
        print(f"[warn] 未找到或未能读取 {vocab_csv}，labels.json 将使用纯数字映射")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    counter: Counter[int] = Counter()

    for idx, (pcap_path, sid) in enumerate(pairs, start=1):
        print(f"[pcap] {idx:5d}/{len(pairs):5d}  sid={sid:2d}  path={pcap_path}")
        try:
            mats = pcap_to_mfrs(pcap_path)
        except Exception as e:
            print(f"[warn] 解析失败，跳过: {pcap_path} ({e})")
            continue

        if not mats:
            continue

        X_list.extend(mats)
        y_list.extend([sid] * len(mats))
        counter[sid] += len(mats)

    if not X_list:
        raise SystemExit("没有从任何 pcap 中解析出有效流")

    X = np.stack(X_list, axis=0)              # (N, MFR_H, MFR_W)
    y = np.asarray(y_list, dtype=np.int64)    # (N,)

    data_npz = out_dir / "data.npz"
    np.savez(data_npz, X=X, y=y)
    print(f"[done] data.npz -> {data_npz} (samples={X.shape[0]})")

    # 写 class_count.csv
    cc_csv = out_dir / "class_count.csv"
    with cc_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["service_id", "service", "flows"])
        for sid in sorted(counter.keys()):
            name = svc_vocab.get(sid, str(sid))
            wr.writerow([sid, name, counter[sid]])
    print(f"[done] class_count -> {cc_csv}")

    # 写 labels.json（兼容 train_df_simple 这种风格）
    max_sid = max(counter.keys())
    label2id = {str(i): i for i in range(max_sid + 1)}
    id2label = {str(i): svc_vocab.get(i, str(i)) for i in range(max_sid + 1)}

    labels_json = out_dir / "labels.json"
    with labels_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] labels.json -> {labels_json}")

    # 写 meta.json
    meta = {
        "dataset": DATASET_KEY,
        "schema": SCHEMA_KEY,
        "created_at": now_iso(),
        "config": {
            "MAX_PKTS": MAX_PKTS,
            "PKT_BYTES": PKT_BYTES,
            "MFR_H": MFR_H,
            "MFR_W": MFR_W,
            "MIN_PKTS": MIN_PKTS,
            "PAYLOAD_ONLY": PAYLOAD_ONLY,
            "EXCLUDE_L4": sorted(list(EXCLUDE_L4)),
            "EXCLUDE_MCAST_IPS": sorted(list(EXCLUDE_MCAST_IPS)),
        },
        "num_samples": int(X.shape[0]),
        "num_classes": int(max_sid + 1),
    }
    meta_json = out_dir / "meta.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] meta.json -> {meta_json}")


if __name__ == "__main__":
    main()
