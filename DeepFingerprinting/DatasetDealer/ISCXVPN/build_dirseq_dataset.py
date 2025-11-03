#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dirseq_dataset.py
读取 label_map.csv -> 增量解析 pcaps -> 合并输出 dirseq 产物（±1 方向序列，截断但不 padding）。
"""

from __future__ import annotations
import os, io, gzip, json, csv, time, struct, hashlib, socket, datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime, UTC

# ===== 配置 =====
DATASET_KEY     = "iscx"    # 需与 make_label_map.py 相同
SCHEMA_KEY      = "dirseq"  # 固定
ART_ROOT        = "artifacts"

# 解析策略（变更会触发重算）
PAYLOAD_ONLY    = False     # False: TCP/UDP所有包；True: 仅有负载的包
MIN_LEN         = 3         # 短于此包数的流丢弃
TRUNCATE_LEN    = 5000      # None 不截断；整数则超出截断到该值（不 padding）
EXCLUDE_L4      = {("udp", 5353)}  # (proto,port) 黑名单（整条流丢弃）
EXCLUDE_MCAST_IPS = {"224.0.0.251", "ff02::fb", "ff05::fb"}

# 缓存策略
USE_SHA256      = False     # 文件指纹是否包含 sha256；更稳但慢

PCAP_EXTS       = (".pcap", ".pcapng", ".pcap.gz", ".pcapng.gz")
# ==================

# --- I/O utils ---
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def now_iso() -> str:
    # 2025-10-31T08:30:00Z 这种格式，秒级，无微秒
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

def file_fp(path: str, use_sha256: bool=False) -> Dict[str,object]:
    st = os.stat(path); out = {"size": int(st.st_size), "mtime": int(st.st_mtime), "sha256": None}
    if use_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out

def cfg_sig() -> str:
    cfg = dict(PAYLOAD_ONLY=PAYLOAD_ONLY, MIN_LEN=MIN_LEN,
               TRUNCATE_LEN=TRUNCATE_LEN,
               EXCLUDE_L4=sorted(list(EXCLUDE_L4)), EXCLUDE_MCAST_IPS=sorted(list(EXCLUDE_MCAST_IPS)))
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()

# --- pcap readers ---
PCAP_MAGIC = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
PCAPNG_MAGIC = 0x0A0D0D0A
GZIP_MAGIC = b"\x1f\x8b"
try:
    import dpkt
except Exception as e:
    raise SystemExit("需要安装 dpkt: pip install dpkt") from e

def open_maybe_gzip(path: str):
    f = open(path, "rb"); head = f.read(2); f.seek(0)
    if head == GZIP_MAGIC:
        data = gzip.open(f).read(); f.close(); return io.BytesIO(data)
    return f

def iter_packets(path: str):
    f = open_maybe_gzip(path)
    head4 = f.read(4); f.seek(0)
    magic = struct.unpack(">I", head4)[0] if len(head4)==4 else None
    try:
        if magic in PCAP_MAGIC:
            r = dpkt.pcap.Reader(f)
            for _, buf in r: yield buf
        elif magic == PCAPNG_MAGIC:
            r = dpkt.pcapng.Reader(f)
            for item in r:
                if isinstance(item,(list,tuple)) and len(item)>=2: yield item[1]
        else:
            try:
                r = dpkt.pcap.Reader(f)
                for _, buf in r: yield buf
            except Exception:
                f.seek(0); r = dpkt.pcapng.Reader(f)
                for item in r:
                    if isinstance(item,(list,tuple)) and len(item)>=2: yield item[1]
    finally:
        f.close()

def parse_ip_l4(pkt: bytes):
    try:
        eth = dpkt.ethernet.Ethernet(pkt); ip = eth.data
    except Exception:
        ip = None
    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            sll = dpkt.sll.SLL(pkt); ip = sll.data
        except Exception:
            return None
    if isinstance(ip, dpkt.ip.IP):
        af = socket.AF_INET; proto = ip.p; l4 = ip.data
        src = socket.inet_ntop(af, ip.src); dst = socket.inet_ntop(af, ip.dst)
    elif isinstance(ip, dpkt.ip6.IP6):
        af = socket.AF_INET6; proto = ip.nxt; l4 = ip.data
        src = socket.inet_ntop(af, ip.src); dst = socket.inet_ntop(af, ip.dst)
    else:
        return None
    if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        if l4.sport is None or l4.dport is None: return None
        payload_len = len(l4.data or b"")
        return src, int(l4.sport), dst, int(l4.dport), int(proto), int(payload_len)
    return None

PROTO_NAME = {6: "tcp", 17: "udp"}
def is_excluded(proto: int, sport: int, dport: int, src: str, dst: str) -> bool:
    pn = PROTO_NAME.get(proto)
    if pn and ((pn, sport) in EXCLUDE_L4 or (pn, dport) in EXCLUDE_L4):
        return True
    if src in EXCLUDE_MCAST_IPS or dst in EXCLUDE_MCAST_IPS:
        return True
    return False

def pcap_to_dirseqs(path: str) -> List[np.ndarray]:
    flows: Dict[tuple, Dict[str,object]] = {}
    for buf in iter_packets(path):
        rec = parse_ip_l4(buf)
        if rec is None:
            continue
        src, sport, dst, dport, proto, payload_len = rec
        if is_excluded(proto, sport, dport, src, dst):
            continue
        if PAYLOAD_ONLY and payload_len <= 0:
            continue
        a, b = (src, sport), (dst, dport)
        key = (a, b, proto) if a <= b else (b, a, proto)
        ent = flows.get(key)
        if ent is None:
            ent = {"origin": (src, sport, dst, dport), "seq": []}
            flows[key] = ent
        ent["seq"].append(+1 if (src, sport, dst, dport) == ent["origin"] else -1)

    seqs: List[np.ndarray] = []
    for ent in flows.values():
        arr = np.asarray(ent["seq"], dtype=np.int8)
        if TRUNCATE_LEN is not None and len(arr) > TRUNCATE_LEN:
            arr = arr[:TRUNCATE_LEN]
        if len(arr) >= MIN_LEN:
            seqs.append(arr)
    return seqs

# --- main ---
def main():
    ds_dir   = ensure_dir(os.path.join(ART_ROOT, DATASET_KEY))
    map_csv  = os.path.join(ds_dir, "label_map.csv")
    if not os.path.exists(map_csv):
        raise FileNotFoundError(f"未找到映射表: {map_csv}（请先运行 make_label_map.py）")

    # 产物 & 缓存
    out_dir  = ensure_dir(os.path.join(ds_dir, SCHEMA_KEY))
    cache_dir= ensure_dir(os.path.join(out_dir, "cache"))
    cache_idx= os.path.join(out_dir, "cache.json")

    # 读取映射
    pairs: List[Tuple[str,str]] = []
    with open(map_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            p, lab = row["path"], row["label"]
            if os.path.isfile(p) and p.lower().endswith(PCAP_EXTS):
                pairs.append((p, lab))
    if not pairs:
        raise SystemExit("label_map.csv 中没有有效条目")

    # 缓存索引
    cache = {"version":1, "config_sig": cfg_sig(), "items": {}}
    if os.path.exists(cache_idx):
        try:
            with open(cache_idx, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass
    cache.setdefault("items", {})
    items: Dict[str,dict] = cache["items"]

    cfg_changed = (cache.get("config_sig") != cfg_sig())

    # 解析或跳过
    new_cnt = rep_cnt = skip_cnt = 0
    for pcap_path, label in pairs:
        key = os.path.abspath(pcap_path)
        fp  = file_fp(pcap_path, USE_SHA256)
        npz_name = hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz"
        npz_path = os.path.join(cache_dir, npz_name)

        need = False; reason = "new"
        ent = items.get(key)
        if ent is None:
            need = True
        else:
            same = (ent.get("size")==fp["size"] and ent.get("mtime")==fp["mtime"]
                    and (not USE_SHA256 or ent.get("sha256")==fp["sha256"]))
            has_npz = os.path.exists(npz_path)
            same_cfg= (ent.get("config_sig")==cfg_sig())
            if not same or not has_npz or (cfg_changed and not same_cfg):
                need = True; reason = "modified" if same else "changed"

        if need:
            try:
                seqs = pcap_to_dirseqs(pcap_path)
                np.savez(npz_path, X=np.array(seqs, dtype=object), allow_pickle=True)
                items[key] = dict(path=pcap_path, label=label,
                                  size=fp["size"], mtime=fp["mtime"], sha256=fp["sha256"],
                                  npz=os.path.basename(npz_path), count=len(seqs),
                                  config_sig=cfg_sig(), updated_at=now_iso(), reason=reason)
                if reason == "new": new_cnt += 1
                else: rep_cnt += 1
            except Exception as e:
                items[key] = dict(path=pcap_path, label=label,
                                  size=fp["size"], mtime=fp["mtime"], sha256=fp["sha256"],
                                  npz=None, count=0, error=str(e),
                                  config_sig=cfg_sig(), updated_at=now_iso(), reason="error")
        else:
            # 标签变更也允许更新
            if ent.get("label") != label:
                ent["label"] = label; ent["updated_at"] = now_iso()
            skip_cnt += 1

    cache["items"] = items; cache["config_sig"] = cfg_sig()
    with open(cache_idx, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"[cache] new={new_cnt} reparse={rep_cnt} skip={skip_cnt} indexed={len(items)}")

    # 合并 -> 产物
    label2files: Dict[str, List[str]] = {}
    for it in items.values():
        if it.get("npz") and it.get("count",0)>0:
            label2files.setdefault(it["label"], []).append(os.path.join(cache_dir, it["npz"]))
    labels = sorted(label2files.keys())
    label2id = {lab:i for i,lab in enumerate(labels)}
    id2label = {str(i):lab for lab,i in label2id.items()}

    X_all: List[np.ndarray] = []; y_all: List[int] = []
    for lab in labels:
        tot = 0
        for npz in label2files[lab]:
            try:
                obj = np.load(npz, allow_pickle=True)
                X = obj["X"]
                seqs = [np.asarray(x, dtype=np.int8) for x in (list(X) if X.dtype==object else X)]
                X_all.extend(seqs); y_all.extend([label2id[lab]]*len(seqs)); tot += len(seqs)
            except Exception as e:
                print(f"[merge-warn] {npz}: {e}")
        print(f"  - {lab:24s} -> {tot}")

    # 写出
    np.savez(os.path.join(out_dir, "data.npz"),
             X=np.array(X_all, dtype=object), y=np.asarray(y_all, dtype=np.int64), allow_pickle=True)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "class_count.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["class","class_id","count"])
        for lab, cid in label2id.items():
            wr.writerow([lab, cid, sum(1 for v in y_all if v==cid)])
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": DATASET_KEY, "schema": SCHEMA_KEY, "created_at": now_iso(),
            "config": dict(PAYLOAD_ONLY=PAYLOAD_ONLY, MIN_LEN=MIN_LEN,
                           TRUNCATE_LEN=TRUNCATE_LEN,
                           EXCLUDE_L4=sorted(list(EXCLUDE_L4)),
                           EXCLUDE_MCAST_IPS=sorted(list(EXCLUDE_MCAST_IPS)))
        }, f, ensure_ascii=False, indent=2)

    print(f"[done] artifact -> {out_dir}/data.npz, labels.json, class_count.csv, meta.json")
    print(f"[cache dir] {os.path.join(out_dir, 'cache')}")

if __name__ == "__main__":
    main()
