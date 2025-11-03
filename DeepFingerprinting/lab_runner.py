#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab_runner.py
多数据集 × 多模型 的极简编排与缓存化数据处理（参数全部写在代码里）：
- artifacts/<dataset>/<schema>/ 里存数据产物，供任意模型复用
- models 声明自己需要的 schema；runner 自动确保产物存在，再开训
- 支持大批量 pcap 的增量缓存：OUTDIR/cache/ 每个 pcap 独立 npz，按 size+mtime(+sha256) 与配置签名判断是否跳过
- 内置一个 schema：dirseq（±1 方向序列），和一个模型训练器：DFNoDefNet（你已有定义）
"""
from __future__ import annotations
import os, io, gzip, json, csv, time, struct, hashlib, socket, datetime, random
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# =============== 你的模型实现（已提供） ===============
# 确保同目录存在 Model_NoDef_pytorch.py，内含 class DFNoDefNet(nn.Module)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from Model_NoDef_pytorch import DFNoDefNet
# ====================================================


# =============== 第 0 部分：全局配置（你只改这里） ===============
ROOT_OUT      = "artifacts"         # 所有数据集产物的根目录
RUNS_DIR      = "runs"              # 训练产物根目录（权重/日志/指标）
PROTO_NAME = {6: "tcp", 17: "udp"}  # 其他协议默认不过滤

# 数据集注册表：每个条目会产出一个独立的 artifact（与模型解耦）
# key: 数据集ID； sources: [(路径或目录, 标签)]；schema_opts: 该 schema 的构建选项
DATASETS: Dict[str, Dict] = {
    "iscx_bin": {
        "schema": "dirseq",
        "sources": [
            ("/netdisk/dataset/public_dataset/ISCX-VPN-NonVPN-2016", "sftp"),
            ("/home/pcz/DL/ML&DL/Dataset/novpn/newjersey_ubuntu24.04_novpn_20250116_083705_abc.com.pcap", "abc.com"),
        ],
        "schema_opts": dict(
            # 方向序列抽取
            PAYLOAD_ONLY=False,  # False=统计所有TCP/UDP包；True=仅有负载的包
            MIN_LEN=3,           # 短于该长度的流丢弃

            # ——新—— 协议+端口黑名单（整条流丢弃）
            # 下面示例等价于原来排除 mDNS(UDP/5353)；你也可加 ('udp',5355) LLMNR、('udp',1900) SSDP 等
            EXCLUDE_L4=[('udp', 5353)],

            # 组播地址黑名单（整条流丢弃）
            EXCLUDE_MCAST_IPS=["224.0.0.251", "ff02::fb", "ff05::fb"],

            # ——新—— 仅做“长度截断”：
            #   若为 None：不截断（保持原始变长）
            #   若为整数：长度超过则截断到该值；不足的不处理（不 padding）
            TRUNCATE_LEN=5000,

            # 缓存指纹是否含 sha256（更稳但慢）
            USE_SHA256=False,
        ),
    },
}

# 模型注册表：声明所需 schema，并给出训练器函数（下面定义）
MODELS: Dict[str, Dict] = {
    "df_nodf": {
        "needs_schema": "dirseq",
        "trainer": "trainer_df_nodf",   # 调用下方同名函数
        "train_opts": dict(
            SEED=2025, EPOCHS=30, BATCH_SIZE=16, MAX_LEN=5000, LR=0.002,
            VAL_RATIO=0.05, TEST_RATIO=0.20,
            NUM_WORKERS=2, USE_CLASS_WEIGHT=True,
            USE_WEIGHTED_SAMPLER=True, ENABLE_AMP=True,
            BALANCED_BATCH=True,       # 每个 epoch 近似平衡采样（更抗不均衡）
        ),
    },
    # 你可以再注册别的模型（声明需要的 schema，并实现对应 trainer_*）
}

# 需要跑哪些组合（模型 × 数据集）
EXPERIMENTS: List[Tuple[str, str]] = [
    ("df_nodf", "iscx_bin"),
    # ("df_nodf", "another_ds"),
]
# ===========================================================


# =============== 第 1 部分：通用工具 ===============
def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def sha1_bytes(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()

def file_fp(path: str, use_sha256: bool=False) -> Dict[str, object]:
    st = os.stat(path)
    out = {"size": int(st.st_size), "mtime": int(st.st_mtime), "sha256": None}
    if use_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out

def config_sig(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return sha1_bytes(blob)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =============== 第 2 部分：schema 构建（先实现 dirseq） ===============
try:
    import dpkt
except Exception as e:
    raise SystemExit("需要 `dpkt`：pip install dpkt") from e

PCAP_MAGIC = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
PCAPNG_MAGIC = 0x0A0D0D0A
GZIP_MAGIC = b"\x1f\x8b"
PCAP_FILE_EXT = (".pcap", ".pcapng", ".pcap.gz", ".pcapng.gz")

def open_maybe_gzip(path: str):
    f = open(path, "rb")
    head = f.read(2); f.seek(0)
    if head == GZIP_MAGIC:
        data = gzip.open(f).read(); f.close()
        return io.BytesIO(data)
    return f

def iter_raw_packets(path: str):
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
                if isinstance(item, (list, tuple)) and len(item)>=2: yield item[1]
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

def inet_str(x: bytes, af: int) -> str:
    import socket
    return socket.inet_ntop(af, x)

def parse_ip_l4(pkt_bytes: bytes):
    try:
        eth = dpkt.ethernet.Ethernet(pkt_bytes); ip_pkt = eth.data
    except Exception:
        ip_pkt = None
    if not isinstance(ip_pkt, (dpkt.ip.IP, dpkt.ip6.IP6)):
        try:
            sll = dpkt.sll.SLL(pkt_bytes); ip_pkt = sll.data
        except Exception:
            return None
    if isinstance(ip_pkt, dpkt.ip.IP):
        import socket
        src, dst, proto, l4 = inet_str(ip_pkt.src, socket.AF_INET), inet_str(ip_pkt.dst, socket.AF_INET), ip_pkt.p, ip_pkt.data
    elif isinstance(ip_pkt, dpkt.ip6.IP6):
        import socket
        src, dst, proto, l4 = inet_str(ip_pkt.src, socket.AF_INET6), inet_str(ip_pkt.dst, socket.AF_INET6), ip_pkt.nxt, ip_pkt.data
    else:
        return None
    if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        if l4.sport is None or l4.dport is None: return None
        payload_len = len(l4.data or b"")
        return src, int(l4.sport), dst, int(l4.dport), int(proto), int(payload_len)
    return None

def scan_sources(sources: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    out = []
    for root, lab in sources:
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.isdir(root):
            for dp, _, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(PCAP_FILE_EXT):
                        out.append((os.path.join(dp, fn), lab))
        elif os.path.isfile(root) and root.lower().endswith(PCAP_FILE_EXT):
            out.append((root, lab))
    # 去重复（最后一次标签覆盖）
    m = {}
    for p, lab in out: m[p] = lab
    return sorted(m.items(), key=lambda x: x[0])

def build_dirseq_artifact(dataset_key: str, cfg: dict) -> str:
    """
    构建 dirseq（±1 方向序列）schema 的数据产物：
    - 缓存目录：artifacts/<dataset>/dirseq/cache/
    - 产物目录：artifacts/<dataset>/dirseq/
    """
    ds = DATASETS[dataset_key]
    assert ds["schema"] == "dirseq"
    opts = ds["schema_opts"].copy()
    PAYLOAD_ONLY = bool(opts.get("PAYLOAD_ONLY", True))
    MIN_LEN = int(opts.get("MIN_LEN", 20))

    # 新：协议+端口的黑名单，统一成 set[(proto, port)]
    _raw_ex_l4 = opts.get("EXCLUDE_L4", [])
    EXCLUDE_L4 = set((str(p).lower(), int(port)) for p, port in _raw_ex_l4)

    EXCLUDE_MCAST_IPS = set(opts.get("EXCLUDE_MCAST_IPS", []))

    # 新：仅做截断；None 表示不截断
    TRUNCATE_LEN = (int(opts["TRUNCATE_LEN"]) if "TRUNCATE_LEN" in opts and opts["TRUNCATE_LEN"] is not None else None)

    USE_SHA256 = bool(opts.get("USE_SHA256", False))

    # 产物路径
    ds_dir   = ensure_dir(os.path.join(ROOT_OUT, dataset_key, "dirseq"))
    cache_dir= ensure_dir(os.path.join(ds_dir, "cache"))
    cache_idx_path = os.path.join(ds_dir, "cache.json")

    # 配置签名（决定是否重算）
    cfg_affect = dict(
        schema="dirseq",
        PAYLOAD_ONLY=PAYLOAD_ONLY,
        MIN_LEN=MIN_LEN,
        EXCLUDE_L4=sorted(list(EXCLUDE_L4)),
        EXCLUDE_MCAST_IPS=sorted(EXCLUDE_MCAST_IPS),
        TRUNCATE_LEN=TRUNCATE_LEN, )
    cfg_sig = config_sig(cfg_affect)

    # 读取/初始化缓存索引
    cache = {"version": 1, "config_sig": cfg_sig, "items": {}}
    if os.path.exists(cache_idx_path):
        try: cache = load_json(cache_idx_path)
        except Exception: pass
        cache.setdefault("items", {})
        cache.setdefault("config_sig", cfg_sig)

    items = cache["items"]

    # 扫描源
    jobs = scan_sources(ds["sources"])
    print(f"[{dataset_key}] scan -> {len(jobs)} files")

    def is_excluded(proto: int, sport: int, dport: int, src: str, dst: str) -> bool:
        # 协议+端口黑名单
        pn = PROTO_NAME.get(proto)
        if pn is not None:
            if (pn, sport) in EXCLUDE_L4 or (pn, dport) in EXCLUDE_L4:
                return True
        # 组播地址黑名单
        if src in EXCLUDE_MCAST_IPS or dst in EXCLUDE_MCAST_IPS:
            return True
        return False

    def pcap_to_dirseqs(path: str) -> List[np.ndarray]:
        flows = {}
        for buf in iter_raw_packets(path):
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
            # ——新：长度截断（不做 padding）——
            if TRUNCATE_LEN is not None and len(arr) > TRUNCATE_LEN:
                arr = arr[:TRUNCATE_LEN]
            # 最后再做最小长度过滤
            if len(arr) >= MIN_LEN:
                seqs.append(arr)
        return seqs

    # 遍历解析（缓存化）
    new_cnt = rep_cnt = skip_cnt = 0
    for pcap_path, label in tqdm(jobs, ncols=100, desc=f"{dataset_key}/pcap"):
        key = os.path.abspath(pcap_path)
        fp  = file_fp(pcap_path, USE_SHA256)
        npz_name = hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz"
        npz_path = os.path.join(cache_dir, npz_name)

        need = False
        reason = "new"
        if key not in items:
            need = True
        else:
            it = items[key]
            same_size = (it.get("size")==fp["size"])
            same_mtime= (it.get("mtime")==fp["mtime"])
            same_sha  = (not USE_SHA256) or (it.get("sha256")==fp["sha256"])
            same_cfg  = (it.get("config_sig")==cfg_sig)
            if not (same_size and same_mtime and same_sha):
                need=True; reason="modified"
            elif not same_cfg:
                need=True; reason="config_changed"
            elif not os.path.exists(npz_path):
                need=True; reason="npz_missing"

        if need:
            try:
                seqs = pcap_to_dirseqs(pcap_path)

                # ✅ 仅保存“截断后但未 padding”的变长序列；不再做任何滑窗切段增广
                np.savez(npz_path, X=np.array(seqs, dtype=object), allow_pickle=True)

                items[key] = dict(path=pcap_path, label=label, size=fp["size"], mtime=fp["mtime"], sha256=fp["sha256"],
                    npz=os.path.basename(npz_path), count=len(seqs), config_sig=cfg_sig, updated_at=now_iso(),
                    reason=reason)
                if reason == "new":
                    new_cnt += 1
                else:
                    rep_cnt += 1
            except Exception as e:
                items[key] = dict(path=pcap_path, label=label, size=fp["size"], mtime=fp["mtime"], sha256=fp["sha256"],
                    npz=None, count=0, error=str(e), config_sig=cfg_sig, updated_at=now_iso(), reason="error")
        else:
            # 更新标签也允许（不重算）
            if items[key].get("label") != label:
                items[key]["label"] = label; items[key]["updated_at"]=now_iso()
            skip_cnt+=1

    cache["items"] = items; cache["config_sig"] = cfg_sig; save_json(cache_idx_path, cache)
    print(f"[{dataset_key}] cache: new={new_cnt} reparse={rep_cnt} skip={skip_cnt} indexed={len(items)}")

    # 合并缓存 -> artifact
    label_to_files: Dict[str, List[str]] = defaultdict(list)
    for it in items.values():
        if it.get("npz") and it.get("count",0)>0:
            label_to_files[it["label"]].append(os.path.join(cache_dir, it["npz"]))
    labels = sorted(label_to_files.keys())
    label2id = {lab:i for i,lab in enumerate(labels)}
    id2label = {str(i):lab for lab,i in label2id.items()}

    X_all, y_all = [], []
    for lab in labels:
        tot = 0
        for npz in label_to_files[lab]:
            try:
                obj = np.load(npz, allow_pickle=True)
                X = obj["X"]
                seqs = [np.asarray(x, dtype=np.int8) for x in (list(X) if X.dtype==object else X)]
                X_all.extend(seqs); y_all.extend([label2id[lab]]*len(seqs)); tot += len(seqs)
            except Exception as e:
                print(f"[merge-warn] {npz}: {e}")
        print(f"  - {lab:20s} -> {tot}")

    # 写产物
    art_dir = ds_dir  # = artifacts/<dataset>/dirseq
    np.savez(os.path.join(art_dir, "data.npz"),
             X=np.array(X_all, dtype=object), y=np.asarray(y_all, dtype=np.int64),
             allow_pickle=True)
    save_json(os.path.join(art_dir, "labels.json"), {"label2id": label2id, "id2label": id2label})
    with open(os.path.join(art_dir, "class_count.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["class","class_id","count"])
        for lab, cid in label2id.items():
            wr.writerow([lab, cid, sum(1 for yy in y_all if yy==cid)])
    save_json(os.path.join(art_dir, "meta.json"), dict(
        dataset=dataset_key, schema="dirseq", created_at=now_iso(), config=cfg_affect))

    print(f"[{dataset_key}] artifact -> {art_dir}/data.npz, labels.json, class_count.csv, meta.json")
    return art_dir  # 返回该数据集该 schema 的产物目录

SCHEMAS: Dict[str, Callable[[str, dict], str]] = {
    # schema_key -> builder(dataset_key, dataset_cfg) -> artifact_dir
    "dirseq": build_dirseq_artifact,
}
# =============================================================


# =============== 第 3 部分：模型训练器（示例：DFNoDefNet@dirseq） ===============
class DirSeqDataset(Dataset):
    def __init__(self, X: List[np.ndarray], y: Optional[np.ndarray], max_len: int):
        self.X = X; self.y = y; self.max_len = max_len
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        seq = self.X[idx]
        if len(seq) >= self.max_len:
            arr = seq[:self.max_len]
        else:
            arr = np.zeros(self.max_len, dtype=np.int8)
            arr[:len(seq)] = seq
        x = torch.from_numpy(arr.astype(np.float32, copy=False))
        y = -1 if self.y is None else int(self.y[idx])
        return x, y

def stratified_indices(y: np.ndarray, vr: float, tr: float, seed: int):
    rng = np.random.default_rng(seed)
    cls = np.unique(y); tr_idx, va_idx, te_idx = [], [], []
    for c in cls:
        idx = np.where(y==c)[0]; rng.shuffle(idx); n=len(idx)
        n_te = max(1, int(round(n*tr))) if n>=3 else (1 if n>=2 else 0)
        n_va = max(1, int(round(n*vr))) if n>=3 else (0 if n==2 else 0)
        if n_te+n_va >= n: n_te = 1 if n>=2 else 0; n_va = 0
        te_idx.extend(idx[:n_te]); va_idx.extend(idx[n_te:n_te+n_va]); tr_idx.extend(idx[n_te+n_va:])
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)
    return tr_idx, va_idx, te_idx

def make_loaders_dirseq(X: List[np.ndarray], y: np.ndarray, max_len: int,
                        VAL_RATIO: float, TEST_RATIO: float, BATCH_SIZE: int,
                        NUM_WORKERS:int, USE_WEIGHTED_SAMPLER: bool, BALANCED_BATCH: bool):
    base = DirSeqDataset(X, y, max_len)
    tr_idx, va_idx, te_idx = stratified_indices(y, VAL_RATIO, TEST_RATIO, seed=2025)

    def _dist(ix):
        vv, cc = np.unique(y[ix], return_counts=True)
        return dict(zip(map(int,vv), map(int,cc)))

    print(f"split -> train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    print("train dist:", _dist(tr_idx)); print("val   dist:", _dist(va_idx)); print("test  dist:", _dist(te_idx))

    # 训练集采样
    if BALANCED_BATCH:
        # 每 epoch 近似平衡各类样本数量
        from math import inf
        cls_to = defaultdict(list)
        for i in tr_idx: cls_to[int(y[i])].append(i)
        k = min(len(v) for v in cls_to.values())  # 最小类的数量
        chosen = []
        for _, idxs in cls_to.items():
            np.random.shuffle(idxs); chosen.extend(idxs[:k])
        np.random.shuffle(chosen)
        train_loader = DataLoader(Subset(base, chosen), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    elif USE_WEIGHTED_SAMPLER:
        uniq, cts = np.unique(y[tr_idx], return_counts=True)
        counts = np.zeros(int(np.max(y))+1, dtype=np.int64); counts[uniq] = cts
        w = counts.astype(np.float32); w[w==0]=1.0; w = 1.0/np.sqrt(w)
        sample_w = w[y[tr_idx]]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                        num_samples=len(tr_idx), replacement=True)
        train_loader = DataLoader(Subset(base, tr_idx), batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(Subset(base, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)

    val_loader  = DataLoader(Subset(base, va_idx), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(Subset(base, te_idx), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, tr_idx

def trainer_df_nodf(artifact_dir: str, out_dir: str, opts: dict):
    """
    使用 DFNoDefNet 训练/测试 dirseq 产物
    artifact_dir: artifacts/<dataset>/dirseq/
    out_dir     : runs/<model>/<dataset>/
    """
    set_seed(int(opts.get("SEED", 2025)))
    ensure_dir(out_dir)

    # 载入数据
    obj = np.load(os.path.join(artifact_dir, "data.npz"), allow_pickle=True)
    X = obj["X"]; X = [np.asarray(x, dtype=np.int8) for x in (list(X) if X.dtype==object else X)]
    y = obj["y"].astype(np.int64)
    labels = load_json(os.path.join(artifact_dir, "labels.json"))["id2label"]
    id2label = {int(k): v for k, v in labels.items()}
    num_classes = int(np.max(y)) + 1
    print(f"samples={len(X)}  classes={num_classes}")

    # DataLoaders
    train_loader, val_loader, test_loader, tr_idx = make_loaders_dirseq(
        X, y, int(opts["MAX_LEN"]), float(opts["VAL_RATIO"]), float(opts["TEST_RATIO"]),
        int(opts["BATCH_SIZE"]), int(opts["NUM_WORKERS"]),
        bool(opts["USE_WEIGHTED_SAMPLER"]), bool(opts["BALANCED_BATCH"])
    )

    # 类权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(opts["USE_CLASS_WEIGHT"]):
        uniq, cts = np.unique(y[tr_idx], return_counts=True)
        counts = np.zeros(num_classes, dtype=np.int64); counts[uniq] = cts
        w = counts.astype(np.float32); w[w==0]=1.0; w = 1.0/np.sqrt(w)
        w = w*(num_classes/w.sum())
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("class_weight =", w.tolist())
    else:
        class_weight = None

    # 模型
    model = DFNoDefNet()
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 classifier 层")
    if model.classifier.out_features != num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)

    # 训练配置
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=float(opts["LR"]))
    scaler = GradScaler('cuda', enabled=bool(opts["ENABLE_AMP"]) and device.type=='cuda')

    # 日志
    with open(os.path.join(out_dir, "train_log.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    def run_epoch(loader, train: bool):
        model.train(train)
        n_tot, loss_sum = 0, 0.0
        y_true, y_pred = [], []
        for xb, yb in tqdm(loader, ncols=100, disable=not train):
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.set_grad_enabled(train):
                if train and scaler is not None and bool(opts["ENABLE_AMP"]) and device.type=='cuda':
                    with autocast('cuda'):
                        logits = model(xb); loss = criterion(logits, yb)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    logits = model(xb); loss = criterion(logits, yb)
                    if train:
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            n = xb.size(0); loss_sum += loss.item()*n; n_tot += n
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        acc = float((y_true==y_pred).mean())
        return loss_sum/max(1,n_tot), acc, y_true, y_pred

    # 训练
    best_val, best_ckpt = -1.0, None
    E = int(opts["EPOCHS"])
    for epoch in range(1, E+1):
        print(f"\n===== Epoch {epoch}/{E} =====")
        tr_loss, tr_acc, *_ = run_epoch(train_loader, True)
        va_loss, va_acc, *_ = run_epoch(val_loader, False)
        print(f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        with open(os.path.join(out_dir, "train_log.csv"), "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{va_loss:.6f},{va_acc:.6f}\n")
        if va_acc > best_val:
            best_val = va_acc
            best_ckpt = {"model": model.state_dict(), "cfg":{"num_classes":num_classes,"MAX_LEN":int(opts["MAX_LEN"])}}
            torch.save(best_ckpt, os.path.join(out_dir, "best.pt"))
            print(f"[saved best] -> {os.path.join(out_dir, 'best.pt')} (val_acc={best_val:.4f})")

    # 测试
    if best_ckpt is None:
        best_ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    te_loss, te_acc, y_true, y_pred = run_epoch(test_loader, False)
    print("\n──────── Closed-World Test ────────")
    print(f"test loss={te_loss:.4f}  acc={te_acc:.4f}")

    # 指标/混淆矩阵
    def confusion(y_t, y_p, ncls):
        m = np.zeros((ncls,ncls), dtype=np.int64)
        for t,p in zip(y_t, y_p): m[int(t), int(p)] += 1
        return m
    def class_report(y_t, y_p, ncls):
        rep = {}
        for c in range(ncls):
            tp = int(((y_t==c)&(y_p==c)).sum())
            fp = int(((y_t!=c)&(y_p==c)).sum())
            fn = int(((y_t==c)&(y_p!=c)).sum())
            tn = int(((y_t!=c)&(y_p!=c)).sum())
            prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn)
            f1 = 2*prec*rec / max(1e-12, prec+rec); acc = (tp+tn)/max(1,tp+fp+fn+tn)
            rep[c] = {"precision":prec,"recall":rec,"f1":f1,"acc":acc,"tp":tp,"fp":fp,"fn":fn,"tn":tn}
        return rep

    cm = confusion(y_true, y_pred, num_classes).tolist()
    rep = {id2label.get(k,str(k)): {kk: float(vv) for kk,vv in class_report(y_true,y_pred,num_classes)[k].items()}
           for k in range(num_classes)}
    save_json(os.path.join(out_dir, "metrics.json"), {
        "best_val_acc": float(best_val), "test_loss": float(te_loss), "test_acc": float(te_acc),
        "per_class": rep, "confusion_matrix": cm, "epochs": int(opts["EPOCHS"]),
        "batch_size": int(opts["BATCH_SIZE"]), "max_len": int(opts["MAX_LEN"]), "lr": float(opts["LR"]),
        "use_class_weight": bool(opts["USE_CLASS_WEIGHT"]),
        "use_weighted_sampler": bool(opts["USE_WEIGHTED_SAMPLER"]),
        "balanced_batch": bool(opts["BALANCED_BATCH"]),
    })
    with open(os.path.join(out_dir, "test_predictions.csv"), "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for t,p in zip(y_true.tolist(), y_pred.tolist()):
            f.write(f"{t},{p}\n")

# =============== 第 4 部分：编排器（实验矩阵） ===============
def run():
    for model_key, dataset_key in EXPERIMENTS:
        m = MODELS[model_key]
        ds = DATASETS[dataset_key]
        need_schema = m["needs_schema"]
        assert ds["schema"] == need_schema, f"{model_key} 需要 {need_schema}，但 {dataset_key} 提供 {ds['schema']}"

        # 1) 确保产物存在（按 schema 构建 + 缓存）
        builder = SCHEMAS[need_schema]
        art_dir = builder(dataset_key, ds)

        # 2) 训练/测试
        out_dir = ensure_dir(os.path.join(RUNS_DIR, model_key, dataset_key))
        trainer_fn = globals()[m["trainer"]]
        print(f"\n=== Train {model_key} on {dataset_key} ===")
        trainer_fn(artifact_dir=art_dir, out_dir=out_dir, opts=m["train_opts"].copy())
        print(f"[done] outputs -> {out_dir}\n")

if __name__ == "__main__":
    run()
