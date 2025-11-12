#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal yet efficient trainer for DFNoDefNet using streamed NPZs.

Assumptions:
- Each .npz contains arrays: 'flows' (object array of variable-length int8 seqs with values -1/1),
  and 'labels' (object array of str label names).
- A global labels.json provides label2id/id2label mapping.
- Model is available as: from Model_NoDef_pytorch import DFNoDefNet

This script:
- Streams NPZs file-by-file to keep RAM low.
- Uses IterableDataset with worker-sharding, per-file shuffling, pinned memory, prefetch, AMP.
- Tracks accuracy + macro-F1 with O(num_classes) memory (TP/FP/FN counters) to avoid huge confusion matrices.
"""
import hashlib
import os, gc, json, sys, time, random
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from collections import Counter
# —— 导入你的 DF 模型（保持工程结构）——
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402
# =========================
# ====== Global Params =====
# =========================
# Paths
NPZ_ROOT: str = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz"
LABELS_JSON: str = "/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/labels.json"
RUNS_DIR: str = "./runs/df_npz"
SAVE_BEST_PATH: str = str(Path(RUNS_DIR, "best.pt"))
LOG_INTERVAL_STEPS: int = 200  # print every X optimizer steps

# Data / batching
INPUT_LEN: int = 5000            # truncate/pad each sequence to this length
BATCH_SIZE: int = 2048
NUM_WORKERS: int = 4             # <=4 建议；每个 worker 只加载自己负责的文件，避免重复加载
PREFETCH_FACTOR: int = 4
PIN_MEMORY: bool = True
DROP_LAST: bool = True
TRAIN_VAL_SPLIT: float = 0.9     # split by files
SHUFFLE_FILES: bool = True       # shuffle file order each epoch
SHUFFLE_WITHIN_FILE: bool = True # shuffle sample indices within file (per-file permutation)

# Training
EPOCHS: int = 30
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTH: float = 0.0        # set >0 if desired (e.g. 0.1)
GRAD_CLIP_NORM: float = 1.0
ACCUM_STEPS: int = 1             # gradient accumulation
USE_COMPILE: bool = True         # torch.compile for model (PyTorch 2.0+)
SEED: int = 42

TRAIN_RATIO: float = 0.8
VAL_RATIO: float   = 0.1
TEST_RATIO: float  = 0.1
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

HASH_SALT: str = f"df-split-v1-{SEED}"  # 影响划分但对每次运行稳定

USE_CLASS_SUBSET: bool = True      # 先开着，想全量训练就改为 False
MAX_CLASSES: int = 100
CLASS_SUBSET_MODE: str = "first_k" # 可选: "first_k" | "alphabetical" | "random_k" | "whitelist"
CLASS_WHITELIST: list[str] = []    # 当 mode="whitelist" 时，填域名列表
FILTER_FILES_BY_CLASS: bool = True # 过滤掉非子集类别的 .npz 文件（更快）

MIN_SAMPLES_PER_CLASS: int = 10_000
VERIFY_SPLIT_COUNTS: bool = False       # 如要核对各类在train/val/test的实际数量，设True会再扫描一次
# AMP / GPU
USE_BF16_IF_AVAILABLE: bool = True  # 优先 bfloat16（无需GradScaler），否则 fallback 为 float16+GradScaler
CUDNN_BENCHMARK: bool = True

# =========================
# ======   Utilities  =====
# =========================
def _id2label_lookup(id2label: dict, i: int) -> str:
    # id2label 的 key 可能是 str 也可能是 int，做个兼容
    return id2label.get(i, id2label.get(str(i)))

def build_class_subset(label2id_full: dict, id2label_full: dict) -> tuple[dict, dict, set[str]]:
    labels_all = list(label2id_full.keys())

    if not USE_CLASS_SUBSET or MAX_CLASSES is None:
        # 不裁剪
        # 统一把 id2label 的 key 转成 int，便于后续使用
        id2label_norm = {int(k): v for k, v in id2label_full.items()} if any(isinstance(k, str) for k in id2label_full.keys()) else id2label_full
        return label2id_full, id2label_norm, set(labels_all)

    if CLASS_SUBSET_MODE == "first_k":
        # 按原 id 顺序取前 K
        all_ids = sorted(int(v) for v in label2id_full.values())
        kept_ids = all_ids[:MAX_CLASSES]
        kept_labels = [_id2label_lookup(id2label_full, i) for i in kept_ids]
    elif CLASS_SUBSET_MODE == "alphabetical":
        kept_labels = sorted(labels_all)[:MAX_CLASSES]
    elif CLASS_SUBSET_MODE == "random_k":
        rng = random.Random(SEED)
        tmp = sorted(labels_all)
        rng.shuffle(tmp)
        kept_labels = tmp[:MAX_CLASSES]
    elif CLASS_SUBSET_MODE == "whitelist":
        kept_labels = [lbl for lbl in CLASS_WHITELIST if lbl in label2id_full]
    else:
        kept_labels = sorted(labels_all)[:MAX_CLASSES]

    # 重新映射到 0..K-1
    label2id_new = {lbl: i for i, lbl in enumerate(kept_labels)}
    id2label_new = {i: lbl for lbl, i in label2id_new.items()}
    return label2id_new, id2label_new, set(kept_labels)

def _file_label_from_path(p: str) -> str:
    # 你的 .npz 路径形如 .../npz/<domain>/<file>.npz
    return os.path.basename(os.path.dirname(p))

def scan_label_freq(files: list[str]) -> Counter:
    """仅扫描 labels 计数，不读 flows，节省内存"""
    cnt = Counter()
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as data:
                for s in data["labels"]:
                    cnt[str(s)] += 1
        except Exception as e:
            print(f"[warn] count-skip {p}: {e}")
    return cnt

def select_labels_topk_with_min_count(
    files: list[str],
    label2id_full: dict,
    k: int,
    min_count: int,
) -> tuple[dict, dict, set[str], Counter]:
    """按频次降序选出 >=min_count 的前 k 个标签，并重新映射到 0..k-1"""
    cnt = scan_label_freq(files)
    candidates = [(lbl, c) for lbl, c in cnt.items() if (lbl in label2id_full and c >= min_count)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    if len(candidates) < k:
        print(f"[warn] only {len(candidates)} labels meet >= {min_count} samples; using all of them.")
    kept = [lbl for (lbl, _) in candidates[:k]]
    if not kept:
        raise RuntimeError(f"No labels meet min_count={min_count}. Lower threshold or check data.")
    label2id_new = {lbl: i for i, lbl in enumerate(kept)}
    id2label_new = {i: lbl for lbl, i in label2id_new.items()}
    return label2id_new, id2label_new, set(kept), cnt

def compute_split_counts(files: list[str], label2id: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """（可选）统计三拆分每类样本数，用与 _stable_hash01/_which_split 同一规则"""
    n = len(label2id)
    tr = np.zeros(n, dtype=np.int64)
    va = np.zeros(n, dtype=np.int64)
    te = np.zeros(n, dtype=np.int64)
    rev = label2id  # 直接查
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                labels = data["labels"]
                for i, s in enumerate(labels):
                    lab = str(s)
                    if lab not in rev:
                        continue
                    p = _stable_hash01(lab, path, int(i))
                    sp = _which_split(p)
                    j = rev[lab]
                    if sp == "train": tr[j] += 1
                    elif sp == "val": va[j] += 1
                    else: te[j] += 1
        except Exception as e:
            print(f"[warn] split-count skip {path}: {e}")
    return tr, va, te

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _stable_hash01(label: str, path: str, idx: int) -> float:
    """将 (label, path, idx, salt) 哈成 [0,1) 的浮点数，跨进程/多worker稳定"""
    s = f"{label}\t{path}\t{idx}\t{HASH_SALT}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    v = int.from_bytes(h, "little")
    return (v & ((1 << 64) - 1)) / float(1 << 64)

def _which_split(p: float) -> str:
    if p < TRAIN_RATIO:
        return "train"
    elif p < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"

def list_npz_files(root: str) -> List[str]:
    root_p = Path(root)
    files = [str(p) for p in root_p.rglob("*.npz")]
    files.sort()
    return files


def load_label_maps(path: str) -> Tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    label2id = obj["label2id"]
    id2label = obj["id2label"]
    # sanitize keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    return label2id, id2label


def split_files(files: List[str], ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files_copy = files[:]
    rng.shuffle(files_copy)
    n_train = int(len(files_copy) * ratio)
    return files_copy[:n_train], files_copy[n_train:]


def _pad_truncate_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    # arr: 1D int8 array with values in {-1, 1}
    n = arr.shape[0]
    if n >= target_len:
        return arr[:target_len].astype(np.float32, copy=False)
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = arr.astype(np.float32, copy=False)
    return out


def collate_pad_to_tensor(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    # batch: list of (np.ndarray[int8], int label_id)
    x_list = []
    y_list = []
    for flow, y in batch:
        x_list.append(_pad_truncate_to_len(flow, INPUT_LEN))
        y_list.append(y)
    x = np.stack(x_list, axis=0)              # (B, L)
    x = np.expand_dims(x, axis=1)             # (B, 1, L) for Conv1d
    X = torch.from_numpy(x)                   # float32
    Y = torch.tensor(y_list, dtype=torch.long)
    return X, Y


# =========================
# ======   Dataset    =====
# =========================

class NPZStreamDataset(IterableDataset):
    """
    Streams samples file-by-file to keep RAM usage low.
    Each worker loads a disjoint subset of files (by modulo slicing).
    We shuffle file order and/or indices within each file if desired.
    """
    def __init__(
        self,
        files: List[str],
        label2id: dict,
        shuffle_files: bool,
        shuffle_within: bool,
        seed: int,
        split: str,                    # <-- 新增：'train' | 'val' | 'test'
    ):
        super().__init__()
        self.files = files
        self.label2id = label2id
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        self.seed = seed
        assert split in {"train", "val", "test"}
        self.split = split

    def _worker_files(self) -> List[str]:
        info = get_worker_info()
        files = self.files
        if info is None:
            return files
        # shard files by worker id
        return [f for i, f in enumerate(files) if (i % info.num_workers) == info.id]

    def _rng(self) -> np.random.Generator:
        info = get_worker_info()
        base = self.seed
        wid = 0 if info is None else info.id + 1
        # time-based entropy to vary epochs even with persistent workers
        return np.random.default_rng(seed=base * 1009 + wid * 9173 + int(time.time()) % 1000000)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        files = self._worker_files()
        rng = self._rng()

        files_local = files[:]
        if self.shuffle_files:
            rng.shuffle(files_local)

        for path in files_local:
            try:
                with np.load(path, allow_pickle=True) as data:
                    flows = data["flows"]
                    labels = data["labels"]
                    n = len(labels)
                    idxs = np.arange(n)
                    if self.shuffle_within:
                        rng.shuffle(idxs)
                    for i in idxs:
                        lab_name = str(labels[i])
                        if lab_name not in self.label2id:
                            continue
                        # —— 核心：按样本稳定哈希到 train/val/test —— #
                        p = _stable_hash01(lab_name, path, int(i))
                        sp = _which_split(p)
                        if sp != self.split:
                            continue

                        flow = np.asarray(flows[i], dtype=np.int8)
                        y = int(self.label2id[lab_name])
                        yield flow, y
                del flows, labels
                gc.collect()
            except Exception as e:
                print(f"[warn] skip file due to error: {path} -> {e}")


# =========================
# ======   Metrics    =====
# =========================

@torch.no_grad()
def compute_macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    # avoid division by zero
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    support = tp + fn
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(f1[mask].mean())


def update_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                  y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    # Vectorized per-batch updates on CPU-backed numpy arrays
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    eq = (yt == yp)
    # TP
    if eq.any():
        np.add.at(tp, yt[eq], 1)
    # FP and FN
    neq = ~eq
    if neq.any():
        np.add.at(fp, yp[neq], 1)
        np.add.at(fn, yt[neq], 1)


# =========================
# ======  Training    =====
# =========================

def build_model(num_classes: int) -> nn.Module:
    # Be tolerant to possible constructor signatures
    try:
        model = DFNoDefNet(num_classes=num_classes)
    except TypeError:
        try:
            model = DFNoDefNet(n_classes=num_classes)
        except TypeError:
            model = DFNoDefNet(num_classes)
    return model


def setup_device_amp() -> Tuple[torch.device, torch.dtype, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    # 仅用【旧 API】控制 TF32，避免与 inductor 的旧接口检查“混用”
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # AMP dtype choice: BF16 (no scaler) if supported; else FP16 (with scaler)
    use_bf16 = bool(device.type == "cuda" and USE_BF16_IF_AVAILABLE and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    need_scaler = (amp_dtype == torch.float16) and (device.type == "cuda")
    return device, amp_dtype, need_scaler


def make_loader(
    files: List[str],
    label2id: dict,
    shuffle_files: bool,
    shuffle_within: bool,
    split: str,                       # <-- 新增
) -> DataLoader:
    ds = NPZStreamDataset(
        files=files,
        label2id=label2id,
        shuffle_files=shuffle_files,
        shuffle_within=shuffle_within,
        seed=SEED,
        split=split,                  # <-- 新增
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=DROP_LAST,
        collate_fn=collate_pad_to_tensor
    )
    return loader

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: "torch.amp.GradScaler | None",
                    device: torch.device,
                    amp_dtype: torch.dtype,
                    num_classes: int,
                    epoch_idx: int) -> Tuple[float, float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH).to(device)

    step = 0
    running_loss = 0.0
    correct = 0
    total = 0

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else torch.no_grad()
    if device.type != "cuda":
        # On CPU, just fall back to standard forward (no autocast)
        class _DummyCtx:
            def __enter__(self): return None
            def __exit__(self, a, b, c): return False
        autocast_ctx = _DummyCtx()

    optimizer.zero_grad(set_to_none=True)
    last_print = time.time()

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(X)  # shape (B, C)
            loss = criterion(logits, Y)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # stats
        batch_loss = float(loss.detach().cpu())
        running_loss += batch_loss
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == Y).sum().item())
        total += Y.size(0)
        update_counts(tp, fp, fn, Y, preds)

        step += 1
        if step % LOG_INTERVAL_STEPS == 0:
            now = time.time()
            speed = LOG_INTERVAL_STEPS / max(now - last_print, 1e-6)
            acc = correct / max(total, 1)
            mf1 = compute_macro_f1(tp, fp, fn)
            avg_loss = running_loss / step
            print(f"[ep{epoch_idx:02d}] step={step} loss={avg_loss:.4f} acc={acc:.4f} mf1={mf1:.4f} ({speed:.1f} it/s)")
            last_print = now

    epoch_loss = running_loss / max(step, 1)
    epoch_acc = correct / max(total, 1)
    epoch_mf1 = compute_macro_f1(tp, fp, fn)
    return epoch_loss, epoch_acc, epoch_mf1


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             amp_dtype: torch.dtype,
             num_classes: int,
             epoch_idx: int) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else torch.no_grad()
    if device.type != "cuda":
        class _DummyCtx:
            def __enter__(self): return None
            def __exit__(self, a, b, c): return False
        autocast_ctx = _DummyCtx()

    steps = 0
    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(X)
            loss = criterion(logits, Y)

        running_loss += float(loss.detach().cpu())
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == Y).sum().item())
        total += Y.size(0)
        update_counts(tp, fp, fn, Y, preds)
        steps += 1

    avg_loss = running_loss / max(steps, 1)
    acc = correct / max(total, 1)
    mf1 = compute_macro_f1(tp, fp, fn)
    print(f"[VAL ep{epoch_idx:02d}] loss={avg_loss:.4f} acc={acc:.4f} mf1={mf1:.4f}")
    return avg_loss, acc, mf1


def main() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    set_global_seed(SEED)

    # Discover files and labels
    files = list_npz_files(NPZ_ROOT)
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files under: {NPZ_ROOT}")

    label2id_full, id2label_full = load_label_maps(LABELS_JSON)

    label2id, id2label, kept_labels, freq_cnt = select_labels_topk_with_min_count(files, label2id_full, MAX_CLASSES,
                                                                                  MIN_SAMPLES_PER_CLASS)
    files = [f for f in files if _file_label_from_path(f) in kept_labels]

    num_classes = len(label2id)
    print(f"Found files={len(files)}  classes={num_classes}  (>= {MIN_SAMPLES_PER_CLASS}/class, top-{MAX_CLASSES})")

    # Build device/AMP
    device, amp_dtype, need_scaler = setup_device_amp()

    # Build model
    model = build_model(num_classes)
    if hasattr(model, "to"):
        model = model.to(device)

    # Optional compile
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[info] model compiled with torch.compile")
        except Exception as e:
            print(f"[warn] torch.compile failed, using eager. reason={e}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AMP scaler (only needed for fp16)
    scaler = torch.amp.GradScaler('cuda') if need_scaler else None

    # Dataloaders
    train_loader = make_loader(files, label2id, shuffle_files=SHUFFLE_FILES, shuffle_within=SHUFFLE_WITHIN_FILE, split="train")
    val_loader = make_loader(files, label2id, shuffle_files=False, shuffle_within=False, split="val")
    test_loader = make_loader(files, label2id, shuffle_files=False, shuffle_within=False, split="test")
    print(f"Split mode: per-sample hashing  (train/val/test = {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f})")

    best_val_acc = -1.0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_mf1 = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, num_classes, ep)
        va_loss, va_acc, va_mf1 = evaluate(model, val_loader, device, amp_dtype, num_classes, ep)
        dt = time.time() - t0
        print(f"[{ep:02d}/{EPOCHS}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} (macroF1={va_mf1:.4f})  time={dt:.1f}s")

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(),
                        "epoch": ep,
                        "val_acc": va_acc,
                        "val_macro_f1": va_mf1,
                        "num_classes": num_classes,
                        "input_len": INPUT_LEN}, SAVE_BEST_PATH)
            print(f"  saved best -> {SAVE_BEST_PATH} (val_acc={va_acc:.4f})")

        # (optional) Free cached GPU memory between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 评测 TEST（用 best 权重）
    if os.path.exists(SAVE_BEST_PATH):
        ckpt = torch.load(SAVE_BEST_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, te_mf1 = evaluate(model, test_loader, device, amp_dtype, num_classes, epoch_idx=0)
    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} macroF1={te_mf1:.4f}")


if __name__ == "__main__":
    main()
