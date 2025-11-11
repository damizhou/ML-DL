#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_df_simple.py — 多NPZ（flows/labels）+ 外部 labels.json 映射 + 懒加载（最小可运行）
环境变量：
  MULTI_NPZ_ROOT=/path/to/npz_root
  LABELS_JSON=/path/to/labels.json      # 可选，默认在 MULTI_NPZ_ROOT/labels.json
  EPOCHS=5  BATCH_SIZE=512  MAX_LEN=5000
  NUM_WORKERS=2  NPZ_CACHE_FILES=2
"""

from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from typing import List, Tuple, Dict
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# —— 基础设置：避免把CPU吃满（可由环境变量覆盖）——
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP", "1")))
torch.backends.cudnn.benchmark = True

# —— 超参（保持很少）——
EPOCHS          = 30
BATCH_SIZE      = 2048
MAX_LEN         = 5000
MULTI_NPZ_ROOT  = '/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz'
NUM_WORKERS     = 8
NPZ_CACHE_FILES = 128
LABELS_JSON     = r'/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/labels.json'

# —— 导入你的 DF 模型（保持工程结构）——
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402


# ============ 工具 ============
def to_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return str(x)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0: return 0.0
    return float((y_true == y_pred).mean())

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> float:
    if y_true.size == 0: return 0.0
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    f1s = []
    for c in range(ncls):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        d = 2*tp + fp + fn
        f1s.append((2.0*tp/d) if d>0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def simple_split(n: int, val_ratio=0.1, test_ratio=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int32)
    np.random.default_rng(42).shuffle(idx)
    n_te = int(round(n*test_ratio)); n_va = int(round(n*val_ratio))
    te = idx[:n_te]; va = idx[n_te:n_te+n_va]; tr = idx[n_te+n_va:]
    return tr, va, te


# ============ 索引构建（仅读 labels，按 labels.json 映射为 id） ============
def load_label_mapping(labels_json_path: Path) -> Tuple[Dict[str,int], Dict[int,str]]:
    with labels_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    l2i = {str(k): int(v) for k, v in obj.get("label2id", {}).items()}
    i2l = {int(k): str(v) for k, v in obj.get("id2label", {}).items()}
    if not l2i or not i2l:
        raise ValueError(f"labels.json 缺少 label2id/id2label：{labels_json_path}")
    return l2i, i2l

def build_index(root_dir: Path, label2id: Dict[str,int]):
    """返回 (file_paths, file_ids, flow_idx, y_ids, ncls)
    全程不把 flows 驻内存，仅生成紧凑索引与整数标签数组。
    """
    files = sorted(root_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found under: {root_dir}")

    file_paths: List[str] = []
    file_ids: List[int] = []
    flow_idx: List[int] = []
    y_ids:    List[int] = []

    for path in tqdm(files, desc="scan npz", unit="file", mininterval=1.0):
        try:
            with np.load(path, allow_pickle=True) as obj:
                if "flows" not in obj.files or "labels" not in obj.files:
                    continue
                n = len(obj["flows"])
                if n <= 0:
                    continue
                labs = np.asarray(obj["labels"]).reshape(-1)
        except Exception:
            continue

        fid = len(file_paths)
        file_paths.append(str(path))

        if labs.size == 1:
            lid = label2id[to_str(labs[0])]
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            y_ids.extend([lid]*n)
        elif labs.size == n:
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            for j in range(n):
                lid = label2id[to_str(labs[j])]
                y_ids.append(lid)
        else:
            # 长度异常：按首元素广播
            lid = label2id[to_str(labs[0])] if labs.size>0 else label2id[to_str("")]
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            y_ids.extend([lid]*n)

    if not y_ids:
        raise RuntimeError("No usable samples after scanning.")

    return (
        file_paths,
        np.asarray(file_ids, dtype=np.int32),
        np.asarray(flow_idx, dtype=np.int32),
        np.asarray(y_ids,   dtype=np.int64),
        max(y_ids)+1
    )


# ============ 懒加载 Dataset（文件级小缓存） ============
class LazyNPZDataset(Dataset):
    def __init__(self, file_paths: List[str], file_ids: np.ndarray, flow_idx: np.ndarray,
                 y_ids: np.ndarray, max_len: int, cache_files: int = 2):
        self.file_paths = file_paths
        self.file_ids   = file_ids.astype(np.int32,  copy=False)
        self.flow_idx   = flow_idx.astype(np.int32,  copy=False)
        self.y          = y_ids.astype(np.int64,     copy=False)
        self.max_len    = max_len
        self.cache_max  = max(1, cache_files)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()  # fid -> flows(obj array)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def _get_flows(self, fid: int):
        if fid in self._cache:
            self._cache.move_to_end(fid); return self._cache[fid]
        with np.load(self.file_paths[fid], allow_pickle=True) as obj:
            flows = obj["flows"]
        self._cache[fid] = flows
        if len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)
        return flows

    def __getitem__(self, i: int):
        fid = int(self.file_ids[i]); j = int(self.flow_idx[i])
        s = np.asarray(self._get_flows(fid)[j], dtype=np.int8)
        if s.size >= self.max_len:
            a = s[: self.max_len]
        else:
            a = np.zeros(self.max_len, dtype=np.int8); a[: s.size] = s
        return torch.from_numpy(a.astype(np.float32)), int(self.y[i])


# ============ 主程序 ============
def main():
    if not MULTI_NPZ_ROOT:
        raise RuntimeError("请先设置 MULTI_NPZ_ROOT 指向包含 .npz 的根目录")
    root_dir = Path(MULTI_NPZ_ROOT).expanduser().resolve()
    labels_json = Path(LABELS_JSON).expanduser().resolve() if LABELS_JSON else (root_dir / "labels.json")
    if not labels_json.exists():
        raise FileNotFoundError(f"未找到 labels.json：{labels_json}（请先运行你生成映射的脚本）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device:", torch.cuda.get_device_name(0))

    label2id, id2label = load_label_mapping(labels_json)
    file_paths, file_ids, flow_idx, y_ids, ncls = build_index(root_dir, label2id)
    print(f"samples={len(y_ids)}  classes={ncls}")

    base = LazyNPZDataset(file_paths, file_ids, flow_idx, y_ids, MAX_LEN, cache_files=NPZ_CACHE_FILES)
    tr_idx, va_idx, te_idx = simple_split(len(base), 0.1, 0.1)

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(Subset(base, tr_idx), shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(Subset(base, va_idx), shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(Subset(base, te_idx), shuffle=False, **dl_kwargs)

    model = DFNoDefNet()
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 .classifier 线性层")
    if model.classifier.out_features != ncls:
        model.classifier = nn.Linear(model.classifier.in_features, ncls)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # 混合精度（GPU 时启用）
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    for ep in range(1, EPOCHS + 1):
        # ------- Train -------
        model.train()
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"train {ep:02d}", mininterval=3.0)
        for step, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = xb.size(0)
            run_loss += loss.item() * n
            run_n += n
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 训练指标
        tr_loss = run_loss / max(1, run_n)
        tr_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        tr_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        tr_acc = accuracy(tr_y, tr_p)

        # ------- Val -------
        model.eval()
        with torch.no_grad():
            run_loss, run_n = 0.0, 0
            y_true_chunks, y_pred_chunks = [], []
            for xb, yb in tqdm(val_loader, total=len(val_loader), desc="val", leave=False, mininterval=1.0):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                n = xb.size(0)
                run_loss += loss.item() * n
                run_n += n
                y_true_chunks.append(yb.detach().cpu().numpy())
                y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        # 验证指标（注意 ncls 要已在前面计算好）
        va_loss = run_loss / max(1, run_n)
        va_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        va_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        va_acc = accuracy(va_y, va_p)
        va_f1m = macro_f1(va_y, va_p, ncls)

        # ------- 日志 -------
        print(f"[{ep:02d}/{EPOCHS}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} (macroF1={va_f1m:.4f})")


if __name__ == "__main__":
    main()
