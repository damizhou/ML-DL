#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_test_df_from_npz.py
读取已处理好的方向序列数据（.npz），使用 DFNoDefNet 训练与测试（PyTorch）。
- 零参数：所有路径与超参都在顶部常量里
- 自动匹配类别数（覆盖 DFNoDefNet 的 classifier 输出维度）
- 兼容不均衡：类权重 + WeightedRandomSampler
- 保存：最优模型、metrics.json、训练日志、测试集预测
"""

from __future__ import annotations
import os, json, random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# 来自你上传的文件：Model_NoDef_pytorch.py
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402

# ============ 配置（零参数开箱即用） ============
OUTDIR           = "outputs_binary"                 # 输出目录
NPZ_PATH         = os.path.join(OUTDIR, "binary.npz")   # 已处理的数据
LABELS_JSON      = os.path.join(OUTDIR, "labels.json")  # 标签映射（可选但推荐）

SEED             = 2025
EPOCHS           = 30
BATCH_SIZE       = 16           # 小样本更稳
MAX_LEN          = 5000         # 与 DF 结构一致（你的 DFNoDefNet 内部也按 5000 计算扁平维）
LR               = 0.002
VAL_RATIO        = 0.05
TEST_RATIO       = 0.20
NUM_WORKERS      = 2
USE_CLASS_WEIGHT = True
USE_WEIGHTED_SAMPLER = True
ENABLE_AMP       = True         # GPU 混合精度
SAVE_BEST_PATH   = os.path.join(OUTDIR, "dfnet_best.pt")
LOG_CSV_PATH     = os.path.join(OUTDIR, "train_log.csv")
METRICS_JSON     = os.path.join(OUTDIR, "metrics.json")
TEST_PRED_CSV    = os.path.join(OUTDIR, "test_predictions.csv")
# ============================================


# --------------- Utils ---------------
def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_trunc(seq: np.ndarray, max_len: int) -> np.ndarray:
    if len(seq) >= max_len:
        return seq[:max_len]
    out = np.zeros(max_len, dtype=np.int8)
    out[:len(seq)] = seq
    return out

def load_npz(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=True)
    X, y = obj["X"], obj["y"].astype(np.int64)
    # 统一为 Python 列表存储的 np.int8 序列
    if isinstance(X, np.ndarray) and X.dtype != object:
        X = [np.asarray(x, dtype=np.int8) for x in X]
    else:
        X = [np.asarray(x, dtype=np.int8) for x in list(X)]
    return X, y

def load_labels(labels_json: str) -> Dict[int, str]:
    if not os.path.exists(labels_json):
        # 无映射文件时，用 id 字符串占位
        return {}
    m = json.load(open(labels_json, "r", encoding="utf-8"))
    return {int(k): v for k, v in m.get("id2label", {}).items()}

# --------------- Dataset ---------------
class DirSeqDataset(Dataset):
    def __init__(self, X: List[np.ndarray], y: Optional[np.ndarray], max_len: int):
        self.X = X; self.y = y; self.max_len = max_len
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        # 输入为 float32，形状 (L,)；模型前向里会自动 unsqueeze 到 (1,L)
        seq = pad_trunc(self.X[idx], self.max_len).astype(np.float32, copy=False)
        x = torch.from_numpy(seq)
        y = -1 if self.y is None else int(self.y[idx])
        return x, y

# --------------- Split ---------------
def stratified_indices(y: np.ndarray, vr: float, tr: float, seed: int):
    rng = np.random.default_rng(seed)
    cls = np.unique(y)
    tr_idx, va_idx, te_idx = [], [], []
    for c in cls:
        idx = np.where(y == c)[0]; rng.shuffle(idx); n = len(idx)
        # 最小保留：当类很少时，至少给 1 个测试，验证可以为 0
        n_te = max(1, int(round(n*tr))) if n >= 3 else (1 if n >= 2 else 0)
        n_va = max(1, int(round(n*vr))) if n >= 3 else (0 if n == 2 else 0)
        if n_te + n_va >= n:
            n_te = 1 if n >= 2 else 0; n_va = 0
        te_idx.extend(idx[:n_te]); va_idx.extend(idx[n_te:n_te+n_va]); tr_idx.extend(idx[n_te+n_va:])
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)
    return tr_idx, va_idx, te_idx

def make_loaders(X: List[np.ndarray], y: np.ndarray, max_len: int):
    base = DirSeqDataset(X, y, max_len)
    tr_idx, va_idx, te_idx = stratified_indices(y, VAL_RATIO, TEST_RATIO, SEED)

    def _dist(idxs):
        vv, cc = np.unique(y[idxs], return_counts=True)
        return dict(zip(map(int, vv), map(int, cc)))

    print(f"split -> train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    print("train dist:", _dist(tr_idx))
    print("val   dist:", _dist(va_idx))
    print("test  dist:", _dist(te_idx))

    # 训练集采样方式
    if USE_WEIGHTED_SAMPLER:
        uniq, cts = np.unique(y[tr_idx], return_counts=True)
        counts = np.zeros(int(np.max(y))+1, dtype=np.int64); counts[uniq] = cts
        w = counts.astype(np.float32); w[w==0] = 1.0; w = 1.0 / np.sqrt(w)
        sample_w = w[y[tr_idx]]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                        num_samples=len(tr_idx), replacement=True)
        train_loader = DataLoader(Subset(base, tr_idx), batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(Subset(base, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)

    val_loader  = DataLoader(Subset(base, va_idx),  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(Subset(base, te_idx),  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, tr_idx

# --------------- Metrics ---------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m

def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, Dict[str, float]]:
    rep = {}
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-12, prec + rec)
        acc  = (tp + tn) / max(1, tp + fp + fn + tn)
        rep[c] = {"precision":prec, "recall":rec, "f1":f1, "acc":acc, "tp":tp, "fp":fp, "fn":fn, "tn":tn}
    return rep

# --------------- Train / Eval ---------------
def run_epoch(model, loader, device, criterion, optimizer=None, scaler: Optional[GradScaler]=None):
    train = optimizer is not None
    model.train(train)
    n_tot, loss_sum = 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in tqdm(loader, disable=not train, ncols=100):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if train and scaler is not None:
                with autocast('cuda'):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        n = xb.size(0)
        loss_sum += loss.item() * n
        n_tot += n
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = float((y_true == y_pred).mean())
    return loss_sum / max(1, n_tot), acc, y_true, y_pred

# --------------- Main ---------------
def main():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) 载入数据
    X, y = load_npz(NPZ_PATH)
    num_classes = int(np.max(y)) + 1
    id2label = load_labels(LABELS_JSON)
    print(f"samples={len(X)}  classes={num_classes}")

    # 2) 加载器
    train_loader, val_loader, test_loader, tr_idx = make_loaders(X, y, MAX_LEN)

    # 3) 类权重（基于训练集）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_CLASS_WEIGHT:
        uniq, cts = np.unique(y[tr_idx], return_counts=True)
        counts = np.zeros(num_classes, dtype=np.int64); counts[uniq] = cts
        w = counts.astype(np.float32); w[w==0] = 1.0; w = 1.0 / np.sqrt(w)
        w = w * (num_classes / w.sum())  # 归一到均值≈1
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("class_weight =", w.tolist())
    else:
        class_weight = None

    # 4) 构建模型（覆盖输出维度）
    model = DFNoDefNet()  # 你的定义里 classifier 默认是 95 个类
    if getattr(model, "classifier", None) is None:
        raise AttributeError("DFNoDefNet 未找到 classifier 层，请检查 Model_NoDef_pytorch.py")
    if model.classifier.out_features != num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
    scaler = GradScaler('cuda', enabled=ENABLE_AMP and device.type == 'cuda')

    # 5) 训练
    best_val, best_ckpt = -1.0, None
    with open(LOG_CSV_PATH, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader,   device, criterion, optimizer=None, scaler=None)
        print(f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        with open(LOG_CSV_PATH, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{va_loss:.6f},{va_acc:.6f}\n")

        if va_acc > best_val:
            best_val = va_acc
            best_ckpt = {
                "model": model.state_dict(),
                "cfg": {"MAX_LEN": MAX_LEN, "num_classes": num_classes},
            }
            torch.save(best_ckpt, SAVE_BEST_PATH)
            print(f"[saved best] -> {SAVE_BEST_PATH} (val_acc={best_val:.4f})")

    # 6) 测试
    if best_ckpt is None:
        best_ckpt = torch.load(SAVE_BEST_PATH, map_location=device)
        model.load_state_dict(best_ckpt["model"])
    else:
        model.load_state_dict(best_ckpt["model"])

    te_loss, te_acc, y_true, y_pred = run_epoch(model, test_loader, device, criterion, optimizer=None, scaler=None)
    print("\n────────────── Closed-World Test ──────────────")
    print(f"test loss={te_loss:.4f}  acc={te_acc:.4f}")

    # 报告与混淆矩阵
    rep = classification_report(y_true, y_pred, num_classes)
    cm  = confusion_matrix(y_true, y_pred, num_classes).tolist()

    # 7) 写出指标与预测
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": float(best_val),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "per_class": {
                (id2label.get(k, str(k))): {kk: float(vv) for kk, vv in rep[k].items()}
                for k in range(num_classes)
            },
            "confusion_matrix": cm,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_len": MAX_LEN, "lr": LR,
            "use_class_weight": bool(USE_CLASS_WEIGHT),
            "use_weighted_sampler": bool(USE_WEIGHTED_SAMPLER)
        }, f, ensure_ascii=False, indent=2)

    # 预测明细（便于排查）
    with open(TEST_PRED_CSV, "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            f.write(f"{t},{p}\n")

    print(f"\n[done] best={best_val:.4f}  test_acc={te_acc:.4f}")
    print(f"outputs -> {SAVE_BEST_PATH}, {METRICS_JSON}, {LOG_CSV_PATH}, {TEST_PRED_CSV}")

if __name__ == "__main__":
    main()
