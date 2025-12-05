#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_df_from_npzs.py

使用 artifacts/<DATASET_KEY>/dirseq/{data.npz, labels.json} 训练 DFNoDefNet（PyTorch）
- 数据层不做 padding；训练时按 MAX_LEN 右侧零填充/截断
- 自动调整分类头到数据集类数
- 支持“按文件分组的分层切分”（若 data.npz 中存在 gid）
- 保存 best.pt 与 metrics.json
"""

from __future__ import annotations
import os, json, random, math
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.amp import GradScaler, autocast

# ========= 可调参数（默认即可跑） =========
DATASET_KEY   = "iscx"
ART_DIR       = os.path.join("artifacts", DATASET_KEY, "dirseq")
RUN_DIR       = os.path.join("runs", "df_nodf", DATASET_KEY)

SEED          = 2025
EPOCHS        = 30
BATCH_SIZE    = 1024
MAX_LEN       = 5000   # 与 DF 论文常用一致
LR            = 3e-4   # 更稳：配合 AdamW + OneCycleLR
VAL_RATIO     = 0.10   # 验证集更小，让训练样本更多
TEST_RATIO    = 0.10
NUM_WORKERS   = 12

# 失衡对策：建议二选一，默认只开采样（更稳）
USE_CLASS_WEIGHT     = False  # 关掉交叉熵类权重（避免与采样叠加过强）
USE_WEIGHTED_SAMPLER = True   # 保留按 1/sqrt(freq) 的加权采样
ENABLE_AMP           = True   # GPU 上开启 autocast + GradScaler
# =======================================

# ==== 导入模型 ====
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402

# ---------- 工具 ----------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DirSeqDS(Dataset):
    """把变长 ±1 序列在 __getitem__ 时做 pad/trunc 到 MAX_LEN"""
    def __init__(self, X: List[np.ndarray], y: np.ndarray, max_len: int):
        self.X, self.y, self.max_len = X, y, max_len
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        s = self.X[i]
        if len(s) >= self.max_len:
            a = s[:self.max_len]
        else:
            a = np.zeros(self.max_len, dtype=np.int8)
            a[:len(s)] = s
        # 模型期望浮点
        return torch.from_numpy(a.astype(np.float32)), int(self.y[i])

def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> Dict[str, float]:
    # 混淆矩阵
    cm = np.zeros((ncls, ncls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    true_pos = cm.sum(axis=1).astype(np.float64)

    # per-class precision/recall/f1
    prec = np.divide(tp, np.maximum(1.0, pred_pos), dtype=np.float64)
    rec  = np.divide(tp, np.maximum(1.0, true_pos), dtype=np.float64)
    f1_i = np.zeros(ncls, dtype=np.float64)
    denom = prec + rec
    nz = denom > 0
    f1_i[nz] = 2 * prec[nz] * rec[nz] / denom[nz]

    # macro：仅对有样本的类取均值
    mask = true_pos > 0
    macro = float(f1_i[mask].mean()) if np.any(mask) else 0.0

    # weighted：按各类支持度加权
    weighted = float((f1_i * true_pos).sum() / max(1.0, true_pos.sum()))

    # micro：全局聚合（单标签多分类时与 accuracy 数值相同）
    tp_s = float(tp.sum())
    fp_s = float(pred_pos.sum() - tp_s)
    fn_s = float(true_pos.sum() - tp_s)
    p_micro = tp_s / max(1.0, tp_s + fp_s)
    r_micro = tp_s / max(1.0, tp_s + fn_s)
    micro = 0.0 if (p_micro + r_micro) == 0 else 2 * p_micro * r_micro / (p_micro + r_micro)

    return {"macro": macro, "micro": micro, "weighted": weighted}

def collate_pad_to_maxlen(batch, max_len: int):
    """
    兼容 Dataset 返回 ndarray 或 torch.Tensor 的情形。
    按 batch 内最大长度做右侧零填充/截断到 max_len，并堆叠为 (N, max_len) float32。
    """
    import torch
    n = len(batch)
    out = torch.zeros((n, max_len), dtype=torch.float32)
    ys  = torch.empty((n,), dtype=torch.long)

    for i, (arr, y) in enumerate(batch):
        # 统一成 1-D torch.float32 向量
        if isinstance(arr, torch.Tensor):
            t = arr
        else:
            # numpy / list -> tensor
            t = torch.as_tensor(arr)
        if t.dim() != 1:
            t = t.view(-1)
        t = t.to(dtype=torch.float32)

        L = int(t.numel())
        if L >= max_len:
            out[i, :max_len] = t[:max_len]
        else:
            out[i, :L] = t
        ys[i] = int(y)

    return out, ys

def stratified_split(y: np.ndarray, val_r: float, test_r: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """分层划分 train/val/test；对极小类做合理的最小切分"""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx); n = len(idx)
        n_te = max(1, round(n*test_r)) if n >= 3 else (1 if n >= 2 else 0)
        n_va = max(1, round(n*val_r))  if n >= 3 else (0 if n == 2 else 0)
        if n_te + n_va >= n:  # 防过切
            n_te = 1 if n >= 2 else 0; n_va = 0
        te.extend(idx[:n_te]); va.extend(idx[n_te:n_te+n_va]); tr.extend(idx[n_te+n_va:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def dist_info(y: np.ndarray, idx: List[int]) -> Dict[int, int]:
    u, c = np.unique(y[idx], return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}

def make_class_weight(train_y: np.ndarray, ncls: int, device: torch.device) -> torch.Tensor:
    """1/sqrt(freq) 归一到均值=1 的 class weight"""
    u, c = np.unique(train_y, return_counts=True)
    counts = np.zeros(ncls, dtype=np.int64); counts[u] = c
    w = counts.astype(np.float32); w[w == 0] = 1.0
    w = 1.0 / np.sqrt(w)
    w *= (ncls / w.sum())  # 归一
    return torch.tensor(w, dtype=torch.float32, device=device)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> List[List[int]]:
    m = np.zeros((ncls, ncls), dtype=np.int64)
    for t, p in zip(y_true, y_pred): m[int(t), int(p)] += 1
    return m.tolist()

def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, ncls: int, id2label: Dict[int, str]):
    out = {}
    for c in range(ncls):
        t = (y_true == c); p = (y_pred == c)
        tp = int(np.sum(t & p)); fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p)); tn = int(np.sum(~t & ~p))
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        out[id2label.get(c, str(c))] = dict(precision=prec, recall=rec, f1=f1, acc=acc,
                                            tp=tp, fp=fp, fn=fn, tn=tn)
    return out

# ---------- 主流程 ----------
def main():
    set_seed(SEED)
    # torch.backends.cudnn.benchmark = True
    os.makedirs(RUN_DIR, exist_ok=True)

    # 读数据
    data_npz = os.path.join(ART_DIR, "data.npz")
    labels_js = os.path.join(ART_DIR, "labels.json")
    if not (os.path.exists(data_npz) and os.path.exists(labels_js)):
        raise FileNotFoundError(f"缺少数据集文件：{os.path.abspath(data_npz)} 或 {os.path.abspath(labels_js)}")

    obj = np.load(data_npz, allow_pickle=True)
    X = obj["X"]
    X = [np.asarray(x, dtype=np.int8) for x in (list(X) if X.dtype == object else X)]
    y = obj["y"].astype(np.int64)

    id2label_raw = json.load(open(labels_js, "r", encoding="utf-8"))["id2label"]
    id2label = {int(k): v for k, v in id2label_raw.items()}
    ncls = int(np.max(y)) + 1

    print(f"samples={len(X)}  classes={ncls}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 划分 =========
    # 优先尝试“分层 + 分组（按文件）”切分：需要 obj["gid"]（每个样本的文件ID）
    use_group_split = False
    tr_idx = va_idx = te_idx = None
    try:
        if "gid" in obj:
            from sklearn.model_selection import StratifiedGroupKFold
            g = obj["gid"].astype(np.int64)
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            folds = list(sgkf.split(np.zeros_like(y), y, groups=g))
            tr_all, te_idx = folds[0]  # 取第1折为测试集（≈20%）

            # 再从训练候选中按“分层”拿出一小部分做验证
            rng = np.random.default_rng(SEED)
            idx = np.asarray(tr_all)
            va, tr = [], []
            for c in np.unique(y[idx]):
                j = idx[y[idx] == c]
                rng.shuffle(j)
                n_va = max(1, int(round(len(j)*VAL_RATIO))) if len(j) >= 10 else (1 if len(j)>=3 else 0)
                va.extend(j[:n_va]); tr.extend(j[n_va:])
            rng.shuffle(tr); rng.shuffle(va)
            tr_idx, va_idx = tr, va
            use_group_split = True
    except Exception as e:
        print(f"[warn] StratifiedGroupKFold failed: {e}")

    # 回退：纯分层切分
    if not use_group_split:
        tr_idx, va_idx, te_idx = stratified_split(y, VAL_RATIO, TEST_RATIO, SEED)

    print(f"split -> train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    print("train dist:", dist_info(y, tr_idx))
    print("val   dist:", dist_info(y, va_idx))
    print("test  dist:", dist_info(y, te_idx))

    base = DirSeqDS(X, y, MAX_LEN)
    collate = lambda b: collate_pad_to_maxlen(b, MAX_LEN)

    if USE_WEIGHTED_SAMPLER:
        # 采样权重来自训练集分布：1/sqrt(freq)
        u, c = np.unique(y[tr_idx], return_counts=True)
        counts = np.zeros(ncls, dtype=np.int64); counts[u] = c
        w = counts.astype(np.float32); w[w == 0] = 1.0
        w = 1.0 / np.sqrt(w)
        sample_w = w[y[tr_idx]]

        train_loader = DataLoader(
            Subset(base, tr_idx),
            batch_size=BATCH_SIZE,
            sampler=WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.float32),
                num_samples=len(tr_idx),
                replacement=True
            ),
            shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4,
            collate_fn=collate
        )
    else:
        train_loader = DataLoader(
            Subset(base, tr_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4,
            collate_fn=collate
        )

    val_loader = DataLoader(
        Subset(base, va_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=collate
    )
    test_loader = DataLoader(
        Subset(base, te_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=collate
    )

    # ========= 模型 =========
    model = DFNoDefNet()
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 .classifier 线性层")
    if model.classifier.out_features != ncls:
        model.classifier = nn.Linear(model.classifier.in_features, ncls)
    model = model.to(device)

    # ========= 损失与优化 =========
    class_weight = (make_class_weight(y[tr_idx], ncls, device) if USE_CLASS_WEIGHT else None)
    if class_weight is not None:
        print("class_weight:", class_weight.detach().cpu().numpy().round(6).tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

    # AdamW + OneCycleLR + 梯度裁剪（稳定收敛）
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(tr_idx) / max(1, BATCH_SIZE)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch
    )

    scaler = GradScaler('cuda', enabled=ENABLE_AMP and device.type == "cuda")

    def run_epoch(loader: DataLoader, train_mode: bool):
        model.train(train_mode)
        n_tot, loss_sum = 0, 0.0
        y_true, y_pred = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train_mode and scaler.is_enabled():
                with autocast('cuda'):
                    logits = model(xb); loss = criterion(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step()
            else:
                logits = model(xb); loss = criterion(logits, yb)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    scheduler.step()

            n = xb.size(0); loss_sum += loss.item() * n; n_tot += n
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        acc = float((y_true == y_pred).mean()) if n_tot > 0 else 0.0
        return loss_sum / max(1, n_tot), acc, y_true, y_pred

    # ========= 训练循环 =========
    best_val = -1.0
    for ep in range(1, EPOCHS + 1):
        tr_l, tr_a, _, _ = run_epoch(train_loader, True)
        va_l, va_a, _, _ = run_epoch(val_loader, False)
        print(f"[{ep:02d}/{EPOCHS}] train {tr_l:.4f}/{tr_a:.4f}  |  val {va_l:.4f}/{va_a:.4f}")
        if va_a > best_val:
            best_val = va_a
            torch.save({
                "model": model.state_dict(),
                "meta": {"ncls": ncls, "MAX_LEN": MAX_LEN}
            }, os.path.join(RUN_DIR, "best.pt"))
            print(f"  saved best -> {os.path.join(RUN_DIR, 'best.pt')} (val_acc={best_val:.4f})")

    # ========= 测试与报告 =========
    ckpt = torch.load(os.path.join(RUN_DIR, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    te_l, te_a, yt, yp = run_epoch(test_loader, False)
    print(f"[TEST] loss={te_l:.4f}  acc={te_a:.4f}")
    f1 = f1_scores(yt, yp, ncls)
    print(f"[TEST] loss={te_l:.4f}  acc={te_a:.4f}  "
          f"f1(macro/micro/weighted)={f1['macro']:.4f}/{f1['micro']:.4f}/{f1['weighted']:.4f}")

    with open(os.path.join(RUN_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val,
            "test_loss": float(te_l),
            "test_acc": float(te_a),
            "f1": f1,
            "per_class": per_class_report(yt, yp, ncls, id2label),
            "confusion_matrix": confusion_matrix(yt, yp, ncls),
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_len": MAX_LEN, "lr": LR,
            "use_class_weight": USE_CLASS_WEIGHT, "use_weighted_sampler": USE_WEIGHTED_SAMPLER
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    start = datetime.now()
    print("[start]", start.strftime("%Y-%m-%d %H:%M:%S"))
    main()
    end = datetime.now()
    print("[end  ]", end.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[done] total time: {end - start}")
