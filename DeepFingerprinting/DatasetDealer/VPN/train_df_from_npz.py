#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import Model_NoDef_pytorch as mdl  # 文件名即模块名

# ---------------- 路径配置（按你的目录结构） ----------------
SCRIPT_DIR   = Path(__file__).resolve().parent  # /home/pcz/.../DatasetDealer/VPN
PROJECT_ROOT = SCRIPT_DIR.parents[1]            # /home/pcz/.../DeepFingerprinting
NPZ_ROOT     = SCRIPT_DIR / "df_npz"            # 由 df_build_npz.py 生成到这里
RUN_DIR      = SCRIPT_DIR / "runs" / "df_npz"

# 训练超参（保持简洁，按需改）
EPOCHS       = 20
BATCH_SIZE   = 128
LR           = 1e-3
NUM_WORKERS  = 4
SEED         = 0
SEQ_LEN      = 5000
SCALE        = 1500.0   # 将有符号包长缩放到 [-1,1]

# ---------------- 复现性 ----------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ---------------- 数据集 ----------------
class NPZDataset(Dataset):
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with np.load(p, allow_pickle=False) as d:
            x = d["x"].astype(np.float32)  # (L,)
            y = int(d["y"])
        # 对齐长度
        if x.shape[0] != SEQ_LEN:
            if x.shape[0] > SEQ_LEN:
                x = x[:SEQ_LEN]
            else:
                pad = np.zeros((SEQ_LEN,), dtype=np.float32)
                pad[:x.shape[0]] = x
                x = pad
        x = x / SCALE
        x = torch.from_numpy(x).unsqueeze(0)  # (1, L) 让 DataLoader 叠成 (B,1,L)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# ---------------- 工具函数 ----------------
def all_npz_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.npz") if p.is_file()])

def group_by_label(paths: List[Path]) -> Dict[int, List[Path]]:
    by: Dict[int, List[Path]] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            y = int(d["y"])
        by.setdefault(y, []).append(p)
    for k in by:
        by[k].sort(key=lambda x: str(x))
    return by

def stratified_split(by: Dict[int, List[Path]], seed:int=0) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    train, val, test = [], [], []
    for y, items in by.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        if n >= 10:
            n_train = int(n * 0.8)
            n_val   = int(n * 0.1)
            n_test  = n - n_train - n_val
        else:
            table = {1:(1,0,0), 2:(1,1,0), 3:(2,1,0), 4:(3,1,0), 5:(3,1,1),
                     6:(4,1,1), 7:(5,1,1), 8:(6,1,1), 9:(7,1,1)}
            n_train, n_val, n_test = table.get(n, (n-2,1,1)) if n>=3 else table[n]
        train += items[:n_train]
        val   += items[n_train:n_train+n_val]
        test  += items[n_train+n_val:n_train+n_val+n_test]
    return train, val, test

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes:int):
    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)      # xb: (B,1,L)
        logits = model(xb)                         # (B,C)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
        idx = yb * num_classes + preds
        cm.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    acc = correct / max(1, total)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = (2*tp + fp + fn).clamp_min(1)
    f1_per_class = (2*tp).float() / denom.float()
    macro_f1 = f1_per_class.mean().item()
    return acc, macro_f1

def build_model(num_classes:int) -> nn.Module:
    # 让 Python 能找到 /home/pcz/.../DeepFingerprinting/Model_NoDef_pytorch.py
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    m = mdl.DFNoDefNet()
    if not hasattr(m, "classifier") or not isinstance(m.classifier, nn.Linear):
        raise RuntimeError("模型中未找到 classifier 线性层")
    in_ch = m.classifier.in_features
    if m.classifier.out_features != num_classes:
        m.classifier = nn.Linear(in_ch, num_classes)
    return m

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # xb: (B,1,L)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        n += yb.size(0)
    return total_loss / max(1, n)

def main():
    assert NPZ_ROOT.exists(), f"找不到 npz 目录: {NPZ_ROOT}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = all_npz_files(NPZ_ROOT)
    assert paths, f"npz 文件为空: {NPZ_ROOT}"
    by = group_by_label(paths)
    num_classes = len(by)
    train_p, val_p, test_p = stratified_split(by, seed=SEED)
    print(f"[data] classes={num_classes} train={len(train_p)} val={len(val_p)} test={len(test_p)}")

    train_ds = NPZDataset(train_p)
    val_ds   = NPZDataset(val_p)
    test_ds  = NPZDataset(test_p) if test_p else None

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True) if test_ds else None

    model = build_model(num_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1.0
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_ld, opt, device)
        val_acc, val_f1 = evaluate(model, val_ld, device, num_classes)
        print(f"[{epoch:02d}/{EPOCHS}] train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {"model": model.state_dict(),
                 "num_classes": num_classes,
                 "seq_len": SEQ_LEN,
                 "scale": SCALE},
                RUN_DIR / "best.pt"
            )

    if test_ld is not None:
        best = torch.load(RUN_DIR / "best.pt", map_location=device)
        model.load_state_dict(best["model"])
        te_acc, te_f1 = evaluate(model, test_ld, device, num_classes)
        print(f"[test] acc={te_acc:.4f} macroF1={te_f1:.4f}")

if __name__ == "__main__":
    main()
