#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_df_simple.py — 极简可跑 + 逐行中文注释 + macro F1
====================================================
目标：给初学者的最小化训练脚本，尽量使用 PyTorch 默认配置，逻辑清晰。
- 不使用：加权采样、类权重、AMP、学习率调度、自定义 collate、复杂数据划分
- 仅做：随机切分 train/val/test（80/10/10），训练若干 epoch，最后在 test 上输出 loss / acc / macro F1
- Dataset 内在 __getitem__ 做右侧零填充/截断到 MAX_LEN，使 DataLoader 能用默认 collate
- 模型使用你现有的 DFNoDefNet；若分类头输出维度与数据集类别数不一致，会自动替换为正确的线性层

注意：
1) 需要在目录 DatasetDealer/ISCXVPN/artifacts/<DATASET_KEY>/dirseq/ 下准备 data.npz 和 labels.json
   - data.npz 应至少包含键：X(变长的 ±1/0 序列)、y(类别 id，从 0 开始)
   - labels.json 应包含键：{"id2label": {"0": "chat", "1": "email", ...}}
2) 本脚本聚焦“最小可跑”；后续想加速或提稳（AMP/分层切分/加权等），可以在此基础上逐项开启。
"""

from __future__ import annotations
import os
import json
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet

# =====================
# 基本常量（保持很少很直观）
# =====================
DATASET_KEY = "iscx"                     # 数据集键名：决定读取的目录
ART_DIR     = os.path.join("../ISCXTor/artifacts",  # 根目录（按你的工程习惯）
                           DATASET_KEY,
    "dirseq",
                           )
EPOCHS      = 30                          # 训练轮数（可按需改大/改小）
BATCH_SIZE  = 1024                        # 批大小（视显存调整）
MAX_LEN     = 5000                        # 每条样本的统一长度（右侧零填充或截断）

# 你的 DF 模型；保持与工程一致（假设同目录可 import 到）
# from ... import Model_NoDef_pytorch as DFNoDefNet # noqa: E402


class DirSeqDS(Dataset):
    """把变长 ±1 序列在 __getitem__ 时做右侧零填充/截断到 MAX_LEN，返回 float32 张量。

    好处：
    - DataLoader 可以使用默认 collate（自动把同 shape 的张量堆叠成 batch）
    - 这种“按需 pad”的策略更节省内存（不需要预先把所有样本都 pad 好）
    """
    def __init__(self, X: List[np.ndarray], y: np.ndarray, max_len: int):
        self.X = X                      # Python 列表，每个元素是 1D numpy 数组（变长）
        self.y = y                      # numpy 向量（类别 id，从 0 开始）
        self.max_len = max_len          # 统一的输出长度

    def __len__(self) -> int:
        return len(self.X)              # 样本总数

    def __getitem__(self, i: int):
        s = self.X[i]                   # 取出第 i 个变长序列（numpy 数组，元素通常为 -1/0/1）
        if len(s) >= self.max_len:      # 若超长，则直接截断
            a = s[: self.max_len]
        else:                           # 若不足，则右侧零填充
            a = np.zeros(self.max_len, dtype=np.int8)
            a[: len(s)] = s
        # 返回：特征张量（float32），标签（int）
        return torch.from_numpy(a.astype(np.float32)), int(self.y[i])


def simple_split(n: int, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最简随机切分，不分层/不分组：返回 train/val/test 的索引。

    - 输入 n：样本总数
    - val_ratio/test_ratio：验证/测试占比（默认各 10%）
    - 输出顺序：train_idx, val_idx, test_idx
    """
    idx = np.arange(n)                  # [0, 1, 2, ..., n-1]
    rng = np.random.default_rng()       # 现代随机数生成器
    rng.shuffle(idx)                    # 原地打乱索引（确保随机）

    n_test = int(round(n * test_ratio)) # 测试集样本数
    n_val  = int(round(n * val_ratio))  # 验证集样本数
    n_train = max(0, n - n_test - n_val)

    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val : n_test + n_val + n_train]
    return tr, va, te


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """最简单的分类准确率：预测==真实 的比例。"""
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> float:
    """计算 macro F1（不依赖第三方库）。

    定义：
      - 对每个类别 c 单独计算 F1_c = 2*TP / (2*TP + FP + FN)
      - 对所有类别取算术平均：macro_F1 = (1/ncls) * ∑_c F1_c
    约定：
      - 若对某类 c 而言 (2*TP + FP + FN) == 0（即该类在真值与预测中都没出现），将其 F1_c 记为 0（sklearn 的做法是可以通过参数控制，这里取直观处理）。
    """
    if y_true.size == 0:
        return 0.0
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    f1_list = []
    for c in range(ncls):
        # 逐类统计 TP/FP/FN
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1_c)
    return float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0


def main() -> None:
    # ---------- 设备选择：优先 GPU ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 读取数据：最小化校验 ----------
    data_npz = os.path.join(ART_DIR, "data.npz")
    labels_js = os.path.join(ART_DIR, "labels.json")
    if not os.path.exists(data_npz) or not os.path.exists(labels_js):
        raise FileNotFoundError("请在 ART_DIR 准备 data.npz 与 labels.json —— 见文件头部注释")

    # np.load 载入对象；X 通常是变长序列构成的 object 数组
    obj = np.load(data_npz, allow_pickle=True)
    X = obj["X"]
    # 统一转为 Python 列表，元素为 numpy int8 向量（变长）
    if X.dtype == object:
        X = list(X)
    X = [np.asarray(x, dtype=np.int8) for x in X]

    # y 为类别 id，从 0 开始
    y = obj["y"].astype(np.int64)

    # 读取 id2label（虽然后续不强依赖，但可保留以对照）
    with open(labels_js, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)["id2label"]
    id2label = {int(k): v for k, v in id2label_raw.items()}
    ncls = int(np.max(y)) + 1

    print(f"samples={len(X)}  classes={ncls}")

    # ---------- 最简数据划分：随机 80/10/10 ----------
    tr_idx, va_idx, te_idx = simple_split(len(X), val_ratio=0.1, test_ratio=0.1)

    # ---------- Dataset / DataLoader（默认参数） ----------
    base = DirSeqDS(X, y, MAX_LEN)
    train_loader = DataLoader(Subset(base, tr_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(base, va_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(base, te_idx), batch_size=BATCH_SIZE, shuffle=False)

    # ---------- 模型：必要的分类头自适配 ----------
    model = DFNoDefNet()
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 .classifier 线性层 —— 无法确定输出类别数")
    if model.classifier.out_features != ncls:
        model.classifier = nn.Linear(model.classifier.in_features, ncls)
    model = model.to(device)

    # ---------- 损失/优化器：全部默认 ----------
    criterion = nn.CrossEntropyLoss().to(device)    # 默认：均匀类权重
    optimizer = torch.optim.Adam(model.parameters()) # 默认：lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0

    # ---------- 训练循环 ----------
    for ep in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []

        for xb, yb in train_loader:              # 逐 batch
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)                   # 前向
            loss = criterion(logits, yb)         # 计算交叉熵

            optimizer.zero_grad()                # 梯度清零
            loss.backward()                      # 反向传播
            optimizer.step()                     # 参数更新

            n = xb.size(0)
            run_loss += loss.item() * n
            run_n += n
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        tr_loss = run_loss / max(1, run_n)
        tr_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        tr_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        tr_acc = accuracy(tr_y, tr_p)

        # --- Val ---（评估时不求导）
        model.eval()
        with torch.no_grad():
            run_loss, run_n = 0.0, 0
            y_true_chunks, y_pred_chunks = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                n = xb.size(0)
                run_loss += loss.item() * n
                run_n += n
                y_true_chunks.append(yb.detach().cpu().numpy())
                y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            va_loss = run_loss / max(1, run_n)
            va_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
            va_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
            va_acc = accuracy(va_y, va_p)
            va_f1m = macro_f1(va_y, va_p, ncls)

        print(f"[{ep:02d}/{EPOCHS}] train {tr_loss:.4f}/{tr_acc:.4f}  |  val {va_loss:.4f}/{va_acc:.4f}  (val macroF1={va_f1m:.4f})")

    # ---------- Test ----------
    model.eval()
    with torch.no_grad():
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            n = xb.size(0)
            run_loss += loss.item() * n
            run_n += n
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        te_loss = run_loss / max(1, run_n)
        te_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        te_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        te_acc = accuracy(te_y, te_p)
        te_f1m = macro_f1(te_y, te_p, ncls)

    print(f"[TEST] loss={te_loss:.4f}  acc={te_acc:.4f}  macroF1={te_f1m:.4f}")


if __name__ == "__main__":
    main()
