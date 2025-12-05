#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


DATA_NPZ    = "artifacts/iscx/yatc/data.npz"
BATCH_SIZE  = 256
EPOCHS      = 20
LR          = 1e-3
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED        = 42


def macro_f1_score(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   num_classes: int) -> float:
    """简易 macro-F1 实现，不依赖 sklearn。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    f1_list = []
    for c in range(num_classes):
        tp = np.logical_and(y_true == c, y_pred == c).sum()
        fp = np.logical_and(y_true != c, y_pred == c).sum()
        fn = np.logical_and(y_true == c, y_pred != c).sum()

        if tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1_list.append(f1)

    return float(np.mean(f1_list))


class SimpleCNN(nn.Module):
    """一个非常轻量的 CNN，输入 (B,1,40,40)。"""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 40 -> 20
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 20 -> 10
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # -> (B,128,1,1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
):
    """在给定 loader 上计算 loss / acc / macroF1。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            bs = yb.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += bs

            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    total_loss /= total
    acc = total_correct / total
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    f1m = macro_f1_score(all_labels, all_preds, num_classes)
    return total_loss, acc, f1m


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    # ---- 加载数据 ----
    data = np.load(DATA_NPZ)
    X = data["X"]             # (N,40,40), uint8
    y = data["y"].astype(np.int64)

    N, H, W = X.shape
    assert (H, W) == (40, 40), (H, W)

    # 归一化 + 增加 channel 维度
    X = X.astype("float32") / 255.0      # (N,40,40)
    X = X[:, None, :, :]                 # (N,1,40,40)

    num_classes = int(y.max()) + 1
    print(f"[info] samples={N}, num_classes={num_classes}")

    # ---- 随机 8:1:1 划分 ----
    rng = np.random.default_rng(SEED)
    indices = np.arange(N)
    rng.shuffle(indices)

    n_test = int(round(N * TEST_RATIO))
    n_val  = int(round(N * VAL_RATIO))

    te_idx = indices[:n_test]
    va_idx = indices[n_test:n_test + n_val]
    tr_idx = indices[n_test + n_val:]

    print(f"[split] train={tr_idx.size}, val={va_idx.size}, test={te_idx.size}")

    def make_loader(idxs, shuffle: bool) -> DataLoader:
        x = torch.from_numpy(X[idxs])
        t = torch.from_numpy(y[idxs])
        ds = TensorDataset(x, t)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    train_loader = make_loader(tr_idx, shuffle=True)
    val_loader   = make_loader(va_idx, shuffle=False)
    test_loader  = make_loader(te_idx, shuffle=False)

    # ---- 模型 / 优化器 ----
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- 训练循环 ----
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            tr_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            tr_correct += (preds == yb).sum().item()
            tr_total += bs

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        va_loss, va_acc, va_f1m = evaluate(
            model, val_loader, device, num_classes, criterion
        )

        print(
            f"[{ep:02d}/{EPOCHS}] "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f} (macroF1={va_f1m:.4f})"
        )

    # ---- 训练结束后，在 test 集上评估一次 ----
    te_loss, te_acc, te_f1m = evaluate(
        model, test_loader, device, num_classes, criterion
    )
    print(
        f"[test] loss={te_loss:.4f} acc={te_acc:.4f} macroF1={te_f1m:.4f}"
    )


if __name__ == "__main__":
    main()

# [test] loss=1.2033 acc=0.4551 macroF1=0.4486
# [test] loss=1.1985 acc=0.4487 macroF1=0.4489