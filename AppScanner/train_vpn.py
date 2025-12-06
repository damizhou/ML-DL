#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AppScanner VPN 训练脚本

使用 unified_vpn_processor.py 生成的 NPZ 数据训练 AppScanner
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from models import AppScannerNN


# =============================================================================
# 配置
# =============================================================================

# 数据路径 (当前模型目录下的 vpn_data)
DATA_DIR = Path(__file__).parent / "vpn_unified_output"

# 模型参数
INPUT_DIM = 54  # 54 维统计特征
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.3

# 训练参数
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "checkpoints"


# =============================================================================
# 数据加载
# =============================================================================

def load_npz_data(data_dir: Path):
    """从 NPZ 目录加载数据"""
    labels_json = data_dir / "labels.json"
    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    all_features = []
    all_labels = []

    for label_name, label_id in label2id.items():
        npz_path = data_dir / f"{label_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            features = data["features"]  # (N, 54)
            all_features.append(features)
            all_labels.extend([label_id] * len(features))

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    return all_features, all_labels, id2label


def normalize_features(X_train, X_val, X_test):
    """标准化特征"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


def split_data(features, labels, train_ratio, val_ratio, seed):
    """划分数据集"""
    np.random.seed(seed)
    indices = np.random.permutation(len(labels))

    n_train = int(len(labels) * train_ratio)
    n_val = int(len(labels) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        (features[train_idx], labels[train_idx]),
        (features[val_idx], labels[val_idx]),
        (features[test_idx], labels[test_idx]),
    )


# =============================================================================
# 训练
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total
    return acc, np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 60)
    print("AppScanner VPN Training")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Device: {DEVICE}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\nLoading data...")
    features, labels, id2label = load_npz_data(DATA_DIR)
    num_classes = len(id2label)

    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {num_classes}")
    print(f"Feature dim: {features.shape[1]}")

    # 划分数据
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = split_data(
        features, labels, TRAIN_RATIO, VAL_RATIO, SEED
    )

    print(f"\nSplit: train={len(train_y)}, val={len(val_y)}, test={len(test_y)}")

    # 标准化
    train_X, val_X, test_X = normalize_features(train_X, val_X, test_X)

    # 创建 DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_y)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 创建模型
    device = torch.device(DEVICE)
    model = AppScannerNN(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: AppScannerNN (hidden={HIDDEN_DIMS})")
    print(f"Training for {EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0
    best_model_path = OUTPUT_DIR / "appscanner_vpn_best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 测试
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_acc, preds, labels_arr = evaluate(model, test_loader, device)

    # 计算各项指标
    precision = precision_score(labels_arr, preds, average='macro', zero_division=0)
    recall = recall_score(labels_arr, preds, average='macro', zero_division=0)
    f1 = f1_score(labels_arr, preds, average='macro', zero_division=0)

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")

    # 打印分类报告
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    labels_list = list(range(num_classes))
    target_names = [id2label[i] for i in labels_list]
    print(classification_report(labels_arr, preds, labels=labels_list, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
