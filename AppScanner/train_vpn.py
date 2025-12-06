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
EPOCHS = 10
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


def split_data(features, labels, train_ratio, val_ratio, seed, min_samples=10):
    """分层划分数据集，每个类别按比例划分

    Args:
        features: 特征数组
        labels: 标签数组
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        min_samples: 每个类别最少样本数，不足则剔除该类别
    """
    np.random.seed(seed)

    unique_labels = np.unique(labels)

    train_features, train_labels = [], []
    val_features, val_labels = [], []
    test_features, test_labels = [], []

    kept_classes = []
    removed_classes = []

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]

        if len(class_indices) < min_samples:
            removed_classes.append((label, len(class_indices)))
            continue

        kept_classes.append(label)
        np.random.shuffle(class_indices)

        n_train = int(len(class_indices) * train_ratio)
        n_val = int(len(class_indices) * val_ratio)

        train_idx = class_indices[:n_train]
        val_idx = class_indices[n_train:n_train + n_val]
        test_idx = class_indices[n_train + n_val:]

        train_features.append(features[train_idx])
        train_labels.extend([label] * len(train_idx))

        val_features.append(features[val_idx])
        val_labels.extend([label] * len(val_idx))

        test_features.append(features[test_idx])
        test_labels.extend([label] * len(test_idx))

    if removed_classes:
        print(f"\n[Warning] 以下类别样本数不足 {min_samples}，已剔除:")
        for label, count in removed_classes:
            print(f"  - 类别 {label}: {count} 个样本")

    return (
        (np.vstack(train_features), np.array(train_labels)),
        (np.vstack(val_features), np.array(val_labels)),
        (np.vstack(test_features), np.array(test_labels)),
        kept_classes
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
    features, labels, id2label_orig = load_npz_data(DATA_DIR)

    print(f"Total samples: {len(labels)}")
    print(f"Original classes: {len(id2label_orig)}")
    print(f"Feature dim: {features.shape[1]}")

    # 划分数据（分层划分，剔除样本不足的类别）
    (train_X, train_y), (val_X, val_y), (test_X, test_y), kept_classes = split_data(
        features, labels, TRAIN_RATIO, VAL_RATIO, SEED, min_samples=10
    )

    # 重新映射标签到连续的 0, 1, 2, ...
    old_to_new = {old_label: new_label for new_label, old_label in enumerate(kept_classes)}
    train_y = np.array([old_to_new[y] for y in train_y])
    val_y = np.array([old_to_new[y] for y in val_y])
    test_y = np.array([old_to_new[y] for y in test_y])

    # 更新 id2label
    id2label = {new_label: id2label_orig[old_label] for old_label, new_label in old_to_new.items()}
    num_classes = len(kept_classes)

    print(f"Kept classes: {num_classes}")
    print(f"Split (stratified): train={len(train_y)}, val={len(val_y)}, test={len(test_y)}")

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
