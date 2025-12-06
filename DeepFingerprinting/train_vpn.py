#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFingerprinting VPN 训练脚本

使用 unified_vpn_processor.py 生成的 NPZ 数据训练 DeepFingerprinting
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from Model_NoDef_pytorch import DFNoDefNet


# =============================================================================
# 配置
# =============================================================================

# 数据路径 (当前模型目录下的 vpn_data)
DATA_DIR = Path(__file__).parent / "vpn_unified_output"

# 模型参数
MAX_LEN = 5000  # 固定输入长度

# 训练参数
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "checkpoints"


# =============================================================================
# 数据集
# =============================================================================

class DFDataset(Dataset):
    def __init__(self, flows, labels, max_len):
        self.flows = flows
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        flow = self.flows[idx]
        label = self.labels[idx]

        # Pad/truncate to max_len
        if len(flow) >= self.max_len:
            flow = flow[:self.max_len]
        else:
            flow = np.pad(flow, (0, self.max_len - len(flow)), mode='constant')

        return torch.FloatTensor(flow).unsqueeze(0), label  # (1, max_len)


def load_npz_data(data_dir: Path):
    """从 NPZ 目录加载数据"""
    labels_json = data_dir / "labels.json"
    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    all_flows = []
    all_labels = []

    for label_name, label_id in label2id.items():
        npz_path = data_dir / f"{label_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            flows = data["flows"]
            all_flows.extend(flows)
            all_labels.extend([label_id] * len(flows))

    return all_flows, np.array(all_labels), id2label


def split_data(flows, labels, train_ratio, val_ratio, seed, min_samples=10):
    """分层划分数据集，每个类别按比例划分

    Args:
        flows: 流列表
        labels: 标签数组
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        min_samples: 每个类别最少样本数，不足则剔除该类别
    """
    np.random.seed(seed)

    unique_labels = np.unique(labels)

    train_flows, train_labels = [], []
    val_flows, val_labels = [], []
    test_flows, test_labels = [], []

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

        train_flows.extend([flows[i] for i in train_idx])
        train_labels.extend([label] * len(train_idx))

        val_flows.extend([flows[i] for i in val_idx])
        val_labels.extend([label] * len(val_idx))

        test_flows.extend([flows[i] for i in test_idx])
        test_labels.extend([label] * len(test_idx))

    if removed_classes:
        print(f"\n[Warning] 以下类别样本数不足 {min_samples}，已剔除:")
        for label, count in removed_classes:
            print(f"  - 类别 {label}: {count} 个样本")

    return (
        (train_flows, np.array(train_labels)),
        (val_flows, np.array(val_labels)),
        (test_flows, np.array(test_labels)),
        kept_classes
    )


# =============================================================================
# 训练
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

            with torch.amp.autocast('cuda'):
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
    print("DeepFingerprinting VPN Training")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Max sequence length: {MAX_LEN}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\nLoading data...")
    flows, labels, id2label_orig = load_npz_data(DATA_DIR)

    print(f"Total samples: {len(labels)}")
    print(f"Original classes: {len(id2label_orig)}")

    # 划分数据（分层划分，剔除样本不足的类别）
    (train_flows, train_y), (val_flows, val_y), (test_flows, test_y), kept_classes = split_data(
        flows, labels, TRAIN_RATIO, VAL_RATIO, SEED, min_samples=10
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

    # 创建 DataLoader
    train_loader = DataLoader(
        DFDataset(train_flows, train_y, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        DFDataset(val_flows, val_y, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        DFDataset(test_flows, test_y, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    # 创建模型
    device = torch.device(DEVICE)
    model = DFNoDefNet(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"\nModel: DFNoDefNet")
    print(f"Training for {EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0
    best_model_path = OUTPUT_DIR / "df_vpn_best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, _, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

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
