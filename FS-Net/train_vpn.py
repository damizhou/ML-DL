#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS-Net VPN 训练脚本

使用 unified_vpn_processor.py 生成的 NPZ 数据训练 FS-Net
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

from models import create_fsnet


# =============================================================================
# 配置
# =============================================================================

# 数据路径 (当前模型目录下的 vpn_data)
DATA_DIR = Path(__file__).parent / "vpn_unified_output"

# 模型参数
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
ALPHA = 1.0  # 重建损失权重

# 训练参数
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
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

class FSNetDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, label


def collate_fn(batch):
    """变长序列 padding，返回 (x, lengths, labels)"""
    sequences, labels = zip(*batch)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    max_len = max(lengths)

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        # 转换: 负值 → 0~1500, 正值 → 1501~3001
        indices = np.where(seq < 0, -seq, seq + 1501)
        padded[i, :len(seq)] = indices

    return torch.LongTensor(padded), torch.LongTensor(lengths), torch.LongTensor(labels)


def load_npz_data(data_dir: Path):
    """从 NPZ 目录加载数据"""
    labels_json = data_dir / "labels.json"
    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    all_sequences = []
    all_labels = []

    for label_name, label_id in label2id.items():
        npz_path = data_dir / f"{label_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            sequences = data["sequences"]
            all_sequences.extend(sequences)
            all_labels.extend([label_id] * len(sequences))

    return all_sequences, np.array(all_labels), id2label


def split_data(sequences, labels, train_ratio, val_ratio, seed):
    """划分数据集"""
    np.random.seed(seed)
    indices = np.random.permutation(len(labels))

    n_train = int(len(labels) * train_ratio)
    n_val = int(len(labels) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        ([sequences[i] for i in train_idx], labels[train_idx]),
        ([sequences[i] for i in val_idx], labels[val_idx]),
        ([sequences[i] for i in test_idx], labels[test_idx]),
    )


# =============================================================================
# 训练
# =============================================================================

def compute_recon_loss(recon_logits, x, lengths):
    """计算重建损失"""
    batch_size, seq_len, vocab_size = recon_logits.size()
    # 创建 mask，只计算非 padding 位置
    mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    recon_logits_flat = recon_logits.view(-1, vocab_size)
    x_flat = x.view(-1)
    mask_flat = mask.view(-1)

    recon_loss = nn.functional.cross_entropy(recon_logits_flat, x_flat, reduction='none')
    recon_loss = (recon_loss * mask_flat.float()).sum() / mask_flat.float().sum()
    return recon_loss


def train_epoch(model, loader, optimizer, device, alpha):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad()
        class_logits, recon_logits = model(x, lengths)

        cls_loss = nn.functional.cross_entropy(class_logits, y)
        recon_loss = compute_recon_loss(recon_logits, x, lengths)
        loss = cls_loss + alpha * recon_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = class_logits.argmax(dim=1)
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
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            class_logits, _ = model(x, lengths)
            preds = class_logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total
    return acc, np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 60)
    print("FS-Net VPN Training")
    print("=" * 60)
    print(f"Data: {DATA_DIR}")
    print(f"Device: {DEVICE}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\nLoading data...")
    sequences, labels, id2label = load_npz_data(DATA_DIR)
    num_classes = len(id2label)

    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {num_classes}")

    # 划分数据
    (train_seq, train_y), (val_seq, val_y), (test_seq, test_y) = split_data(
        sequences, labels, TRAIN_RATIO, VAL_RATIO, SEED
    )

    print(f"\nSplit: train={len(train_y)}, val={len(val_y)}, test={len(test_y)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        FSNetDataset(train_seq, train_y),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        FSNetDataset(val_seq, val_y),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        FSNetDataset(test_seq, test_y),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # 创建模型
    device = torch.device(DEVICE)
    model = create_fsnet(
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel: FSNet (embed={EMBED_DIM}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS})")
    print(f"Training for {EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0
    best_model_path = OUTPUT_DIR / "fsnet_vpn_best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, ALPHA)
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
    target_names = [id2label[i] for i in range(num_classes)]
    print(classification_report(labels_arr, preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
