#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFingerprinting 统一训练脚本

支持多种数据格式:
1. 单 NPZ 文件: data.npz (X, y) + labels.json
2. 多 NPZ 目录: 每个 NPZ 包含 flows/labels
3. 统一目录格式: <label>.npz + labels.json

Usage:
    python run_train.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from Model_NoDef_pytorch import DFNoDefNet


# =============================================================================
# 配置
# =============================================================================

# 数据路径 (支持三种格式，自动检测)
DATA_PATH = Path(__file__).parent / "data"

# 模型参数
MAX_LEN = 5000              # 固定输入长度 (论文默认)

# 训练参数
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
NUM_WORKERS = 0             # Windows 兼容
SEED = 42

# 数据集划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 启用 cuDNN 优化
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "checkpoints"


# =============================================================================
# 数据集类
# =============================================================================

class DFDataset(Dataset):
    """DeepFingerprinting 数据集 (内存加载)"""

    def __init__(self, flows: List[np.ndarray], labels: np.ndarray, max_len: int):
        self.flows = flows
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        flow = self.flows[idx]
        label = int(self.labels[idx])

        # Pad/truncate to max_len
        if len(flow) >= self.max_len:
            flow = flow[:self.max_len]
        else:
            flow = np.pad(flow, (0, self.max_len - len(flow)), mode='constant')

        return torch.from_numpy(flow.astype(np.float32)), label


class LazyDFDataset(Dataset):
    """DeepFingerprinting 数据集 (懒加载，适用于大规模数据)"""

    def __init__(
        self,
        file_paths: List[str],
        file_ids: np.ndarray,
        flow_indices: np.ndarray,
        labels: np.ndarray,
        max_len: int,
        cache_size: int = 1000
    ):
        self.file_paths = file_paths
        self.file_ids = file_ids
        self.flow_indices = flow_indices
        self.labels = labels
        self.max_len = max_len
        self.cache_size = cache_size

        # 使用 LRU 缓存
        self._load_flows = lru_cache(maxsize=cache_size)(self._load_flows_impl)

    def _load_flows_impl(self, path: str) -> np.ndarray:
        with np.load(path, allow_pickle=True) as data:
            return data["flows"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_id = int(self.file_ids[idx])
        flow_idx = int(self.flow_indices[idx])
        label = int(self.labels[idx])

        flows = self._load_flows(self.file_paths[file_id])
        flow = np.asarray(flows[flow_idx], dtype=np.float32)

        # Pad/truncate to max_len
        if len(flow) >= self.max_len:
            flow = flow[:self.max_len]
        else:
            padded = np.zeros(self.max_len, dtype=np.float32)
            padded[:len(flow)] = flow
            flow = padded

        return torch.from_numpy(flow), label


# =============================================================================
# 数据加载
# =============================================================================

def detect_data_format(data_path: Path) -> str:
    """自动检测数据格式"""
    if data_path.is_file() and data_path.suffix == '.npz':
        return 'single_npz'

    if data_path.is_dir():
        # 检查是否有 labels.json
        labels_json = data_path / "labels.json"
        if labels_json.exists():
            # 检查 NPZ 文件格式
            npz_files = list(data_path.glob("*.npz"))
            if npz_files:
                with np.load(npz_files[0], allow_pickle=True) as data:
                    if 'X' in data.files and 'y' in data.files:
                        return 'single_npz'
                    elif 'flows' in data.files:
                        return 'unified_dir'

        # 检查子目录中的 NPZ
        sub_npz = list(data_path.rglob("*.npz"))
        if sub_npz:
            return 'multi_npz'

    raise ValueError(f"无法识别数据格式: {data_path}")


def load_single_npz(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载单个 NPZ 文件 (data.npz 格式)"""
    # 查找 labels.json
    if data_path.is_file():
        labels_json = data_path.parent / "labels.json"
        npz_path = data_path
    else:
        labels_json = data_path / "labels.json"
        npz_path = data_path / "data.npz"

    # 加载标签映射
    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}

    # 加载数据
    with np.load(npz_path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"]

    # 转换为列表
    if X.dtype == object:
        flows = list(X)
    else:
        flows = [X[i] for i in range(len(X))]

    flows = [np.asarray(f, dtype=np.int8) for f in flows]
    labels = y.astype(np.int64)

    return flows, labels, id2label


def load_unified_dir(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载统一目录格式 (<label>.npz + labels.json)"""
    labels_json = data_path / "labels.json"

    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta.get("label2id", {})
    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}

    all_flows = []
    all_labels = []

    for label_name, label_id in label2id.items():
        npz_path = data_path / f"{label_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            flows = data["flows"]
            all_flows.extend(flows)
            all_labels.extend([label_id] * len(flows))

    flows = [np.asarray(f, dtype=np.int8) for f in all_flows]
    labels = np.array(all_labels, dtype=np.int64)

    return flows, labels, id2label


def load_multi_npz(data_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """加载多 NPZ 目录 (懒加载索引)"""
    # 查找 labels.json
    labels_json = data_path / "labels.json"
    if not labels_json.exists():
        # 尝试在子目录中查找
        for p in data_path.rglob("labels.json"):
            labels_json = p
            break

    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = {str(k): int(v) for k, v in meta.get("label2id", {}).items()}
    id2label = {int(k): str(v) for k, v in meta.get("id2label", {}).items()}

    # 扫描所有 NPZ 文件
    file_paths = []
    file_ids = []
    flow_indices = []
    labels = []

    npz_files = sorted(data_path.rglob("*.npz"))
    print(f"扫描 NPZ 文件: {len(npz_files)} 个")

    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "flows" not in data.files:
                    continue

                flows = data["flows"]
                n = len(flows)
                if n == 0:
                    continue

                # 获取标签
                if "labels" in data.files:
                    file_labels = np.asarray(data["labels"]).reshape(-1)
                    if file_labels.size == 1:
                        file_labels = np.full(n, file_labels[0])
                else:
                    # 从文件名推断标签
                    label_name = npz_path.stem
                    if label_name not in label2id:
                        continue
                    file_labels = np.full(n, label2id[label_name])

                fid = len(file_paths)
                file_paths.append(str(npz_path))

                for i in range(n):
                    file_ids.append(fid)
                    flow_indices.append(i)

                    label = file_labels[i] if i < len(file_labels) else file_labels[0]
                    if isinstance(label, (bytes, np.bytes_)):
                        label = label.decode('utf-8')
                    if isinstance(label, str):
                        label = label2id.get(label, 0)
                    labels.append(int(label))

        except Exception as e:
            print(f"跳过文件 {npz_path}: {e}")
            continue

    return (
        file_paths,
        np.array(file_ids, dtype=np.int32),
        np.array(flow_indices, dtype=np.int32),
        np.array(labels, dtype=np.int64),
        id2label
    )


def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """分层划分数据集"""
    np.random.seed(seed)

    unique_labels = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
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

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:n_train + n_val])
        test_idx.extend(class_indices[n_train + n_val:])

    if removed_classes:
        print(f"\n[Warning] 以下类别样本数不足 {min_samples}，已剔除:")
        for label, count in removed_classes:
            print(f"  - 类别 {label}: {count} 个样本")

    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
        kept_classes
    )


# =============================================================================
# 训练与评估
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Dict[str, float]:
    """单轮训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == 'cuda'

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """评估模型"""
    model.eval()

    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 计算 TPR/FPR
    tpr_list, fpr_list = [], []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        tn = ((all_preds != c) & (all_labels != c)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr_avg": np.mean(tpr_list),
        "fpr_avg": np.mean(fpr_list),
        "predictions": all_preds,
        "labels": all_labels
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("DeepFingerprinting Training")
    print("=" * 70)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 检测数据格式
    print(f"\nData path: {DATA_PATH}")
    data_format = detect_data_format(DATA_PATH)
    print(f"Data format: {data_format}")

    # 加载数据
    print("\nLoading data...")
    use_lazy = False

    if data_format == 'single_npz':
        flows, labels, id2label = load_single_npz(DATA_PATH)
    elif data_format == 'unified_dir':
        flows, labels, id2label = load_unified_dir(DATA_PATH)
    elif data_format == 'multi_npz':
        file_paths, file_ids, flow_indices, labels, id2label = load_multi_npz(DATA_PATH)
        use_lazy = True
    else:
        raise ValueError(f"不支持的数据格式: {data_format}")

    print(f"Total samples: {len(labels)}")
    print(f"Original classes: {len(id2label)}")

    # 分层划分数据集
    train_idx, val_idx, test_idx, kept_classes = stratified_split(
        labels, TRAIN_RATIO, VAL_RATIO, SEED, min_samples=10
    )

    # 重映射标签
    old_to_new = {old: new for new, old in enumerate(kept_classes)}
    num_classes = len(kept_classes)
    new_id2label = {new: id2label[old] for old, new in old_to_new.items()}

    print(f"Kept classes: {num_classes}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 打印类别分布
    print("\nClass distribution:")
    for old_label in kept_classes:
        new_label = old_to_new[old_label]
        count = (labels[train_idx] == old_label).sum()
        name = id2label.get(old_label, f"Class_{old_label}")
        print(f"  [{new_label:2d}] {name:20s}: {count:6d}")

    # 创建数据集
    if use_lazy:
        # 懒加载模式
        base_dataset = LazyDFDataset(
            file_paths, file_ids, flow_indices, labels, MAX_LEN
        )

        # 重映射标签
        remapped_labels = np.array([old_to_new.get(l, 0) for l in labels])
        base_dataset.labels = remapped_labels

        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)
        test_dataset = Subset(base_dataset, test_idx)
    else:
        # 内存加载模式
        train_flows = [flows[i] for i in train_idx]
        val_flows = [flows[i] for i in val_idx]
        test_flows = [flows[i] for i in test_idx]

        train_labels = np.array([old_to_new[labels[i]] for i in train_idx])
        val_labels = np.array([old_to_new[labels[i]] for i in val_idx])
        test_labels = np.array([old_to_new[labels[i]] for i in test_idx])

        train_dataset = DFDataset(train_flows, train_labels, MAX_LEN)
        val_dataset = DFDataset(val_flows, val_labels, MAX_LEN)
        test_dataset = DFDataset(test_flows, test_labels, MAX_LEN)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 创建模型
    model = DFNoDefNet(num_classes=num_classes).to(device)
    print(f"\nModel: DFNoDefNet")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 打印训练配置
    print("\nTraining Configuration:")
    print(f"  Max length:    {MAX_LEN}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Num classes:   {num_classes}")
    print("=" * 70)

    # 训练循环
    best_f1 = 0.0
    best_model_path = OUTPUT_DIR / "best.pth"

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # 验证
        val_metrics = evaluate(model, val_loader, device, num_classes)

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            saved_marker = " *"
        else:
            saved_marker = ""

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f}{saved_marker}"
        )

    # 最终测试
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, num_classes)

    # 打印结果
    print(f"\nOverall Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    print(f"  FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    # 分类报告
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    target_names = [new_id2label[i] for i in range(num_classes)]
    print(classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0
    ))

    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
