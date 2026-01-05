"""
DeepFingerprinting Ablation Study Training Script

实现三个消融实验：

实验1 (Baseline: 首页指纹):
    训练集: 数据集B (仅首页) 70%
    验证集: 数据集B (仅首页) 15%
    测试集: 数据集B (子页面) + 数据集A (连续会话) 15%
    预期: 在首页上表现好，在子页面和连续会话上表现差

实验2 (Ours: 全站指纹):
    训练集: 数据集B (首页 + 子页面) 70%
    验证集: 数据集B (首页 + 子页面) 15%
    测试集: 数据集A (连续会话) 100%
    预期: 跨场景泛化能力强，在连续会话上表现好

实验3 (进阶: 直接用连续会话训练):
    训练集: 数据集A 70%
    验证集: 数据集A 15%
    测试集: 数据集A 15%
    预期: 对比实验2，验证细粒度采集的有效性

Usage:
    python train_ablation.py --experiment 1
    python train_ablation.py --experiment 2 --epochs 100
    python train_ablation.py --experiment 3 --batch_size 64
"""

import os
import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from Model_NoDef_pytorch import DFNoDefNet


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "/home/pcz/DL/ML_DL/DeepFingerprinting/data/ablation_study"
OUTPUT_DIR = "/home/pcz/DL/ML_DL/DeepFingerprinting/checkpoints/ablation_study"

# 按批次划分比例 (避免数据泄露)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(experiment_name: str):
    """Setup logging configuration."""
    log_dir = Path(OUTPUT_DIR) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class DFDataset(Dataset):
    """Deep Fingerprinting Dataset - 惰性加载模式。"""

    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray, max_len: int = 5000):
        """
        Args:
            sequences: 变长序列列表 (方向序列: ±1)
            labels: 标签数组
            max_len: 固定序列长度
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = min(len(seq), self.max_len)

        # Pad/truncate + 转换
        x = np.zeros(self.max_len, dtype=np.float32)
        x[:seq_len] = seq[:seq_len].astype(np.float32)

        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_b(data_path: str) -> Tuple[List, np.ndarray, List, np.ndarray, Dict]:
    """
    Load Dataset B (single) with homepage and subpage separation.

    Returns:
        homepage_sequences, homepage_labels, subpage_sequences, subpage_labels, label_map
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['homepage_sequences'],
        data['homepage_labels'],
        data['subpage_sequences'],
        data['subpage_labels'],
        data['label_map']
    )


def load_dataset_a(data_path: str) -> Tuple[List, np.ndarray, Dict]:
    """
    Load Dataset A (batch) - aggregate session data.

    Returns:
        sequences, labels, label_map
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data['sequences'], data['labels'], data['label_map']


def align_labels(
    dataset_b_label_map: Dict,
    dataset_a_label_map: Dict
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Align label mappings between datasets B and A.

    Returns:
        unified_label_map: Unified label mapping
        b_to_unified: Mapping from B's labels to unified labels
        a_to_unified: Mapping from A's labels to unified labels
    """
    # 获取所有唯一的网站名称
    all_websites = set(dataset_b_label_map.values()) | set(dataset_a_label_map.values())
    unified_label_map = {i: website for i, website in enumerate(sorted(all_websites))}

    # 创建反向映射
    website_to_unified = {website: i for i, website in unified_label_map.items()}

    # 创建转换数组
    b_to_unified = np.array([
        website_to_unified[dataset_b_label_map[i]]
        for i in range(len(dataset_b_label_map))
    ])

    a_to_unified = np.array([
        website_to_unified[dataset_a_label_map[i]]
        for i in range(len(dataset_a_label_map))
    ])

    return unified_label_map, b_to_unified, a_to_unified


def split_by_batch(
    sequences: List,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Tuple[List, np.ndarray], Tuple[List, np.ndarray], Tuple[List, np.ndarray]]:
    """
    按批次划分数据集，避免数据泄露。

    这里简化处理：假设每个类别的样本已经按时间顺序排列。
    实际应用中，应根据实际的批次ID进行划分。
    """
    np.random.seed(seed)

    # 按类别分组
    unique_labels = np.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in unique_labels:
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]

        # 按顺序划分（模拟批次划分）
        n_samples = len(label_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])

    # 打乱每个集合内部的顺序（但保持批次分离）
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    train_sequences = [sequences[i] for i in train_indices]
    train_labels = labels[train_indices]

    val_sequences = [sequences[i] for i in val_indices]
    val_labels = labels[val_indices]

    test_sequences = [sequences[i] for i in test_indices]
    test_labels = labels[test_indices]

    return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)


def create_dataloaders(
    train_data: Tuple[List, np.ndarray],
    val_data: Tuple[List, np.ndarray],
    test_data: Tuple[List, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 0,
    input_len: int = 5000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from split data."""

    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data
    test_sequences, test_labels = test_data

    train_dataset = DFDataset(train_sequences, train_labels, input_len)
    val_dataset = DFDataset(val_sequences, val_labels, input_len)
    test_dataset = DFDataset(test_sequences, test_labels, input_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Training Engine
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict:
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class metrics
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_count = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_tpr = []
    per_class_fpr = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        count = cm[i, :].sum()

        per_class_count.append(count)
        per_class_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        per_class_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        per_class_tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        per_class_fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        p = per_class_precision[-1]
        r = per_class_recall[-1]
        per_class_f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

    tpr_avg = np.mean(per_class_tpr)
    fpr_avg = np.mean(per_class_fpr)

    return {
        'loss': total_loss / total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tpr_avg': tpr_avg,
        'fpr_avg': fpr_avg,
        'per_class_count': per_class_count,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_tpr': per_class_tpr,
        'per_class_fpr': per_class_fpr,
    }


def train_df(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    save_dir: str,
    logger,
) -> Tuple[nn.Module, Dict]:
    """Train DeepFingerprinting model."""
    model = model.to(device)

    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    # Infer num_classes from model
    num_classes = model.classifier.out_features

    for epoch in range(1, epochs + 1):
        epoch_start = datetime.now()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        val_metrics = evaluate(model, val_loader, device, num_classes)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'num_classes': num_classes,
            }, os.path.join(save_dir, 'best_model.pth'))

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"Time: {epoch_time:.1f}s {'*' if is_best else ''}"
        )

    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history


def print_evaluation_results(metrics: Dict, label_map: Dict, logger):
    """Print detailed evaluation results."""
    logger.info("\n" + "=" * 90)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 90)

    logger.info(f"\nOverall Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  TPR_AVE:   {metrics['tpr_avg']:.4f}")
    logger.info(f"  FPR_AVE:   {metrics['fpr_avg']:.4f}")

    logger.info("\n" + "-" * 90)
    logger.info("Per-Class Results:")
    logger.info("-" * 90)
    logger.info(f"{'Class':<25} {'Count':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TPR':>10} {'FPR':>10}")
    logger.info("-" * 90)

    for i, (count, precision, recall, f1, tpr, fpr) in enumerate(zip(
        metrics['per_class_count'],
        metrics['per_class_precision'],
        metrics['per_class_recall'],
        metrics['per_class_f1'],
        metrics['per_class_tpr'],
        metrics['per_class_fpr']
    )):
        class_name = label_map.get(i, f"Class_{i}")
        if len(class_name) > 24:
            class_name = class_name[:21] + "..."
        logger.info(f"{class_name:<25} {count:>8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tpr:>10.4f} {fpr:>10.4f}")


# =============================================================================
# Experiment Implementations
# =============================================================================

def experiment_1_baseline(
    logger,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    input_len: int = 5000,
) -> Dict[str, Any]:
    """
    实验1: 基准线 - 仅首页指纹

    训练集: 数据集B (仅首页) 70%
    验证集: 数据集B (仅首页) 15%
    测试集: 数据集B (子页面) + 数据集A (连续会话)
    """
    logger.info("=" * 70)
    logger.info("实验1: 基准线 - 仅首页指纹训练")
    logger.info("=" * 70)

    # Load datasets
    dataset_b_path = Path(DATA_DIR) / 'dataset_b_single.pkl'
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading datasets...")
    homepage_sequences, homepage_labels, subpage_sequences, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_sequences, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)
    homepage_labels = b_to_unified[homepage_labels]
    subpage_labels = b_to_unified[subpage_labels]
    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split homepage data (train/val from homepage only)
    train_data, val_data, _ = split_by_batch(
        homepage_sequences, homepage_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=seed
    )

    # Test set: subpage + aggregate
    test_sequences = subpage_sequences + aggregate_sequences
    test_labels = np.concatenate([subpage_labels, aggregate_labels], axis=0)
    test_data = (test_sequences, test_labels)

    logger.info(f"Train samples: {len(train_data[1])} (homepage only)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage only)")
    logger.info(f"Test samples: {len(test_data[1])} (subpage: {len(subpage_labels)}, aggregate: {len(aggregate_labels)})")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=batch_size,
        num_workers=0,
        input_len=input_len,
    )

    # Create model
    model = DFNoDefNet(num_classes=num_classes)

    # Train model
    save_dir = Path(OUTPUT_DIR) / 'experiment_1_baseline'
    save_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_df(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=str(save_dir),
        logger=logger,
    )

    # Test
    logger.info("\nEvaluating on test set...")
    metrics = evaluate(model, test_loader, device, num_classes)
    print_evaluation_results(metrics, unified_label_map, logger)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'label_map': unified_label_map,
    }


def experiment_2_proposed(
    logger,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    input_len: int = 5000,
) -> Dict[str, Any]:
    """
    实验2: 提出的方法 - 全站指纹

    训练集: 数据集B (首页 + 子页面) 70%
    验证集: 数据集B (首页 + 子页面) 15%
    测试集: 数据集A (连续会话) 100%
    """
    logger.info("=" * 70)
    logger.info("实验2: 提出的方法 - 全站指纹训练")
    logger.info("=" * 70)

    # Load datasets
    dataset_b_path = Path(DATA_DIR) / 'dataset_b_single.pkl'
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading datasets...")
    homepage_sequences, homepage_labels, subpage_sequences, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_sequences, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)

    # Combine all B data (homepage + subpage)
    all_b_sequences = homepage_sequences + subpage_sequences
    all_b_labels = np.concatenate([
        b_to_unified[homepage_labels],
        b_to_unified[subpage_labels]
    ], axis=0)

    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split B data (train/val)
    train_data, val_data, _ = split_by_batch(
        all_b_sequences, all_b_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=seed
    )

    # Test set: all of A
    test_data = (aggregate_sequences, aggregate_labels)

    logger.info(f"Train samples: {len(train_data[1])} (homepage + subpage)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage + subpage)")
    logger.info(f"Test samples: {len(test_data[1])} (aggregate session)")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=batch_size,
        num_workers=0,
        input_len=input_len,
    )

    # Create model
    model = DFNoDefNet(num_classes=num_classes)

    # Train model
    save_dir = Path(OUTPUT_DIR) / 'experiment_2_proposed'
    save_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_df(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=str(save_dir),
        logger=logger,
    )

    # Test
    logger.info("\nEvaluating on test set...")
    metrics = evaluate(model, test_loader, device, num_classes)
    print_evaluation_results(metrics, unified_label_map, logger)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'label_map': unified_label_map,
    }


def experiment_3_aggregate(
    logger,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    input_len: int = 5000,
) -> Dict[str, Any]:
    """
    实验3: 进阶对比 - 直接用连续会话训练

    训练集: 数据集A 70%
    验证集: 数据集A 15%
    测试集: 数据集A 15%
    """
    logger.info("=" * 70)
    logger.info("实验3: 进阶对比 - 连续会话训练")
    logger.info("=" * 70)

    # Load dataset A
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading dataset A...")
    aggregate_sequences, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    num_classes = len(label_map_a)
    logger.info(f"Total classes: {num_classes}")

    # Split A data
    train_data, val_data, test_data = split_by_batch(
        aggregate_sequences, aggregate_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=seed
    )

    logger.info(f"Train samples: {len(train_data[1])}")
    logger.info(f"Val samples: {len(val_data[1])}")
    logger.info(f"Test samples: {len(test_data[1])}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=batch_size,
        num_workers=0,
        input_len=input_len,
    )

    # Create model
    model = DFNoDefNet(num_classes=num_classes)

    # Train model
    save_dir = Path(OUTPUT_DIR) / 'experiment_3_aggregate'
    save_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_df(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=str(save_dir),
        logger=logger,
    )

    # Test
    logger.info("\nEvaluating on test set...")
    metrics = evaluate(model, test_loader, device, num_classes)
    print_evaluation_results(metrics, label_map_a, logger)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'label_map': label_map_a,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DeepFingerprinting Ablation Study')

    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                        help='Experiment number (1: baseline, 2: proposed, 3: aggregate)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--input_len', type=int, default=5000,
                        help='Fixed input sequence length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Setup logging
    experiment_name = f"experiment_{args.experiment}"
    logger = setup_logging(experiment_name)

    logger.info("DeepFingerprinting Ablation Study")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Input length: {args.input_len}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiment
    if args.experiment == 1:
        results = experiment_1_baseline(
            logger, args.epochs, args.batch_size, args.lr, args.seed, device, args.input_len
        )
    elif args.experiment == 2:
        results = experiment_2_proposed(
            logger, args.epochs, args.batch_size, args.lr, args.seed, device, args.input_len
        )
    elif args.experiment == 3:
        results = experiment_3_aggregate(
            logger, args.epochs, args.batch_size, args.lr, args.seed, device, args.input_len
        )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    logger.info("\n" + "=" * 70)
    logger.info("Experiment Complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
