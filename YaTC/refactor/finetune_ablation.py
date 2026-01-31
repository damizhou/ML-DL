#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC Ablation Study Fine-tuning Script

针对消融实验的微调脚本，加载预训练权重进行有监督分类训练。

核心目的：证明"采集子页面是必要的"以及"全站指纹采集策略是有效的"。

设计原则：
    - 三个实验共享同一个预训练模型（使用外部数据集预训练）
    - 唯一变量是微调数据的来源
    - 控制变量，确保实验公平

实验1 (Baseline: 首页指纹):
    训练集: 数据集B (仅首页) 80%
    验证集: 数据集B (仅首页) 10%
    测试集: 数据集B (子页面) + 数据集A (连续会话)
    预期: 在首页上表现好，在子页面和连续会话上表现差
    结论: 首页指纹无法泛化到全站流量

实验2 (Ours: 全站指纹):
    训练集: 数据集B (首页 + 子页面) 80%
    验证集: 数据集B (首页 + 子页面) 10%
    测试集: 数据集A (连续会话) 100%
    预期: 跨场景泛化能力强，在连续会话上表现好
    结论: 覆盖子页面的采集是必要的

实验3 (进阶: 直接用连续会话训练):
    训练集: 数据集A 80%
    验证集: 数据集A 10%
    测试集: 数据集A 10%
    预期: 对比实验2，验证细粒度采集的有效性
    结论: 验证细粒度采集策略的稳健性

Usage:
    # 先使用 pretrain.py 在外部数据集上预训练
    python pretrain.py  # 生成 ../checkpoints/pretrained.pth

    # 三个实验共享同一个预训练模型
    python finetune_ablation.py --experiment 1 --pretrained ../checkpoints/pretrained.pth
    python finetune_ablation.py --experiment 2 --pretrained ../checkpoints/pretrained.pth
    python finetune_ablation.py --experiment 3 --pretrained ../checkpoints/pretrained.pth

    # 不使用预训练，从头训练
    python finetune_ablation.py --experiment 1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from models import traformer_yatc
from engine import (
    train_one_epoch,
    evaluate,
    get_param_groups_with_layer_decay,
    load_pretrained_weights,
    save_checkpoint,
)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "/root/autodl-tmp/YaTC/data/ablation_study"
OUTPUT_DIR = "/root/YaTC/checkpoints/ablation_study"
PRETRAINED_PATH = Path(__file__).parent.parent / "checkpoints" / "pretrained.pth"
# 按批次划分比例 (避免数据泄露)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# =============================================================================
# Dataset Classes
# =============================================================================

class AblationDataset(Dataset):
    """Dataset for ablation study with MFR images."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        id2label: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            images: MFR images of shape (N, 40, 40)
            labels: Labels of shape (N,)
            id2label: Label ID to name mapping
        """
        self.images = images
        self.labels = labels
        self.id2label = id2label or {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        return img_tensor, label


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_b(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset B (single page) with homepage and subpage separation.

    Expected PKL structure:
    {
        "homepage_images": (N1, 40, 40),
        "homepage_labels": (N1,),
        "subpage_images": (N2, 40, 40),
        "subpage_labels": (N2,),
        "label_map": {"id2label": {...}, "label2id": {...}}
    }

    Returns:
        homepage_images, homepage_labels, subpage_images, subpage_labels, label_map
    """
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['homepage_images'],
        data['homepage_labels'],
        data['subpage_images'],
        data['subpage_labels'],
        data['label_map']
    )


def load_dataset_a(data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset A (aggregate session) data.

    Expected PKL structure:
    {
        "images": (N, 40, 40),
        "labels": (N,),
        "label_map": {"id2label": {...}, "label2id": {...}}
    }

    Returns:
        images, labels, label_map
    """
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data['images'], data['labels'], data['label_map']


def align_labels(
    label_map_b: Dict,
    label_map_a: Dict
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Align label mappings between datasets B and A.

    Returns:
        unified_label_map: Unified label mapping {id: name}
        b_to_unified: Mapping array from B's labels to unified labels
        a_to_unified: Mapping array from A's labels to unified labels
    """
    id2label_b = label_map_b.get('id2label', label_map_b)
    id2label_a = label_map_a.get('id2label', label_map_a)

    # Convert keys to int if necessary
    if isinstance(list(id2label_b.keys())[0], str):
        id2label_b = {int(k): v for k, v in id2label_b.items()}
    if isinstance(list(id2label_a.keys())[0], str):
        id2label_a = {int(k): v for k, v in id2label_a.items()}

    # Get all unique website names
    all_websites = set(id2label_b.values()) | set(id2label_a.values())
    unified_label_map = {i: website for i, website in enumerate(sorted(all_websites))}

    # Create reverse mapping
    website_to_unified = {website: i for i, website in unified_label_map.items()}

    # Create conversion arrays
    max_b = max(id2label_b.keys()) + 1
    max_a = max(id2label_a.keys()) + 1

    b_to_unified = np.zeros(max_b, dtype=np.int64)
    for old_id, name in id2label_b.items():
        b_to_unified[old_id] = website_to_unified[name]

    a_to_unified = np.zeros(max_a, dtype=np.int64)
    for old_id, name in id2label_a.items():
        a_to_unified[old_id] = website_to_unified[name]

    return unified_label_map, b_to_unified, a_to_unified


def split_by_batch(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

    train_data = (images[train_indices], labels[train_indices])
    val_data = (images[val_indices], labels[val_indices])
    test_data = (images[test_indices], labels[test_indices])

    return train_data, val_data, test_data


def create_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    id2label: Optional[Dict] = None
) -> DataLoader:
    """Create DataLoader from images and labels."""
    dataset = AblationDataset(images, labels, id2label)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle
    )


# =============================================================================
# Training Functions
# =============================================================================

def train_and_evaluate(
    logger,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    id2label: Dict[int, str],
    config: Dict[str, Any],
    save_dir: Path
) -> Dict[str, Any]:
    """
    Train and evaluate YaTC model.

    Returns:
        Dictionary containing model, history, and metrics
    """
    device = torch.device(config['device'])

    # Enable cuDNN optimization
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Create model
    model = traformer_yatc(
        num_classes=num_classes,
        drop_path_rate=config['drop_path_rate']
    )

    # Load pre-trained weights if provided
    if config.get('pretrained_path') and Path(config['pretrained_path']).exists():
        model = load_pretrained_weights(model, config['pretrained_path'])
        logger.info(f"Loaded pre-trained weights from: {config['pretrained_path']}")
    else:
        logger.info("Training from scratch (no pre-trained weights)")

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with layer-wise learning rate decay
    param_groups = get_param_groups_with_layer_decay(
        model=model,
        base_lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        layer_decay=config['layer_decay'],
        num_layers=4
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['learning_rate'],
        betas=(0.9, 0.999)
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
    }

    # Training loop
    best_val_acc = 0.0
    best_model_path = save_dir / "yatc_best.pth"

    logger.info(f"\nStarting training for {config['epochs']} epochs...")
    logger.info("-" * 60)

    for epoch in range(config['epochs']):
        epoch_start_time = datetime.now()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=config['epochs'],
            warmup_epochs=config['warmup_epochs'],
            base_lr=config['learning_rate'],
            print_freq=config['print_freq']
        )

        # Evaluate on validation set
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes
        )

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        marker = ""
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=0,
                loss=train_metrics['loss'],
                output_path=best_model_path
            )
            marker = " *"

        # Calculate epoch time
        epoch_end_time = datetime.now()
        epoch_elapsed = epoch_end_time - epoch_start_time
        epoch_seconds = int(epoch_elapsed.total_seconds())
        epoch_mins, epoch_secs = divmod(epoch_seconds, 60)

        logger.info(
            f"Epoch {epoch+1:3d} | Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Time: {epoch_mins:02d}:{epoch_secs:02d}{marker}"
        )

    # Final test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Test Results")
    logger.info("=" * 60)

    # Load best model
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=True)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=num_classes
    )

    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 Score:  {test_metrics['f1']:.4f}")

    # Get predictions for classification report
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for samples, targets in test_loader:
            samples = samples.to(device)
            outputs = model(samples)
            _, preds = outputs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    # Print classification report
    logger.info("\n" + "-" * 60)
    logger.info("Classification Report:")
    logger.info("-" * 60)
    labels_list = list(range(num_classes))
    target_names = [id2label.get(i, str(i)) for i in labels_list]
    logger.info(classification_report(
        all_labels, all_preds,
        labels=labels_list,
        target_names=target_names,
        zero_division=0
    ))

    return {
        'model': model,
        'history': history,
        'metrics': test_metrics,
        'label_map': id2label,
    }


# =============================================================================
# Experiment Implementations
# =============================================================================

def experiment_1_baseline(
    logger,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    实验1: 基准线 - 仅首页指纹

    训练集: 数据集B (仅首页) 80%
    验证集: 数据集B (仅首页) 10%
    测试集: 数据集B (子页面) + 数据集A (连续会话)
    """
    logger.info("=" * 70)
    logger.info("实验1: 基准线 - 仅首页指纹训练")
    logger.info("=" * 70)

    # Load datasets
    dataset_b_path = Path(DATA_DIR) / 'dataset_b_single.pkl'
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading datasets...")
    homepage_images, homepage_labels, subpage_images, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_images, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)
    homepage_labels = b_to_unified[homepage_labels]
    subpage_labels = b_to_unified[subpage_labels]
    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split homepage data (train/val from homepage only)
    train_data, val_data, _ = split_by_batch(
        homepage_images, homepage_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config['seed']
    )

    # Test set: subpage + aggregate
    test_images = np.concatenate([subpage_images, aggregate_images], axis=0)
    test_labels = np.concatenate([subpage_labels, aggregate_labels], axis=0)

    logger.info(f"Train samples: {len(train_data[1])} (homepage only)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage only)")
    logger.info(f"Test samples: {len(test_labels)} (subpage: {len(subpage_labels)}, aggregate: {len(aggregate_labels)})")

    # Create dataloaders
    train_loader = create_dataloader(
        train_data[0], train_data[1],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = create_dataloader(
        val_data[0], val_data[1],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    test_loader = create_dataloader(
        test_images, test_labels,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        id2label=unified_label_map
    )

    # Train and evaluate
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    return train_and_evaluate(
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        id2label=unified_label_map,
        config=config,
        save_dir=save_dir
    )


def experiment_2_proposed(
    logger,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    实验2: 提出的方法 - 全站指纹

    训练集: 数据集B (首页 + 子页面) 80%
    验证集: 数据集B (首页 + 子页面) 10%
    测试集: 数据集A (连续会话) 100%
    """
    logger.info("=" * 70)
    logger.info("实验2: 提出的方法 - 全站指纹训练")
    logger.info("=" * 70)

    # Load datasets
    dataset_b_path = Path(DATA_DIR) / 'dataset_b_single.pkl'
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading datasets...")
    homepage_images, homepage_labels, subpage_images, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_images, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)

    # Combine all B data (homepage + subpage)
    all_b_images = np.concatenate([homepage_images, subpage_images], axis=0)
    all_b_labels = np.concatenate([
        b_to_unified[homepage_labels],
        b_to_unified[subpage_labels]
    ], axis=0)

    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split B data (train/val)
    train_data, val_data, _ = split_by_batch(
        all_b_images, all_b_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config['seed']
    )

    logger.info(f"Train samples: {len(train_data[1])} (homepage + subpage)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage + subpage)")
    logger.info(f"Test samples: {len(aggregate_labels)} (aggregate session)")

    # Create dataloaders
    train_loader = create_dataloader(
        train_data[0], train_data[1],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = create_dataloader(
        val_data[0], val_data[1],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    test_loader = create_dataloader(
        aggregate_images, aggregate_labels,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        id2label=unified_label_map
    )

    # Train and evaluate
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    return train_and_evaluate(
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        id2label=unified_label_map,
        config=config,
        save_dir=save_dir
    )


def experiment_3_aggregate(
    logger,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    实验3: 进阶对比 - 直接用连续会话训练

    训练集: 数据集A 80%
    验证集: 数据集A 10%
    测试集: 数据集A 10%
    """
    logger.info("=" * 70)
    logger.info("实验3: 进阶对比 - 连续会话训练")
    logger.info("=" * 70)

    # Load dataset A
    dataset_a_path = Path(DATA_DIR) / 'dataset_a_batch.pkl'

    logger.info("Loading dataset A...")
    aggregate_images, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    # Get id2label mapping
    id2label = label_map_a.get('id2label', label_map_a)
    if isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}

    num_classes = len(id2label)
    logger.info(f"Total classes: {num_classes}")

    # Split A data
    train_data, val_data, test_data = split_by_batch(
        aggregate_images, aggregate_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config['seed']
    )

    logger.info(f"Train samples: {len(train_data[1])}")
    logger.info(f"Val samples: {len(val_data[1])}")
    logger.info(f"Test samples: {len(test_data[1])}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_data[0], train_data[1],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = create_dataloader(
        val_data[0], val_data[1],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    test_loader = create_dataloader(
        test_data[0], test_data[1],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        id2label=id2label
    )

    # Train and evaluate
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    return train_and_evaluate(
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        id2label=id2label,
        config=config,
        save_dir=save_dir
    )


# =============================================================================
# Main
# =============================================================================

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description='YaTC Ablation Study')

    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                        help='Experiment number (1: baseline, 2: proposed, 3: aggregate)')
    parser.add_argument('--pretrained', type=str, default=PRETRAINED_PATH,
                        help='Path to pre-trained model weights')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='Layer-wise learning rate decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency')

    args = parser.parse_args()

    # Setup output directory per experiment run
    run_dir = Path(OUTPUT_DIR) / f"experiment{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = str(run_dir)
    logger = setup_logging(OUTPUT_DIR)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("YaTC Ablation Study")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Device: {device}")
    logger.info(f"Pretrained: {args.pretrained or 'None'}")
    logger.info(f"Split ratio: train {TRAIN_RATIO:.2f} | val {VAL_RATIO:.2f} | test {TEST_RATIO:.2f}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Layer decay: {args.layer_decay}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Drop path rate: {args.drop_path_rate}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info("=" * 60)

    # Create config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'layer_decay': args.layer_decay,
        'warmup_epochs': args.warmup_epochs,
        'drop_path_rate': args.drop_path_rate,
        'label_smoothing': args.label_smoothing,
        'seed': args.seed,
        'num_workers': args.num_workers,
        'print_freq': args.print_freq,
        'device': device,
        'pretrained_path': args.pretrained,
    }

    # Record start time
    start_time = datetime.now()

    # Run experiment
    if args.experiment == 1:
        results = experiment_1_baseline(logger, config)
    elif args.experiment == 2:
        results = experiment_2_proposed(logger, config)
    elif args.experiment == 3:
        results = experiment_3_aggregate(logger, config)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    # Record end time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("\n" + "=" * 70)
    logger.info("Experiment Complete!")
    logger.info("=" * 70)
    logger.info(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    # Save results
    results_path = Path(OUTPUT_DIR) / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': args.experiment,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            'metrics': results['metrics'],
            'history': results['history'],
        }, f, indent=2)
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
