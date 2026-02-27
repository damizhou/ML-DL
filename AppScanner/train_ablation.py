"""
AppScanner Ablation Study Training Script

实现三个消融实验：

实验1 (Baseline: 首页指纹):
    训练集: 数据集B (仅首页) 80%
    验证集: 数据集B (仅首页) 10%
    测试集: 数据集B (子页面) + 数据集A (连续会话) 10%
    预期: 在首页上表现好，在子页面和连续会话上表现差

实验2 (Ours: 全站指纹):
    训练集: 数据集B (首页 + 子页面) 80%
    验证集: 数据集B (首页 + 子页面) 10%
    测试集: 数据集A (连续会话) 100%
    预期: 跨场景泛化能力强，在连续会话上表现好

实验3 (进阶: 直接用连续会话训练):
    训练集: 数据集A 80%
    验证集: 数据集A 10%
    测试集: 数据集A 10%
    预期: 对比实验2，验证细粒度采集的有效性

Usage:
    python train_ablation.py --experiment 1 --model nn
    python train_ablation.py --experiment 2 --model rf
    python train_ablation.py --experiment 3 --model nn --epochs 100
"""

import os
import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import AppScannerNN, AppScannerRF, build_model
from data import AppScannerDataset, create_dataloaders_from_split
from engine import train, test, train_random_forest
from config import AppScannerConfig


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data/ablation_study"
OUTPUT_DIR = "checkpoints/ablation_study"

# 按批次划分比例 (避免数据泄露)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str):
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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_b(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset B (single) with homepage and subpage separation.

    Returns:
        homepage_features, homepage_labels, subpage_features, subpage_labels, label_map
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['homepage_features'],
        data['homepage_labels'],
        data['subpage_features'],
        data['subpage_labels'],
        data['label_map']
    )


def load_dataset_a(data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset A (batch) - aggregate session data.

    Returns:
        features, labels, label_map
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data['features'], data['labels'], data['label_map']


def sanitize_features(features: np.ndarray, logger, name: str) -> np.ndarray:
    """Replace NaN/Inf values and log counts."""
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(
            f"{name}: found {nan_count} NaN and {inf_count} Inf values, replacing with 0"
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


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
    features: np.ndarray,
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

    train_data = (features[train_indices], labels[train_indices])
    val_data = (features[val_indices], labels[val_indices])
    test_data = (features[test_indices], labels[test_indices])

    return train_data, val_data, test_data


# =============================================================================
# Experiment Implementations
# =============================================================================

def experiment_1_baseline(
    logger,
    config: AppScannerConfig,
    model_type: str = 'nn'
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
    homepage_features, homepage_labels, subpage_features, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_features, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    homepage_features = sanitize_features(homepage_features, logger, "homepage_features")
    subpage_features = sanitize_features(subpage_features, logger, "subpage_features")
    aggregate_features = sanitize_features(aggregate_features, logger, "aggregate_features")

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)
    homepage_labels = b_to_unified[homepage_labels]
    subpage_labels = b_to_unified[subpage_labels]
    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split homepage data (train/val from homepage only)
    train_data, val_data, _ = split_by_batch(
        homepage_features, homepage_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config.seed
    )

    # Test set: subpage + aggregate
    test_features = np.concatenate([subpage_features, aggregate_features], axis=0)
    test_labels = np.concatenate([subpage_labels, aggregate_labels], axis=0)
    test_data = (test_features, test_labels)

    logger.info(f"Train samples: {len(train_data[1])} (homepage only)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage only)")
    logger.info(f"Test samples: {len(test_data[1])} (subpage: {len(subpage_labels)}, aggregate: {len(aggregate_labels)})")

    # Update config
    config.num_classes = num_classes

    # Create dataloaders (cross-domain: train=homepage, test=subpage+aggregate)
    train_loader, val_loader, test_loader, norm_params = create_dataloaders_from_split(
        train_data, val_data, test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        independent_test_norm=True,
    )


    # Train model
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_type == 'nn':
        model = build_model('nn', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, unified_label_map)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': unified_label_map,
        }

    elif model_type == 'deep':
        model = build_model('deep', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, unified_label_map)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': unified_label_map,
        }

    elif model_type == 'rf':
        train_features, train_labels = train_data
        val_features, val_labels = val_data
        test_features, test_labels = test_data

        results = train_random_forest(
            train_features, train_labels,
            test_features, test_labels,
            n_estimators=config.n_estimators,
            prediction_threshold=config.prediction_threshold,
            X_val=val_features,
            y_val=val_labels,
            label_map=unified_label_map,
        )

        return results

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def experiment_2_proposed(
    logger,
    config: AppScannerConfig,
    model_type: str = 'nn'
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
    homepage_features, homepage_labels, subpage_features, subpage_labels, label_map_b = \
        load_dataset_b(str(dataset_b_path))
    aggregate_features, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    homepage_features = sanitize_features(homepage_features, logger, "homepage_features")
    subpage_features = sanitize_features(subpage_features, logger, "subpage_features")
    aggregate_features = sanitize_features(aggregate_features, logger, "aggregate_features")

    # Align labels
    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)

    # Combine all B data (homepage + subpage)
    all_b_features = np.concatenate([homepage_features, subpage_features], axis=0)
    all_b_labels = np.concatenate([
        b_to_unified[homepage_labels],
        b_to_unified[subpage_labels]
    ], axis=0)

    aggregate_labels = a_to_unified[aggregate_labels]

    num_classes = len(unified_label_map)
    logger.info(f"Total classes: {num_classes}")

    # Split B data (train/val)
    train_data, val_data, _ = split_by_batch(
        all_b_features, all_b_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config.seed
    )

    # Test set: all of A
    test_data = (aggregate_features, aggregate_labels)

    logger.info(f"Train samples: {len(train_data[1])} (homepage + subpage)")
    logger.info(f"Val samples: {len(val_data[1])} (homepage + subpage)")
    logger.info(f"Test samples: {len(test_data[1])} (aggregate session)")

    # Update config
    config.num_classes = num_classes

    # Create dataloaders (cross-domain: train=single pages, test=aggregate sessions)
    train_loader, val_loader, test_loader, norm_params = create_dataloaders_from_split(
        train_data, val_data, test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        independent_test_norm=True,
    )


    # Train model
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_type == 'nn':
        model = build_model('nn', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, unified_label_map)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': unified_label_map,
        }

    elif model_type == 'deep':
        model = build_model('deep', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, unified_label_map)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': unified_label_map,
        }

    elif model_type == 'rf':
        train_features, train_labels = train_data
        val_features, val_labels = val_data
        test_features, test_labels = test_data

        results = train_random_forest(
            train_features, train_labels,
            test_features, test_labels,
            n_estimators=config.n_estimators,
            prediction_threshold=config.prediction_threshold,
            X_val=val_features,
            y_val=val_labels,
            label_map=unified_label_map,
        )

        return results

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def experiment_3_aggregate(
    logger,
    config: AppScannerConfig,
    model_type: str = 'nn'
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
    aggregate_features, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    aggregate_features = sanitize_features(aggregate_features, logger, "aggregate_features")

    num_classes = len(label_map_a)
    logger.info(f"Total classes: {num_classes}")

    # Split A data
    train_data, val_data, test_data = split_by_batch(
        aggregate_features, aggregate_labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config.seed
    )

    logger.info(f"Train samples: {len(train_data[1])}")
    logger.info(f"Val samples: {len(val_data[1])}")
    logger.info(f"Test samples: {len(test_data[1])}")

    # Update config
    config.num_classes = num_classes

    # Create dataloaders (same domain)
    train_loader, val_loader, test_loader, norm_params = create_dataloaders_from_split(
        train_data, val_data, test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


    # Train model
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_type == 'nn':
        model = build_model('nn', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, label_map_a)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': label_map_a,
        }

    elif model_type == 'deep':
        model = build_model('deep', input_dim=54, num_classes=num_classes)
        model, history = train(model, train_loader, val_loader, config, save_dir=str(save_dir))

        # Test
        device = torch.device(config.device)
        metrics = test(model, test_loader, device, config.prediction_threshold, label_map_a)

        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'label_map': label_map_a,
        }

    elif model_type == 'rf':
        train_features, train_labels = train_data
        val_features, val_labels = val_data
        test_features, test_labels = test_data

        results = train_random_forest(
            train_features, train_labels,
            test_features, test_labels,
            n_estimators=config.n_estimators,
            prediction_threshold=config.prediction_threshold,
            X_val=val_features,
            y_val=val_labels,
            label_map=label_map_a,
        )

        return results

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Main
# =============================================================================

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description='AppScanner Ablation Study')

    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                        help='Experiment number (1: baseline, 2: proposed, 3: aggregate)')
    parser.add_argument('--model', type=str, default='nn', choices=['nn', 'rf', 'deep'],
                        help='Model type (nn, rf, deep)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup logging/output directory per experiment run
    run_dir = Path(OUTPUT_DIR) / f"experiment{args.experiment}_{datetime.now().strftime('%Y%m%d')}"
    OUTPUT_DIR = str(run_dir)
    logger = setup_logging(OUTPUT_DIR)

    logger.info("AppScanner Ablation Study")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Split ratio: train {TRAIN_RATIO:.2f} | val {VAL_RATIO:.2f} | test {TEST_RATIO:.2f}")

    # Create config
    config = AppScannerConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.seed = args.seed
    config.num_workers = 4 if args.model in {'nn', 'deep'} else 0

    # Run experiment
    if args.experiment == 1:
        results = experiment_1_baseline(logger, config, args.model)
    elif args.experiment == 2:
        results = experiment_2_proposed(logger, config, args.model)
    elif args.experiment == 3:
        results = experiment_3_aggregate(logger, config, args.model)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    logger.info("\n" + "=" * 70)
    logger.info("Experiment Complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
