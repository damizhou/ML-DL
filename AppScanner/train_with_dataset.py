"""
AppScanner Main Training Script

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

Usage:
    python train.py
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from config import AppScannerConfig, get_config


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


def log(message: str = ""):
    """Log message to both console and file."""
    logging.info(message)


from models import AppScannerNN, AppScannerDeep, build_model
from data import (
    create_dataset_from_directory,
    create_dataset_from_csv,
    create_dataloaders,
    save_dataset,
    load_dataset,
)
from engine import (
    train,
    test,
    train_random_forest,
    compare_approaches,
)
from resume_rf_evaluation import evaluate_saved_forest_splits


# =============================================================================
# Configuration - Modify these parameters directly
# =============================================================================

@dataclass
class TrainArgs:
    """Training arguments - modify these directly instead of command line."""

    # Mode: 'train', 'eval', 'extract', 'compare'
    mode: str = 'train'  # 运行模式：训练/评估/特征提取/方案对比。

    # Data paths
    data_dir: str = './data'  # 原始 PCAP 目录（mode=extract 或无特征文件时使用）。
    csv_path: Optional[str] = None  # CSV 特征文件路径（优先于 data_dir）。
    features_paths: List[str] = None  # 批量运行的数据集路径列表（main 循环逐个处理）。
    features_path: Optional[str] = None  # 当前正在处理的单个数据集路径（循环内自动赋值）。

    # Model configuration
    model_type: str = 'rf'  # 模型类型：'nn' / 'deep' / 'rf'。
    num_classes: Optional[int] = None  # 类别数；None 时从数据自动推断。
    input_dim: int = 54  # 输入维度（AppScanner 统计特征维度，默认 54）。
    hidden_dims: List[int] = None  # MLP 隐层维度列表；None 时在 __post_init__ 设置默认值。
    dropout: float = 0.3  # NN/Deep 模型 dropout 比例。

    # Training parameters
    epochs: int = 100  # 神经网络训练轮数。
    batch_size: int = 128  # 神经网络 DataLoader 批大小。
    lr: float = 0.001  # 学习率。
    weight_decay: float = 1e-4  # 权重衰减（L2 正则）。
    patience: int = 10  # 早停耐心轮数。

    # AppScanner specific
    prediction_threshold: float = 0.9  # 置信度阈值（用于 confidence_accuracy）。
    min_flow_length: int = 7  # 流最小包数（短于该值会被过滤）。
    max_flow_length: int = 260  # 流最大包数（超过该值会截断）。

    # Random Forest
    n_estimators: int = 100  # 随机森林树数量。
    rf_max_depth: Optional[int] = 20  # 单棵树最大深度；None 表示不限制。
    rf_trees_per_batch: int = 25  # 训练阶段每批并行训练的树数（n_jobs）。
    rf_val_trees_per_batch: Optional[int] = 5  # 评估 val 时 batch_first 路径的并行树数。
    rf_test_trees_per_batch: Optional[int] = 5  # 评估 test 时 batch_first 路径的并行树数。
    rf_eval_batch_size: Optional[int] = None  # 评估样本批大小；None 按概率缓冲自动估算。
    rf_eval_prob_buffer_mb: int = 256  # 评估自动批大小的概率缓冲内存预算（MB）。
    rf_eval_strategy: str = 'tree_first'  # 评估策略：auto | batch_first | tree_first。
    rf_tree_first_max_prob_mb: int = 4096  # tree_first 全量概率矩阵上限（MB）；超出时会回退到 batch_first。
    rf_tree_prefetch: int = 1  # tree_first 模式预加载队列长度（后台 I/O 预取树数）。
    rf_tree_eval_workers: int = 5  # tree_first 每批并行评估树数（每组 10 棵并行评估）。
    rf_log_each_tree_time: bool = True  # 是否打印每棵树评估耗时。
    rf_combine_val_test: bool = True  # 是否将 val/test 合并一次推理后再切分结果。
    rf_progress_tree_step: int = 1  # 训练阶段每隔多少棵树打印一次进度。

    # Paths
    output_dir: str = './output'  # 输出目录（模型、日志、结果文件）。
    checkpoint: Optional[str] = None  # 评估模式加载的模型检查点路径。

    # Device: 'auto', 'cuda', 'cpu'
    device: str = 'auto'  # 设备选择：auto 自动选 CUDA/CPU。

    # Misc
    seed: int = 42  # 全局随机种子（数据划分与训练可复现）。
    num_workers: int = 4  # DataLoader 子进程数（主要影响 NN/Deep 训练与评估）。

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        if self.rf_val_trees_per_batch is None:
            self.rf_val_trees_per_batch = 10
        if self.rf_test_trees_per_batch is None:
            self.rf_test_trees_per_batch = 10
        if self.features_paths is None:
            self.features_paths = [
                '/home/pcz/code/DL/AppScanner/data/vpn/vpn_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn/novpn_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn_top10/novpn_top10_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/vpn_top10/vpn_top10_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn_top50/novpn_top50_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/vpn_top50/vpn_top50_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn_top100/novpn_top100_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/vpn_top100/vpn_top100_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn_top500/novpn_top500_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/vpn_top500/vpn_top500_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/novpn_top1000/novpn_top1000_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/vpn_top1000/vpn_top1000_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/ustc/ustc_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/iscxvpn/iscxvpn_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/iscxtor/iscxtor_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/cross_platform/cross_platform_appscanner.pkl',
                # '/home/pcz/code/DL/AppScanner/data/cic_iot_2022/cic_iot_2022_appscanner.pkl',
            ]


def get_args() -> TrainArgs:
    """Get training arguments with optional command line override."""
    import argparse

    parser = argparse.ArgumentParser(description='AppScanner Training Script')
    parser.add_argument('--data_path', type=str, nargs='+', default=None,
                        help='Path(s) to pickle file(s) (overrides default list)')

    args = parser.parse_args()

    # Create TrainArgs with defaults
    train_args = TrainArgs()

    # Override features_paths if provided via command line
    if args.data_path is not None:
        train_args.features_paths = args.data_path

    return train_args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_config_from_args(args) -> AppScannerConfig:
    """Create configuration from command line arguments."""
    config = get_config()

    # Update with args
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.patience = args.patience
    config.prediction_threshold = args.prediction_threshold
    config.min_flow_length = args.min_flow_length
    config.max_flow_length = args.max_flow_length
    config.n_estimators = args.n_estimators
    config.max_depth = args.rf_max_depth
    config.hidden_dims = args.hidden_dims
    config.dropout = args.dropout
    config.input_dim = args.input_dim
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.num_workers = args.num_workers

    if args.num_classes is not None:
        config.num_classes = args.num_classes

    # Device
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device

    return config


def load_data(args, config):
    """Load data based on arguments."""
    if args.features_path is not None:
        log(f"Loading pre-extracted features from {args.features_path}")
        features, labels, label_map = load_dataset(args.features_path)
    elif args.csv_path is not None:
        log(f"Loading features from CSV: {args.csv_path}")
        features, labels, label_map = create_dataset_from_csv(args.csv_path)
    else:
        log(f"Extracting features from PCAP files in {args.data_dir}")
        features, labels, label_map = create_dataset_from_directory(
            args.data_dir,
            min_flow_length=config.min_flow_length,
            max_flow_length=config.max_flow_length,
        )

    # Update num_classes if not specified
    if args.num_classes is None:
        config.num_classes = len(label_map)

    return features, labels, label_map


def mode_train(args, config):
    """Training mode."""
    log("=" * 60)
    log("AppScanner Training")
    log("=" * 60)

    # Load data
    features, labels, label_map = load_data(args, config)
    log(f"Features shape: {features.shape}")
    log(f"Number of classes: {config.num_classes}")

    if args.model_type == 'rf':
        # --- Random Forest branch (memory-optimized) ---
        log(f"\nModel: rf (n_estimators={config.n_estimators}, max_depth={args.rf_max_depth})")
        log(f"RF train trees_per_batch: {args.rf_trees_per_batch}")
        log(f"RF val trees_per_batch: {args.rf_val_trees_per_batch}")
        log(f"RF test trees_per_batch: {args.rf_test_trees_per_batch}")
        log(f"RF eval batch_size: {args.rf_eval_batch_size}")
        log(f"RF eval prob_buffer_mb: {args.rf_eval_prob_buffer_mb}")
        log(f"RF eval strategy: {args.rf_eval_strategy}")
        log(f"RF tree_first max_prob_mb: {args.rf_tree_first_max_prob_mb}")
        log(f"RF tree prefetch: {args.rf_tree_prefetch}")
        log(f"RF tree eval workers: {args.rf_tree_eval_workers}")
        log(f"RF log each tree time: {args.rf_log_each_tree_time}")
        log(f"RF combine val+test: {args.rf_combine_val_test}")

        # Split indices only (no data copy)
        np.random.seed(config.seed)
        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * config.train_ratio)
        n_val = int(n_samples * config.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        log(f"Training samples: {n_train}")
        log(f"Validation samples: {len(val_idx)}")
        log(f"Test samples: {len(test_idx)}")

        # Phase 1: Build X_train only, release features
        import gc
        X_train = features[train_idx]
        y_train = labels[train_idx]
        # Save split info for later phases
        data_path = args.features_path
        del features, labels, indices
        gc.collect()

        # Train (trees saved to disk in parallel batches)
        results = train_random_forest(
            X_train, y_train,
            X_test=None, y_test=None,  # defer evaluation
            n_estimators=config.n_estimators,
            prediction_threshold=config.prediction_threshold,
            n_jobs=args.rf_trees_per_batch,
            max_depth=args.rf_max_depth,
            progress_tree_step=args.rf_progress_tree_step,
            label_map=label_map,
            save_dir=config.output_dir,
            seed=config.seed,
            compute_train_metrics=False,
            eval_batch_size=args.rf_eval_batch_size,
            eval_prob_buffer_mb=args.rf_eval_prob_buffer_mb,
        )

        # Release training data
        del X_train, y_train
        gc.collect()

        # Phase 2: Reload data for val/test evaluation
        log("\nReloading data for evaluation...")
        features, labels, _ = load_dataset(data_path)

        # Evaluate using disk-saved trees (soft voting)
        tree_dir = results['tree_dir']
        n_est = results['n_estimators']
        n_classes = results['n_classes']
        results['rf_eval_strategy'] = args.rf_eval_strategy
        results['rf_tree_first_max_prob_mb'] = args.rf_tree_first_max_prob_mb
        results['rf_tree_prefetch'] = args.rf_tree_prefetch
        results['rf_tree_eval_workers'] = args.rf_tree_eval_workers
        results['rf_log_each_tree_time'] = args.rf_log_each_tree_time
        results['rf_combine_val_test'] = args.rf_combine_val_test
        results['rf_val_trees_per_batch'] = args.rf_val_trees_per_batch
        results['rf_test_trees_per_batch'] = args.rf_test_trees_per_batch
        eval_results = evaluate_saved_forest_splits(
            features,
            labels,
            val_idx=val_idx,
            test_idx=test_idx,
            tree_dir=tree_dir,
            n_estimators=n_est,
            n_classes=n_classes,
            threshold=config.prediction_threshold,
            eval_batch_size=args.rf_eval_batch_size,
            prob_buffer_mb=args.rf_eval_prob_buffer_mb,
            val_trees_per_batch=args.rf_val_trees_per_batch,
            test_trees_per_batch=args.rf_test_trees_per_batch,
            eval_strategy=args.rf_eval_strategy,
            tree_first_max_prob_mb=args.rf_tree_first_max_prob_mb,
            tree_prefetch=args.rf_tree_prefetch,
            tree_eval_workers=args.rf_tree_eval_workers,
            log_each_tree_time=args.rf_log_each_tree_time,
            combine_val_test=args.rf_combine_val_test,
            label_map=label_map,
            logger=log,
        )
        results.update(eval_results)
        del features, labels, val_idx, test_idx
        gc.collect()

        return results

    else:
        # --- NN / Deep branch ---
        # Create dataloaders with 8:1:1 split (train:val:test)
        train_loader, val_loader, test_loader, norm_params = create_dataloaders(
            features, labels,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        input_dim = features.shape[1]

        # Create model
        if args.model_type == 'nn':
            model = AppScannerNN(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        elif args.model_type == 'deep':
            model = AppScannerDeep(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dim=config.hidden_dims[0],
                num_layers=4,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        log(f"\nModel: {args.model_type}")
        log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        model, history = train(
            model, train_loader, val_loader, config,
            save_dir=config.output_dir,
        )

        # Test
        device = torch.device(config.device)
        metrics = test(
            model, test_loader, device,
            prediction_threshold=config.prediction_threshold,
            label_map=label_map,
        )

        # Save final model and metadata
        final_path = os.path.join(config.output_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'label_map': label_map,
            'norm_params': norm_params,
            'history': history,
            'metrics': {
                'accuracy': metrics.accuracy,
                'f1': metrics.f1,
                'confidence_accuracy': metrics.confidence_accuracy,
            },
        }, final_path)
        log(f"\nModel saved to {final_path}")

        # Save training history
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(config.output_dir, f'history_{time}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        log(f"Training history saved to {history_path}")

        return model, metrics


def mode_eval(args, config):
    """Evaluation mode."""
    log("=" * 60)
    log("AppScanner Evaluation")
    log("=" * 60)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(config.output_dir, 'best_model.pth')

    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)

    # Load config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.num_classes = saved_config.num_classes
        config.hidden_dims = saved_config.hidden_dims

    # Load data
    features, labels, label_map = load_data(args, config)

    # Create test loader (use all data for testing)
    _, _, test_loader, _ = create_dataloaders(
        features, labels,
        batch_size=config.batch_size,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        num_workers=config.num_workers,
    )

    # Create and load model
    model = AppScannerNN(
        input_dim=features.shape[1],
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    device = torch.device(config.device)
    metrics = test(
        model, test_loader, device,
        prediction_threshold=config.prediction_threshold,
        label_map=label_map,
    )

    return metrics


def mode_extract(args, config):
    """Feature extraction mode."""
    log("=" * 60)
    log("AppScanner Feature Extraction")
    log("=" * 60)

    # Extract features
    features, labels, label_map = create_dataset_from_directory(
        args.data_dir,
        min_flow_length=config.min_flow_length,
        max_flow_length=config.max_flow_length,
    )

    # Save
    output_path = args.features_path or os.path.join(config.output_dir, 'features.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dataset(features, labels, label_map, output_path)

    log(f"Features shape: {features.shape}")
    log(f"Number of classes: {len(label_map)}")
    log(f"Saved to: {output_path}")

    return features, labels, label_map


def mode_compare(args, config):
    """Compare different approaches."""
    log("=" * 60)
    log("AppScanner Approach Comparison")
    log("=" * 60)

    # Load data
    features, labels, label_map = load_data(args, config)

    # Split data
    np.random.seed(config.seed)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * config.test_ratio)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    log(f"Train samples: {len(y_train)}")
    log(f"Test samples: {len(y_test)}")

    # Compare approaches
    results = compare_approaches(
        X_train, y_train, X_test, y_test, config
    )

    # Save results
    results_path = os.path.join(config.output_dir, 'comparison_results.json')
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")

    return results


def main():
    # 记录开始时间
    start_time = datetime.now()

    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Create base config
    config = create_config_from_args(args)
    base_output_dir = config.output_dir

    log("=" * 70)
    log("AppScanner Multi-Dataset Training")
    log("=" * 70)
    log(f"Datasets to run: {len(args.features_paths)}")
    log(f"Model: {args.model_type}")
    log()

    for i, data_path in enumerate(args.features_paths, 1):
        dataset_start = datetime.now()
        dataset_name = Path(data_path).stem.replace('_appscanner', '')

        # Set current dataset
        args.features_path = data_path
        args.num_classes = None  # Reset for auto-detect
        config.output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup per-dataset logging
        log_path = setup_logging(config.output_dir)

        log(f"\n{'=' * 70}")
        log(f"[{i}/{len(args.features_paths)}] Dataset: {dataset_name}")
        log(f"{'=' * 70}")
        log(f"\nConfiguration:")
        log(f"  Start time: {dataset_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"  Mode: {args.mode}")
        log(f"  Model: {args.model_type}")
        log(f"  Data: {data_path}")
        log(f"  Device: {config.device}")
        log(f"  Prediction threshold: {config.prediction_threshold}")
        if args.model_type == 'rf':
            log(f"  RF max_depth: {args.rf_max_depth}")
            log(f"  RF train trees_per_batch: {args.rf_trees_per_batch}")
            log(f"  RF val trees_per_batch: {args.rf_val_trees_per_batch}")
            log(f"  RF test trees_per_batch: {args.rf_test_trees_per_batch}")
            log(f"  RF eval batch_size: {args.rf_eval_batch_size}")
            log(f"  RF eval prob_buffer_mb: {args.rf_eval_prob_buffer_mb}")
            log(f"  RF eval strategy: {args.rf_eval_strategy}")
            log(f"  RF tree_first max_prob_mb: {args.rf_tree_first_max_prob_mb}")
            log(f"  RF tree prefetch: {args.rf_tree_prefetch}")
            log(f"  RF tree eval workers: {args.rf_tree_eval_workers}")
            log(f"  RF log each tree time: {args.rf_log_each_tree_time}")
            log(f"  RF combine val+test: {args.rf_combine_val_test}")
            log(f"  RF progress_tree_step: {args.rf_progress_tree_step}")
        log(f"  Log file: {log_path}")
        log()

        try:
            # Run mode
            if args.mode == 'train':
                mode_train(args, config)
            elif args.mode == 'eval':
                mode_eval(args, config)
            elif args.mode == 'extract':
                mode_extract(args, config)
            elif args.mode == 'compare':
                mode_compare(args, config)

            dataset_end = datetime.now()
            elapsed = dataset_end - dataset_start
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            log(f"\n[{dataset_name}] Completed in {h:02d}:{m:02d}:{s:02d}")

        except Exception as e:
            log(f"\n[{dataset_name}] FAILED: {e}")
            import traceback
            log(traceback.format_exc())

    # 总计用时
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log()
    log("=" * 70)
    log(f"All {len(args.features_paths)} datasets completed.")
    log(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    log("=" * 70)

if __name__ == '__main__':
    main()
