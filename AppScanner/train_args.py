"""
Shared training/runtime arguments for AppScanner entry scripts.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class _TorchPlaceholder:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def manual_seed_all(seed: int) -> None:
                return None

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

    torch = _TorchPlaceholder()

from config import AppScannerConfig, get_config


RF_MEMORY_BUDGET_MB = 360 * 1024
RF_EVAL_PROB_BUFFER_MB = 512


def _largest_divisor_at_most(total: int, limit: int) -> int:
    """Return the largest divisor of total that does not exceed limit."""
    total = max(1, int(total))
    limit = max(1, min(int(limit), total))
    for candidate in range(limit, 0, -1):
        if total % candidate == 0:
            return candidate
    return 1


def _default_rf_fit_trees_per_batch(n_estimators: int) -> int:
    """Use an even tree batch near half the CPU count to avoid a weak tail batch."""
    cpu_count = max(1, os.cpu_count() or 1)
    limit = min(cpu_count, max(8, cpu_count // 2), n_estimators)
    return _largest_divisor_at_most(n_estimators, limit)


def _default_rf_eval_trees_per_batch(n_estimators: int) -> int:
    """Evaluate an even tree batch, up to all trees, for high CPU utilization."""
    cpu_count = max(1, os.cpu_count() or 1)
    limit = min(cpu_count, n_estimators)
    return _largest_divisor_at_most(n_estimators, limit)


def _default_rf_tree_first_workers(n_estimators: int) -> int:
    """Use an even worker count for high tree_first CPU utilization."""
    cpu_count = max(1, os.cpu_count() or 1)
    limit = min(cpu_count, n_estimators)
    return _largest_divisor_at_most(n_estimators, limit)


@dataclass
class TrainArgs:
    """Training arguments shared across training and resume-eval scripts."""

    # Mode: 'train', 'eval', 'extract', 'compare'
    mode: str = 'train'

    # Data paths
    data_dir: str = './data'
    csv_path: Optional[str] = None
    features_paths: List[str] = None
    features_path: Optional[str] = None

    # Model configuration
    model_type: str = 'rf'
    num_classes: Optional[int] = None
    input_dim: int = 54
    hidden_dims: List[int] = None
    dropout: float = 0.3

    # Training parameters
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10

    # AppScanner specific
    prediction_threshold: float = 0.9
    min_flow_length: int = 7
    max_flow_length: int = 260

    # Random Forest
    n_estimators: int = 100
    rf_max_depth: Optional[int] = 20
    rf_trees_per_batch: Optional[int] = None
    rf_val_trees_per_batch: Optional[int] = None
    rf_test_trees_per_batch: Optional[int] = None
    rf_eval_batch_size: Optional[int] = None
    rf_eval_prob_buffer_mb: int = RF_EVAL_PROB_BUFFER_MB
    rf_eval_strategy: str = 'auto'
    rf_tree_first_max_prob_mb: int = RF_MEMORY_BUDGET_MB
    rf_tree_prefetch: int = 1
    rf_tree_eval_workers: Optional[int] = None
    rf_log_each_tree_time: bool = True
    rf_combine_val_test: bool = True
    rf_progress_tree_step: int = 1
    rf_compute_feature_importance: bool = False

    # Paths
    output_dir: str = './output'
    checkpoint: Optional[str] = None

    # Device: 'auto', 'cuda', 'cpu'
    device: str = 'auto'

    # Misc
    seed: int = 42
    num_workers: int = 4

    def __post_init__(self):
        rf_fit_trees_per_batch = _default_rf_fit_trees_per_batch(self.n_estimators)
        rf_eval_trees_per_batch = _default_rf_eval_trees_per_batch(self.n_estimators)
        rf_tree_first_workers = _default_rf_tree_first_workers(self.n_estimators)
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        if self.rf_trees_per_batch is None:
            self.rf_trees_per_batch = rf_fit_trees_per_batch
        if self.rf_val_trees_per_batch is None:
            self.rf_val_trees_per_batch = rf_eval_trees_per_batch
        if self.rf_test_trees_per_batch is None:
            self.rf_test_trees_per_batch = rf_eval_trees_per_batch
        if self.rf_tree_eval_workers is None:
            self.rf_tree_eval_workers = rf_tree_first_workers
        if self.features_paths is None:
            self.features_paths = [
                '/home/pcz/code/DL/AppScanner/data/vpn/vpn_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn/novpn_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn_top10/novpn_top10_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/vpn_top10/vpn_top10_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn_top50/novpn_top50_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/vpn_top50/vpn_top50_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn_top100/novpn_top100_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/vpn_top100/vpn_top100_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn_top500/novpn_top500_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/vpn_top500/vpn_top500_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/novpn_top1000/novpn_top1000_appscanner.pkl',
                '/home/pcz/code/DL/AppScanner/data/vpn_top1000/vpn_top1000_appscanner.pkl',
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
    parser.add_argument(
        '--data_path',
        type=str,
        nargs='+',
        default=None,
        help='Path(s) to pickle file(s) (overrides default list)',
    )

    args = parser.parse_args()
    train_args = TrainArgs()
    if args.data_path is not None:
        train_args.features_paths = args.data_path
    return train_args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_config_from_args(args) -> AppScannerConfig:
    """Create configuration from shared runtime arguments."""
    config = get_config()

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

    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device

    return config
