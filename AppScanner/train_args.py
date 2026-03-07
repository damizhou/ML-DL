"""
Shared training/runtime arguments for AppScanner entry scripts.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from config import AppScannerConfig, get_config


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
    rf_trees_per_batch: int = 25
    rf_val_trees_per_batch: Optional[int] = 5
    rf_test_trees_per_batch: Optional[int] = 5
    rf_eval_batch_size: Optional[int] = None
    rf_eval_prob_buffer_mb: int = 256
    rf_eval_strategy: str = 'tree_first'
    rf_tree_first_max_prob_mb: int = 4096
    rf_tree_prefetch: int = 1
    rf_tree_eval_workers: int = 5
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
