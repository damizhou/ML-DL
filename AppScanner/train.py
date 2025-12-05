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
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from config import AppScannerConfig, get_config
from models import AppScannerNN, AppScannerDeep, build_model
from data import (
    create_dataset_from_directory,
    create_dataset_from_csv,
    create_dataloaders,
    save_dataset,
    load_dataset,
)
from engine import train, test, train_random_forest, compare_approaches


# =============================================================================
# Configuration - Modify these parameters directly
# =============================================================================

@dataclass
class TrainArgs:
    """Training arguments - modify these directly instead of command line."""

    # Mode: 'train', 'eval', 'extract', 'compare'
    mode: str = 'train'

    # Data paths
    data_dir: str = './data'                    # Directory with PCAP files
    csv_path: Optional[str] = None              # CSV file with features
    features_path: str = '/home/pcz/DL/ML&DL/AppScanner/data/iscxvpn/iscxvpn_appscanner.pkl'  # Pre-extracted features

    # Model configuration
    model_type: str = 'nn'                      # 'nn', 'deep', or 'rf'
    num_classes: int = 12                       # ISCXVPN has 12 classes
    input_dim: int = 54                         # 54 statistical features
    hidden_dims: List[int] = None               # Default: [256, 128, 64]
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


def get_args() -> TrainArgs:
    """Get training arguments."""
    return TrainArgs()


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
        print(f"Loading pre-extracted features from {args.features_path}")
        features, labels, label_map = load_dataset(args.features_path)
    elif args.csv_path is not None:
        print(f"Loading features from CSV: {args.csv_path}")
        features, labels, label_map = create_dataset_from_csv(args.csv_path)
    else:
        print(f"Extracting features from PCAP files in {args.data_dir}")
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
    print("=" * 60)
    print("AppScanner Training")
    print("=" * 60)

    # Load data
    features, labels, label_map = load_data(args, config)
    print(f"Features shape: {features.shape}")
    print(f"Number of classes: {config.num_classes}")

    # Create dataloaders
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        features, labels,
        batch_size=config.batch_size,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    # Create model
    if args.model_type == 'nn':
        model = AppScannerNN(
            input_dim=features.shape[1],
            num_classes=config.num_classes,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
    elif args.model_type == 'deep':
        model = AppScannerDeep(
            input_dim=features.shape[1],
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dims[0],
            num_layers=4,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print(f"\nModel saved to {final_path}")

    # Save training history
    history_path = os.path.join(config.output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, metrics


def mode_eval(args, config):
    """Evaluation mode."""
    print("=" * 60)
    print("AppScanner Evaluation")
    print("=" * 60)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(config.output_dir, 'best_model.pth')

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)

    # Load config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.num_classes = saved_config.num_classes
        config.hidden_dims = saved_config.hidden_dims

    # Load data
    features, labels, label_map = load_data(args, config)

    # Create test loader
    _, _, test_loader, _ = create_dataloaders(
        features, labels,
        batch_size=config.batch_size,
        test_ratio=1.0,  # Use all data for testing
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
    print("=" * 60)
    print("AppScanner Feature Extraction")
    print("=" * 60)

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

    print(f"Features shape: {features.shape}")
    print(f"Number of classes: {len(label_map)}")
    print(f"Saved to: {output_path}")

    return features, labels, label_map


def mode_compare(args, config):
    """Compare different approaches."""
    print("=" * 60)
    print("AppScanner Approach Comparison")
    print("=" * 60)

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

    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # Compare approaches
    results = compare_approaches(
        X_train, y_train, X_test, y_test, config
    )

    # Save results
    results_path = os.path.join(config.output_dir, 'comparison_results.json')
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Create config
    config = create_config_from_args(args)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model_type}")
    print(f"  Data: {args.features_path or args.data_dir}")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Prediction threshold: {config.prediction_threshold}")
    print()

    # Run mode
    if args.mode == 'train':
        mode_train(args, config)
    elif args.mode == 'eval':
        mode_eval(args, config)
    elif args.mode == 'extract':
        mode_extract(args, config)
    elif args.mode == 'compare':
        mode_compare(args, config)


if __name__ == '__main__':
    main()
