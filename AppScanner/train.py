"""
AppScanner Main Training Script

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

Usage:
    # Train on PCAP directory
    python train.py --mode train --data_dir ./data/apps --num_classes 110

    # Train on pre-extracted features (CSV)
    python train.py --mode train --csv_path ./data/features.csv

    # Evaluate trained model
    python train.py --mode eval --checkpoint ./output/best_model.pth --data_dir ./data/apps

    # Extract features from PCAP
    python train.py --mode extract --data_dir ./data/apps --output ./data/features.pkl

    # Compare NN vs RF approaches
    python train.py --mode compare --data_dir ./data/apps
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
from datetime import datetime

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AppScanner: Fingerprinting Smartphone Apps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'eval', 'extract', 'compare'],
        help='Running mode',
    )

    # Data
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Directory containing PCAP files organized by class',
    )
    parser.add_argument(
        '--csv_path', type=str, default=None,
        help='Path to CSV file with pre-extracted features',
    )
    parser.add_argument(
        '--features_path', type=str, default=None,
        help='Path to saved features (pickle)',
    )

    # Model
    parser.add_argument(
        '--model_type', type=str, default='nn',
        choices=['nn', 'deep', 'rf'],
        help='Model type to use',
    )
    parser.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of classes (auto-detected if not specified)',
    )
    parser.add_argument(
        '--input_dim', type=int, default=54,
        help='Input feature dimension',
    )
    parser.add_argument(
        '--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
        help='Hidden layer dimensions',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.3,
        help='Dropout rate',
    )

    # Training
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size',
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay (L2 regularization)',
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience',
    )

    # AppScanner specific
    parser.add_argument(
        '--prediction_threshold', type=float, default=0.9,
        help='Confidence threshold for predictions',
    )
    parser.add_argument(
        '--min_flow_length', type=int, default=7,
        help='Minimum packets per flow',
    )
    parser.add_argument(
        '--max_flow_length', type=int, default=260,
        help='Maximum packets per flow',
    )

    # Random Forest (for comparison)
    parser.add_argument(
        '--n_estimators', type=int, default=100,
        help='Number of trees in Random Forest',
    )

    # Paths
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Output directory',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint for evaluation',
    )

    # Device
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use',
    )

    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers',
    )

    return parser.parse_args()


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
    checkpoint = torch.load(args.checkpoint, map_location=config.device)

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
    args = parse_args()

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
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
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
