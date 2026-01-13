"""
FS-Net Training Script

Paper: FS-Net: A Flow Sequence Network For Encrypted Traffic Classification
Conference: INFOCOM 2019

Usage:
    python train_with_dataset.py
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

from models import create_fsnet, create_fsnet_nd
from data import load_pickle_dataset, create_dataloaders
from engine import train_one_epoch, evaluate, save_checkpoint, load_checkpoint


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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainArgs:
    """Training arguments - modify these directly instead of command line."""

    # Mode: 'train', 'eval'
    mode: str = 'train'

    # Data paths
    data_path: str = ''
    # data_path: str = './data/iscxvpn/iscxvpn_fsnet.pkl'
    # data_path: str = './data/cic_iot_2022/cic_iot_2022_fsnet.pkl'  # Pre-extracted features
    # data_path: str = './data/cross_platform/cross_platform_fsnet.pkl'  # Pre-extracted features
    # data_path: str = './data/iscxtor/iscxtor_fsnet.pkl'  # Pre-extracted features
    # data_path: str = './data/ustc/ustc_fsnet.pkl'  # Pre-extracted features
    # data_path: str = './data/novpn/novpn_fsnet.pkl'  # Pre-extracted features
    # data_path: str = './data/vpn/vpn_fsnet.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/novpn_top10/novpn_top10_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/vpn_top10/vpn_top10_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/novpn_top50/novpn_top50_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/vpn_top50/vpn_top50_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/novpn_top100/novpn_top100_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/vpn_top100/vpn_top100_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/novpn_top500/novpn_top500_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/vpn_top500/vpn_top500_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/novpn_top1000/novpn_top1000_appscanner.pkl'  # Pre-extracted features
    # data_path: str = '/root/autodl-tmp/FS-Net/vpn_top1000/vpn_top1000_appscanner.pkl'  # Pre-extracted features

    # Model configuration
    model_type: str = 'fsnet'               # 'fsnet' or 'fsnet_nd' (no decoder)
    num_classes: Optional[int] = None       # Auto-detect from data
    embed_dim: int = 128                    # Embedding dimension (paper: 128)
    hidden_dim: int = 128                   # Hidden dimension (paper: 128)
    num_layers: int = 2                     # Number of GRU layers (paper: 2)
    dropout: float = 0.3                    # Dropout rate (paper: 0.3)
    alpha: float = 1.0                      # Reconstruction loss weight (paper: 1)

    # Training parameters
    epochs: int = 100
    batch_size: int = 128                   # 论文现实的代码中的数据值
    lr: float = 0.0005                      # Learning rate (paper: 0.0005)
    patience: int = 20                      # Early stopping patience

    # Dataset split ratio (8:1:1)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Paths
    output_dir: str = './output'
    checkpoint: Optional[str] = None

    # Device: 'auto', 'cuda', 'cpu'
    device: str = 'auto'

    # Misc
    seed: int = 42
    num_workers: int = 8                    # 多进程数据加载 (服务器环境)
    use_class_weight: bool = False          # Use class weights for imbalanced data
    prefetch_factor: int = 4                # 每个 worker 预取的 batch 数
    persistent_workers: bool = True         # 保持 worker 进程存活，避免重复创建开销


def get_args() -> TrainArgs:
    """Get training arguments with optional command line override for data_path."""
    import argparse

    parser = argparse.ArgumentParser(description='FS-Net Training Script')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to pickle file (overrides default)')

    args = parser.parse_args()

    # Create TrainArgs with defaults
    train_args = TrainArgs()

    # Override data_path if provided
    if args.data_path is not None:
        train_args.data_path = args.data_path

    return train_args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(args: TrainArgs):
    """Load data from pickle file."""
    log(f"Loading dataset from: {args.data_path}")
    sequences, labels, label_map = load_pickle_dataset(args.data_path)

    log(f"Total samples: {len(labels)}")
    log(f"Num classes: {len(label_map)}")

    return sequences, labels, label_map


# =============================================================================
# Training Mode
# =============================================================================

def mode_train(args: TrainArgs):
    """Training mode."""
    log("=" * 60)
    log("FS-Net Training")
    log("=" * 60)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log(f"Using device: {device}")

    # Enable cuDNN
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Load data
    sequences, labels, label_map = load_data(args)
    num_classes = args.num_classes if args.num_classes else len(label_map)

    # Print class distribution
    log("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        class_name = label_map.get(label_id, f"Unknown({label_id})")
        log(f"  [{label_id:2d}] {class_name:15s}: {count:6d} ({count/len(labels)*100:5.1f}%)")

    # Create dataloaders
    log(f"\nCreating dataloaders with {args.train_ratio}:{args.val_ratio}:{args.test_ratio} split...")
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences, labels,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        use_direction=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers
    )

    # Compute class weights
    class_weight = None
    if args.use_class_weight:
        total = len(labels)
        weights = []
        for i in range(num_classes):
            count = (labels == i).sum()
            weight = total / (num_classes * count) if count > 0 else 1.0
            weights.append(weight)
        class_weight = torch.tensor(weights, dtype=torch.float32)
        log(f"Class weights: {class_weight.tolist()}")

    # Create model
    if args.model_type == 'fsnet_nd':
        model = create_fsnet_nd(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        log("\nUsing FS-Net-ND (no decoder)")
    else:
        model = create_fsnet(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=args.alpha,
            class_weight=class_weight
        )
        log("\nUsing FS-Net")

    model = model.to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training configuration
    log("\nTraining Configuration:")
    log(f"  Data path:     {args.data_path}")
    log(f"  Num classes:   {num_classes}")
    log(f"  Batch size:    {args.batch_size}")
    log(f"  Learning rate: {args.lr}")
    log(f"  Epochs:        {args.epochs}")
    log(f"  Embed dim:     {args.embed_dim}")
    log(f"  Hidden dim:    {args.hidden_dim}")
    log(f"  Num layers:    {args.num_layers}")
    log(f"  Dropout:       {args.dropout}")
    log(f"  Alpha:         {args.alpha}")
    log("=" * 60)

    # Training loop
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
    }

    for epoch in range(1, args.epochs + 1):
        log(f"\nEpoch {epoch}/{args.epochs}")
        log("-" * 40)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        log(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, num_classes)
        log(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, TPR: {val_metrics['tpr_avg']:.4f}, "
            f"FPR: {val_metrics['fpr_avg']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            log(f"  -> Saved best model (F1: {best_f1:.4f})")

    # Final evaluation on test set
    log("\n" + "=" * 60)
    log("Final Evaluation on Test Set")
    log("=" * 60)

    # Load best model
    model, _, _ = load_checkpoint(
        model,
        os.path.join(args.output_dir, 'best_model.pth'),
        device=device
    )

    test_metrics = evaluate(model, test_loader, device, num_classes)

    # Print overall results
    log("Overall Results:")
    log(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    log(f"  Precision: {test_metrics['precision']:.4f}")
    log(f"  Recall:    {test_metrics['recall']:.4f}")
    log(f"  F1 Score:  {test_metrics['f1']:.4f}")
    log(f"  TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    log(f"  FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    # Print per-class results
    log("\n" + "-" * 80)
    log("Per-Class Results:")
    log("-" * 80)
    log(f"{'Class':<20} {'Count':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TPR':>10} {'FPR':>10}")
    log("-" * 80)

    for i in range(num_classes):
        class_name = label_map.get(i, f"Class_{i}")
        count = test_metrics['per_class_count'][i]
        precision = test_metrics['per_class_precision'][i]
        recall = test_metrics['per_class_recall'][i]
        f1 = test_metrics['per_class_f1'][i]
        tpr = test_metrics['per_class_tpr'][i]
        fpr = test_metrics['per_class_fpr'][i]
        log(f"{class_name:<20} {count:>8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tpr:>10.4f} {fpr:>10.4f}")

    log("-" * 80)

    # Save final model and history
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'label_map': label_map,
        'args': args,
        'history': history,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'tpr_avg': test_metrics['tpr_avg'],
            'fpr_avg': test_metrics['fpr_avg'],
        },
    }, final_path)
    log(f"\nModel saved to {final_path}")

    # Save history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(args.output_dir, f'history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log(f"Training history saved to {history_path}")

    return model, test_metrics


# =============================================================================
# Evaluation Mode
# =============================================================================

def mode_eval(args: TrainArgs):
    """Evaluation mode."""
    log("=" * 60)
    log("FS-Net Evaluation")
    log("=" * 60)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log(f"Using device: {device}")

    # Checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, 'best_model.pth')

    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load data
    sequences, labels, label_map = load_data(args)
    num_classes = len(label_map)

    # Create test loader (use all data for testing)
    _, _, test_loader = create_dataloaders(
        sequences, labels,
        batch_size=args.batch_size,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        num_workers=args.num_workers,
        use_direction=True
    )

    # Create and load model
    if args.model_type == 'fsnet_nd':
        model = create_fsnet_nd(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = create_fsnet(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=args.alpha
        )

    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    test_metrics = evaluate(model, test_loader, device, num_classes)

    # Print results
    log("\n" + "=" * 60)
    log("TEST RESULTS")
    log("=" * 60)
    log(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    log(f"Precision: {test_metrics['precision']:.4f}")
    log(f"Recall:    {test_metrics['recall']:.4f}")
    log(f"F1 Score:  {test_metrics['f1']:.4f}")
    log(f"TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    log(f"FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    return test_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    # 记录开始时间
    start_time = datetime.now()

    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Create dataset-specific output directory to avoid conflicts
    dataset_name = Path(args.data_path).stem.replace('_fsnet', '')
    args.output_dir = os.path.join(args.output_dir, dataset_name)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_path = setup_logging(args.output_dir)

    # Print configuration
    log("\nConfiguration:")
    log(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Mode: {args.mode}")
    log(f"  Model: {args.model_type}")
    log(f"  Data: {args.data_path}")
    log(f"  Device: {args.device}")
    log(f"  Epochs: {args.epochs}")
    log(f"  Batch size: {args.batch_size}")
    log(f"  Learning rate: {args.lr}")
    log(f"  Num classes: {args.num_classes if args.num_classes else 'auto'}")
    log(f"  Log file: {log_path}")
    log()

    # Run mode
    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'eval':
        mode_eval(args)

    # 记录结束时间并计算用时
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log()
    log(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == '__main__':
    main()
