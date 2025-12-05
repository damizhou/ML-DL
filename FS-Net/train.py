"""
FS-Net Training Script

Main entry point for training and evaluating FS-Net models.

Usage:
    Training:
        python train.py train --data_path ./data --num_classes 18

    Evaluation:
        python train.py eval --data_path ./data --checkpoint ./checkpoints/best.pth

    Convert PCAP to sequences:
        python train.py convert --input ./pcap_data --output ./seq_data

All hyperparameters follow the paper.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from config import (
    DEFAULT_FSNET_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_DATA_CONFIG,
    FSNetConfig,
    TrainConfig,
)
from models import FSNet, FSNetND, create_fsnet, create_fsnet_nd
from data import (
    FlowSequenceDataset,
    build_dataloader,
    pcap_to_sequences,
    get_dataset_info,
    collate_fn,
)
from engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FS-Net Training')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')

    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to dataset'
    )
    train_parser.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of classes (auto-detected if not specified)'
    )
    train_parser.add_argument(
        '--output_dir', type=str, default='./checkpoints',
        help='Output directory for checkpoints'
    )
    train_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size (default: 128)'
    )
    train_parser.add_argument(
        '--lr', type=float, default=0.0005,
        help='Learning rate (default: 0.0005)'
    )
    train_parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of epochs (default: 100)'
    )
    train_parser.add_argument(
        '--embed_dim', type=int, default=128,
        help='Embedding dimension (default: 128)'
    )
    train_parser.add_argument(
        '--hidden_dim', type=int, default=128,
        help='GRU hidden dimension (default: 128)'
    )
    train_parser.add_argument(
        '--num_layers', type=int, default=2,
        help='Number of GRU layers (default: 2)'
    )
    train_parser.add_argument(
        '--dropout', type=float, default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    train_parser.add_argument(
        '--alpha', type=float, default=1.0,
        help='Reconstruction loss weight (default: 1.0)'
    )
    train_parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (default: 10)'
    )
    train_parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (default: cuda)'
    )
    train_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    train_parser.add_argument(
        '--no_decoder', action='store_true',
        help='Use FS-Net-ND (no decoder) variant'
    )

    # Evaluation arguments
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to dataset'
    )
    eval_parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    eval_parser.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of classes'
    )
    eval_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size'
    )
    eval_parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device'
    )
    eval_parser.add_argument(
        '--no_decoder', action='store_true',
        help='Use FS-Net-ND variant'
    )

    # Convert PCAP arguments
    convert_parser = subparsers.add_parser('convert', help='Convert PCAP to sequences')
    convert_parser.add_argument(
        '--input', type=str, required=True,
        help='Input PCAP directory'
    )
    convert_parser.add_argument(
        '--output', type=str, required=True,
        help='Output sequence directory'
    )
    convert_parser.add_argument(
        '--max_seq_len', type=int, default=100,
        help='Maximum sequence length'
    )

    return parser.parse_args()


def train(args):
    """Train FS-Net model."""
    print("=" * 60)
    print("FS-Net Training")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataset info
    info = get_dataset_info(args.data_path)
    num_classes = args.num_classes or info['num_classes']
    print(f"Dataset: {info['num_classes']} classes, {info['total_samples']} samples")
    print(f"Classes: {info['classes']}")

    # Build data loaders
    train_loader = build_dataloader(
        args.data_path,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_loader = build_dataloader(
        args.data_path,
        split='val' if (Path(args.data_path) / 'val').exists() else 'test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    test_loader = build_dataloader(
        args.data_path,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Create model
    if args.no_decoder:
        model = create_fsnet_nd(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        print("Using FS-Net-ND (no decoder)")
    else:
        model = create_fsnet(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=args.alpha
        )
        print("Using FS-Net")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    best_f1 = 0.0

    print("\nTraining Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Alpha: {args.alpha}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, num_classes)
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, TPR: {val_metrics['tpr_avg']:.4f}, "
            f"FPR: {val_metrics['fpr_avg']:.4f}, FTF: {val_metrics['ftf']:.4f}"
        )

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(args.output_dir, 'best.pth')
            )

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(args.output_dir, f'epoch_{epoch}.pth')
            )

        # Early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    model, _, _ = load_checkpoint(
        model,
        os.path.join(args.output_dir, 'best.pth'),
        device=device
    )

    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Test Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    print(f"  FPR_AVE:   {test_metrics['fpr_avg']:.4f}")
    print(f"  FTF:       {test_metrics['ftf']:.4f}")


def eval_model(args):
    """Evaluate FS-Net model."""
    print("=" * 60)
    print("FS-Net Evaluation")
    print("=" * 60)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataset info
    info = get_dataset_info(args.data_path)
    num_classes = args.num_classes or info['num_classes']
    print(f"Dataset: {num_classes} classes")

    # Build test loader
    test_loader = build_dataloader(
        args.data_path,
        split='test',
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model
    if args.no_decoder:
        model = create_fsnet_nd(num_classes=num_classes)
    else:
        model = create_fsnet(num_classes=num_classes)

    # Load checkpoint
    model, epoch, _ = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    # Evaluate
    metrics = evaluate(model, test_loader, device, num_classes)

    print(f"\nTest Results (Epoch {epoch}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TPR_AVE:   {metrics['tpr_avg']:.4f}")
    print(f"  FPR_AVE:   {metrics['fpr_avg']:.4f}")
    print(f"  FTF:       {metrics['ftf']:.4f}")

    print("\nPer-class TPR:")
    for i, (cls_name, tpr) in enumerate(zip(info['classes'], metrics['per_class_tpr'])):
        print(f"  {cls_name}: {tpr:.4f}")


def convert(args):
    """Convert PCAP files to sequences."""
    from config import DataConfig
    config = DataConfig(max_seq_len=args.max_seq_len)
    pcap_to_sequences(args.input, args.output, config)


def main():
    args = parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_model(args)
    elif args.mode == 'convert':
        convert(args)
    else:
        print("Please specify a mode: train, eval, or convert")
        print("Use --help for more information")


if __name__ == '__main__':
    main()
