"""
YaTC Training Script

Main entry point for training and fine-tuning YaTC models.

Usage:
    Pre-training:
        python train.py pretrain --batch_size 128 --lr 1e-3 --steps 150000 --mask_ratio 0.9

    Fine-tuning:
        python train.py finetune --data_path ./data/ISCXVPN2016_MFR --num_classes 7 \
            --pretrained ./output_dir/pretrained.pth --epochs 200 --lr 2e-3

All hyperparameters are consistent with the paper.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEFAULT_PRETRAIN_CONFIG,
    DEFAULT_FINETUNE_CONFIG,
    DEFAULT_DATASET_CONFIG,
)
from models import mae_yatc, traformer_yatc
from data import (
    build_pretrain_dataloader,
    build_finetune_dataloader,
    build_npz_pretrain_dataloader,
    build_npz_finetune_dataloader
)
from engine import (
    pretrain_one_epoch,
    train_one_epoch,
    evaluate,
    get_param_groups_with_layer_decay,
    load_pretrained_weights,
    save_checkpoint,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YaTC Training Script')
    subparsers = parser.add_subparsers(dest='mode', help='Training mode')

    # Pre-training arguments
    pretrain_parser = subparsers.add_parser('pretrain', help='MAE pre-training')
    pretrain_parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to pre-training data'
    )
    pretrain_parser.add_argument(
        '--output_dir', type=str, default='./output_dir',
        help='Output directory for checkpoints'
    )
    pretrain_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size (default: 128)'
    )
    pretrain_parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Base learning rate (default: 1e-3)'
    )
    pretrain_parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay (default: 0.05)'
    )
    pretrain_parser.add_argument(
        '--steps', type=int, default=150000,
        help='Total training steps (default: 150000)'
    )
    pretrain_parser.add_argument(
        '--warmup_steps', type=int, default=10000,
        help='Warmup steps (default: 10000)'
    )
    pretrain_parser.add_argument(
        '--mask_ratio', type=float, default=0.9,
        help='Mask ratio (default: 0.9)'
    )
    pretrain_parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (default: cuda)'
    )
    pretrain_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    pretrain_parser.add_argument(
        '--save_freq', type=int, default=10000,
        help='Checkpoint save frequency in steps (default: 10000)'
    )
    pretrain_parser.add_argument(
        '--print_freq', type=int, default=100,
        help='Print frequency (default: 100)'
    )
    pretrain_parser.add_argument(
        '--npz', action='store_true',
        help='Use NPZ format data instead of PNG images'
    )

    # Fine-tuning arguments
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tuning')
    finetune_parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to fine-tuning data'
    )
    finetune_parser.add_argument(
        '--output_dir', type=str, default='./output_dir',
        help='Output directory for checkpoints'
    )
    finetune_parser.add_argument(
        '--pretrained', type=str, default='./output_dir/pretrained.pth',
        help='Path to pre-trained model checkpoint'
    )
    finetune_parser.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of classes (auto-detected if using --npz)'
    )
    finetune_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size (default: 128)'
    )
    finetune_parser.add_argument(
        '--lr', type=float, default=2e-3,
        help='Base learning rate (default: 2e-3)'
    )
    finetune_parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay (default: 0.05)'
    )
    finetune_parser.add_argument(
        '--epochs', type=int, default=200,
        help='Total training epochs (default: 200)'
    )
    finetune_parser.add_argument(
        '--warmup_epochs', type=int, default=5,
        help='Warmup epochs (default: 5)'
    )
    finetune_parser.add_argument(
        '--layer_decay', type=float, default=0.65,
        help='Layer-wise learning rate decay (default: 0.65)'
    )
    finetune_parser.add_argument(
        '--drop_path', type=float, default=0.1,
        help='Drop path rate (default: 0.1)'
    )
    finetune_parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing (default: 0.1)'
    )
    finetune_parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (default: cuda)'
    )
    finetune_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    finetune_parser.add_argument(
        '--save_freq', type=int, default=10,
        help='Checkpoint save frequency in epochs (default: 10)'
    )
    finetune_parser.add_argument(
        '--print_freq', type=int, default=100,
        help='Print frequency (default: 100)'
    )
    finetune_parser.add_argument(
        '--npz', action='store_true',
        help='Use NPZ format data instead of PNG images'
    )
    finetune_parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Train split ratio for NPZ data (default: 0.8)'
    )
    finetune_parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Validation split ratio for NPZ data (default: 0.1)'
    )

    # Evaluation arguments
    eval_parser = subparsers.add_parser('eval', help='Evaluation')
    eval_parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to evaluation data'
    )
    eval_parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    eval_parser.add_argument(
        '--num_classes', type=int, required=True,
        help='Number of classes'
    )
    eval_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size (default: 128)'
    )
    eval_parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (default: cuda)'
    )
    eval_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    eval_parser.add_argument(
        '--npz', action='store_true',
        help='Use NPZ format data instead of PNG images'
    )

    return parser.parse_args()


def pretrain(args):
    """Run MAE pre-training."""
    print("=" * 50)
    print("YaTC Pre-training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total steps: {args.steps}")
    print(f"Mask ratio: {args.mask_ratio}")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = mae_yatc()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Create dataloader
    if args.npz:
        dataloader = build_npz_pretrain_dataloader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    else:
        dataloader = build_pretrain_dataloader(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )

    # Training loop
    steps_per_epoch = len(dataloader)
    num_epochs = (args.steps + steps_per_epoch - 1) // steps_per_epoch

    step = 0
    for epoch in range(num_epochs):
        metrics = pretrain_one_epoch(
            model=model,
            data_loader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            step_offset=step,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            mask_ratio=args.mask_ratio,
            print_freq=args.print_freq
        )

        step += steps_per_epoch

        # Save checkpoint
        if (epoch + 1) % (args.save_freq // steps_per_epoch + 1) == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                loss=metrics['loss'],
                output_path=os.path.join(
                    args.output_dir,
                    f'pretrain_epoch{epoch:04d}.pth'
                )
            )

        if step >= args.steps:
            break

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        loss=metrics['loss'],
        output_path=os.path.join(args.output_dir, 'pretrained.pth')
    )

    print("Pre-training completed!")


def finetune(args):
    """Run fine-tuning."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create dataloaders (do this first to get num_classes for NPZ)
    if args.npz:
        train_loader, num_classes = build_npz_finetune_dataloader(
            args.data_path,
            split='train',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        val_loader, _ = build_npz_finetune_dataloader(
            args.data_path,
            split='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        test_loader, _ = build_npz_finetune_dataloader(
            args.data_path,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        if args.num_classes is None:
            args.num_classes = num_classes
    else:
        if args.num_classes is None:
            raise ValueError("--num_classes is required when not using --npz")
        train_loader = build_finetune_dataloader(
            args.data_path,
            split='train',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        test_loader = build_finetune_dataloader(
            args.data_path,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
        val_loader = None

    print("=" * 50)
    print("YaTC Fine-tuning")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Pre-trained: {args.pretrained}")
    print(f"Num classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Layer decay: {args.layer_decay}")
    print(f"Data format: {'NPZ' if args.npz else 'PNG'}")
    print(f"Using device: {device}")
    print("=" * 50)

    # Create model
    model = traformer_yatc(
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path
    )

    # Load pre-trained weights
    if os.path.exists(args.pretrained):
        model = load_pretrained_weights(model, args.pretrained)
    else:
        print(f"Warning: Pre-trained checkpoint not found at {args.pretrained}")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with layer-wise learning rate decay
    param_groups = get_param_groups_with_layer_decay(
        model=model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
        num_layers=4  # Encoder has 4 layers
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999)
    )

    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    # Training loop
    best_acc = 0.0
    eval_loader = val_loader if val_loader is not None else test_loader

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr,
            print_freq=args.print_freq
        )

        # Evaluate on val (or test if no val)
        eval_metrics = evaluate(
            model=model,
            data_loader=eval_loader,
            device=device,
            num_classes=args.num_classes
        )

        # Save best model
        if eval_metrics['accuracy'] > best_acc:
            best_acc = eval_metrics['accuracy']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=0,
                loss=train_metrics['loss'],
                output_path=os.path.join(args.output_dir, 'best.pth')
            )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=0,
                loss=train_metrics['loss'],
                output_path=os.path.join(
                    args.output_dir,
                    f'finetune_epoch{epoch:04d}.pth'
                )
            )

    # Final test evaluation
    if val_loader is not None:
        print("\nFinal Test Evaluation:")
        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=args.num_classes
        )
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    print(f"Fine-tuning completed! Best val accuracy: {best_acc:.4f}")


def eval_model(args):
    """Run evaluation."""
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create dataloader (do this first to get num_classes for NPZ)
    if args.npz:
        test_loader, num_classes = build_npz_finetune_dataloader(
            args.data_path,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
        if args.num_classes is None:
            args.num_classes = num_classes
    else:
        if args.num_classes is None:
            raise ValueError("--num_classes is required when not using --npz")
        test_loader = build_finetune_dataloader(
            args.data_path,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )

    print("=" * 50)
    print("YaTC Evaluation")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num classes: {args.num_classes}")
    print(f"Data format: {'NPZ' if args.npz else 'PNG'}")
    print(f"Using device: {device}")
    print("=" * 50)

    # Create model
    model = traformer_yatc(num_classes=args.num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Evaluate
    metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=args.num_classes
    )

    print(f"\nFinal Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode == 'pretrain':
        pretrain(args)
    elif args.mode == 'finetune':
        finetune(args)
    elif args.mode == 'eval':
        eval_model(args)
    else:
        print("Please specify a mode: pretrain, finetune, or eval")
        print("Use --help for more information")


if __name__ == '__main__':
    main()
