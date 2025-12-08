#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC VPN 训练脚本

使用 unified_vpn_processor.py 生成的 NPZ 数据训练 YaTC
支持预训练 (MAE) 和微调两种模式

使用方法:
    # 预训练 (可选，如果有预训练模型可以跳过)
    python train_vpn.py pretrain

    # 微调 (使用预训练模型)
    python train_vpn.py finetune --pretrained ./checkpoints/pretrained.pth

    # 微调 (不使用预训练，从头训练)
    python train_vpn.py finetune
"""

import os
import sys
import argparse
from pathlib import Path

# Add refactor directory to path
REFACTOR_DIR = Path(__file__).parent / "refactor"
sys.path.insert(0, str(REFACTOR_DIR))

import numpy as np
import torch
from sklearn.metrics import classification_report

from models import mae_yatc, traformer_yatc
from data import build_npz_pretrain_dataloader, build_npz_finetune_dataloader
from engine import (
    pretrain_one_epoch,
    train_one_epoch,
    evaluate,
    get_param_groups_with_layer_decay,
    load_pretrained_weights,
    save_checkpoint,
)


# =============================================================================
# 配置
# =============================================================================

# 数据路径
DATA_ROOT = Path(__file__).parent / "data"
AVAILABLE_DATASETS = {
    "iscxvpn": DATA_ROOT / "iscxvpn",
    "vpn_unified": DATA_ROOT / "vpn_unified_output",
}
DEFAULT_DATASET = "iscxvpn"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "checkpoints"

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 预训练
# =============================================================================

def pretrain(args):
    """运行 MAE 预训练"""
    # 解析数据目录
    data_dir = AVAILABLE_DATASETS.get(args.data, AVAILABLE_DATASETS[DEFAULT_DATASET])

    print("=" * 60)
    print("YaTC VPN Pre-training (MAE)")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Data path: {data_dir}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Steps: {args.steps}")
    print(f"Mask ratio: {args.mask_ratio}")
    print("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    # 创建模型
    model = mae_yatc()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )

    # 创建数据加载器
    dataloader = build_npz_pretrain_dataloader(
        data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True
    )

    print(f"Dataset size: {len(dataloader.dataset)}")

    # 训练循环
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
            print_freq=100
        )

        step += steps_per_epoch

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                loss=metrics['loss'],
                output_path=OUTPUT_DIR / f'pretrain_epoch{epoch:04d}.pth'
            )

        if step >= args.steps:
            break

    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        loss=metrics['loss'],
        output_path=OUTPUT_DIR / 'pretrained.pth'
    )

    print(f"\nPre-training completed!")
    print(f"Model saved to: {OUTPUT_DIR / 'pretrained.pth'}")


# =============================================================================
# 微调
# =============================================================================

def finetune(args):
    """运行微调"""
    # 解析数据目录
    data_dir = AVAILABLE_DATASETS.get(args.data, AVAILABLE_DATASETS[DEFAULT_DATASET])

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    # 创建数据加载器（分层划分，剔除样本不足的类别）
    train_loader, num_classes = build_npz_finetune_dataloader(
        data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        train_ratio=0.8,
        val_ratio=0.1,
        min_samples=10
    )
    val_loader, _ = build_npz_finetune_dataloader(
        data_dir,
        split='val',
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        train_ratio=0.8,
        val_ratio=0.1,
        min_samples=10
    )
    test_loader, _ = build_npz_finetune_dataloader(
        data_dir,
        split='test',
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        train_ratio=0.8,
        val_ratio=0.1,
        min_samples=10
    )

    print("=" * 60)
    print("YaTC VPN Fine-tuning")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Data path: {data_dir}")
    print(f"Device: {DEVICE}")
    print(f"Num classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    if args.pretrained:
        print(f"Pre-trained: {args.pretrained}")
    print("=" * 60)

    # 创建模型
    model = traformer_yatc(
        num_classes=num_classes,
        drop_path_rate=0.1
    )

    # 加载预训练权重
    if args.pretrained and os.path.exists(args.pretrained):
        model = load_pretrained_weights(model, args.pretrained)
        print(f"Loaded pre-trained weights from {args.pretrained}")
    else:
        print("Training from scratch (no pre-trained weights)")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建优化器 (带逐层学习率衰减)
    param_groups = get_param_groups_with_layer_decay(
        model=model,
        base_lr=args.lr,
        weight_decay=0.05,
        layer_decay=0.65,
        num_layers=4
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练循环
    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "yatc_vpn_best.pth"

    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=5,
            base_lr=args.lr,
            print_freq=100
        )

        # 验证
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes
        )

        # 保存最佳模型
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

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

    # 测试
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    # 加载最佳模型
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

    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1 Score:  {test_metrics['f1']:.4f}")

    # 获取预测结果用于分类报告
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

    # 打印分类报告
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    id2label = test_loader.dataset.id2label
    labels_list = list(range(num_classes))
    target_names = [id2label[i] for i in labels_list]
    print(classification_report(all_labels, all_preds, labels=labels_list, target_names=target_names, zero_division=0))


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='YaTC VPN Training')
    subparsers = parser.add_subparsers(dest='mode', help='Training mode')

    # 数据集选项
    data_choices = list(AVAILABLE_DATASETS.keys())

    # 预训练参数
    pretrain_parser = subparsers.add_parser('pretrain', help='MAE pre-training')
    pretrain_parser.add_argument('--data', type=str, default=DEFAULT_DATASET,
                                  choices=data_choices,
                                  help=f'Dataset to use (default: {DEFAULT_DATASET})')
    pretrain_parser.add_argument('--batch_size', type=int, default=128)
    pretrain_parser.add_argument('--lr', type=float, default=1e-3)
    pretrain_parser.add_argument('--steps', type=int, default=50000)
    pretrain_parser.add_argument('--warmup_steps', type=int, default=5000)
    pretrain_parser.add_argument('--mask_ratio', type=float, default=0.9)

    # 微调参数
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tuning')
    finetune_parser.add_argument('--data', type=str, default=DEFAULT_DATASET,
                                  choices=data_choices,
                                  help=f'Dataset to use (default: {DEFAULT_DATASET})')
    finetune_parser.add_argument('--pretrained', type=str, default=None,
                                  help='Path to pre-trained model')
    finetune_parser.add_argument('--batch_size', type=int, default=128)
    finetune_parser.add_argument('--lr', type=float, default=2e-3)
    finetune_parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    if args.mode == 'pretrain':
        pretrain(args)
    elif args.mode == 'finetune':
        finetune(args)
    else:
        print("Usage:")
        print("  python train_vpn.py pretrain [--data iscxvpn|vpn_unified] [--steps 50000]")
        print("  python train_vpn.py finetune [--data iscxvpn|vpn_unified] [--pretrained model.pth]")
        print("")
        print("Available datasets:")
        for name, path in AVAILABLE_DATASETS.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
