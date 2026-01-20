#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC 微调脚本

加载预训练权重进行有监督分类训练。

Usage:
    # 使用预训练模型
    python finetune.py --pretrained ../checkpoints/pretrained.pth

    # 从头训练（不使用预训练）
    python finetune.py

    # 使用 NPZ 格式数据
    python finetune.py --npz --pretrained ../checkpoints/pretrained.pth
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import DEFAULT_FINETUNE_CONFIG
from models import traformer_yatc
from data import build_finetune_dataloader, build_npz_finetune_dataloader, build_split_dataloader
from engine import (
    train_one_epoch,
    evaluate,
    get_param_groups_with_layer_decay,
    load_pretrained_weights,
    save_checkpoint,
)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path) -> str:
    """Setup logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = output_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return str(log_path)


def log(message: str = ""):
    """Log message to both console and file."""
    logging.info(message)


# =============================================================================
# 配置
# =============================================================================

# 数据路径
DATA_PATH = Path(__file__).parent.parent / "data" / "vpn_unified_output_split"

# 是否使用已划分的目录结构 (train/val/test 子目录)
USE_SPLIT_DIR = False
TRAIN_PATH = None  # 由 finetune_all.py 设置
VAL_PATH = None
TEST_PATH = None

# 预训练模型路径 (None 表示从头训练)
PRETRAINED_PATH = Path(__file__).parent.parent / "checkpoints" / "pretrained.pth"

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "output" / DATA_PATH.name.replace('_split', '')

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 0.05
EPOCHS = 100
WARMUP_EPOCHS = 5
LAYER_DECAY = 0.65
DROP_PATH_RATE = 0.1
LABEL_SMOOTHING = 0.1

# 数据划分 (仅在 USE_SPLIT_DIR=False 时使用)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_SAMPLES = 10  # 每个类别最小样本数

# DataLoader 参数
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

# 保存频率
SAVE_FREQ_EPOCHS = 10
PRINT_FREQ = 100

# 数据格式
USE_NPZ = True  # True: NPZ 格式, False: PNG 图像

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 微调
# =============================================================================

def main():
    global OUTPUT_DIR
    OUTPUT_DIR = Path(__file__).parent.parent / "output" / DATA_PATH.name.replace('_split', '')
    # 记录开始时间
    start_time = datetime.now()

    # Setup logging
    log_path = setup_logging(OUTPUT_DIR)

    device = torch.device(DEVICE)

    # Enable cuDNN optimization
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Create dataloaders
    if USE_SPLIT_DIR and TRAIN_PATH and VAL_PATH and TEST_PATH:
        # 使用已划分的目录结构
        train_loader, num_classes = build_split_dataloader(
            TRAIN_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True
        )
        val_loader, _ = build_split_dataloader(
            VAL_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        test_loader, _ = build_split_dataloader(
            TEST_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        id2label = test_loader.dataset.id2label
    elif USE_NPZ:
        # 运行时划分
        train_loader, num_classes = build_npz_finetune_dataloader(
            DATA_PATH,
            split='train',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            min_samples=MIN_SAMPLES
        )
        val_loader, _ = build_npz_finetune_dataloader(
            DATA_PATH,
            split='val',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            min_samples=MIN_SAMPLES
        )
        test_loader, _ = build_npz_finetune_dataloader(
            DATA_PATH,
            split='test',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            min_samples=MIN_SAMPLES
        )
        id2label = test_loader.dataset.id2label
    else:
        train_loader = build_finetune_dataloader(
            DATA_PATH,
            split='train',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True
        )
        test_loader = build_finetune_dataloader(
            DATA_PATH,
            split='test',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        val_loader = None
        num_classes = len(train_loader.dataset.classes)
        id2label = {i: name for i, name in enumerate(train_loader.dataset.classes)}

    log("=" * 60)
    log("YaTC Fine-tuning")
    log("=" * 60)
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Data path: {DATA_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Device: {DEVICE}")
    log(f"Data format: {'NPZ' if USE_NPZ else 'PNG'}")
    log(f"Log file: {log_path}")
    log()
    log("Dataset:")
    log(f"  Num classes:   {num_classes}")
    log(f"  Train samples: {len(train_loader.dataset)}")
    if val_loader:
        log(f"  Val samples:   {len(val_loader.dataset)}")
    log(f"  Test samples:  {len(test_loader.dataset)}")
    log()
    log("Training Configuration:")
    log(f"  Batch size:      {BATCH_SIZE}")
    log(f"  Learning rate:   {LEARNING_RATE}")
    log(f"  Weight decay:    {WEIGHT_DECAY}")
    log(f"  Epochs:          {EPOCHS}")
    log(f"  Warmup epochs:   {WARMUP_EPOCHS}")
    log(f"  Layer decay:     {LAYER_DECAY}")
    log(f"  Drop path rate:  {DROP_PATH_RATE}")
    log(f"  Label smoothing: {LABEL_SMOOTHING}")
    log("=" * 60)

    # Create model
    model = traformer_yatc(
        num_classes=num_classes,
        drop_path_rate=DROP_PATH_RATE
    )

    # Load pre-trained weights
    if PRETRAINED_PATH and PRETRAINED_PATH.exists():
        model = load_pretrained_weights(model, str(PRETRAINED_PATH))
        log(f"Loaded pre-trained weights from: {PRETRAINED_PATH}")
    else:
        log("Training from scratch (no pre-trained weights)")

    model = model.to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with layer-wise learning rate decay
    param_groups = get_param_groups_with_layer_decay(
        model=model,
        base_lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        layer_decay=LAYER_DECAY,
        num_layers=4
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999)
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
    }

    # Training loop
    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / "yatc_best.pth"
    eval_loader = val_loader if val_loader is not None else test_loader

    log(f"\nStarting training for {EPOCHS} epochs...")
    log("-" * 60)

    for epoch in range(EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            base_lr=LEARNING_RATE,
            print_freq=PRINT_FREQ
        )

        # Evaluate
        val_metrics = evaluate(
            model=model,
            data_loader=eval_loader,
            device=device,
            num_classes=num_classes
        )

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        marker = ""
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
            marker = " *"

        log(f"Epoch {epoch+1:3d} | Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}{marker}")

        # Save periodic checkpoint
        if (epoch + 1) % SAVE_FREQ_EPOCHS == 0:
            checkpoint_path = OUTPUT_DIR / f'finetune_epoch{epoch+1:04d}.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=0,
                loss=train_metrics['loss'],
                output_path=checkpoint_path
            )

    # Final test evaluation
    log("\n" + "=" * 60)
    log("Test Results")
    log("=" * 60)

    # Load best model
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

    log(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    log(f"Test Precision: {test_metrics['precision']:.4f}")
    log(f"Test Recall:    {test_metrics['recall']:.4f}")
    log(f"Test F1 Score:  {test_metrics['f1']:.4f}")

    # Get predictions for classification report
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

    # Print classification report
    log("\n" + "-" * 60)
    log("Classification Report:")
    log("-" * 60)
    labels_list = list(range(num_classes))
    target_names = [id2label[i] for i in labels_list]
    log(classification_report(all_labels, all_preds, labels=labels_list,
                              target_names=target_names, zero_division=0))

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = OUTPUT_DIR / f"finetune_history_{timestamp}.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log(f"Training history saved to: {history_path}")

    # Save final model with metadata
    final_model_path = OUTPUT_DIR / "yatc_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'id2label': id2label,
        'history': history,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
        },
    }, final_model_path)
    log(f"Final model saved to: {final_model_path}")

    # 记录结束时间并计算用时
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    log()
    log(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == '__main__':
    main()
