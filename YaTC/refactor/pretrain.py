#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC 预训练脚本 (MAE)

使用掩码自编码器进行自监督预训练。

Usage:
    python pretrain.py --data_path ./data/merged_pretrain --steps 150000

    # 使用 NPZ 格式数据
    python pretrain.py --data_path ./data/merged_pretrain --npz --steps 150000
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import torch

from config import DEFAULT_PRETRAIN_CONFIG
from models import mae_yatc
from data import build_pretrain_dataloader, build_npz_pretrain_dataloader
from engine import pretrain_one_epoch, save_checkpoint


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path) -> str:
    """Setup logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
DATA_PATH = Path(__file__).parent.parent / "data" / "merged_pretrain"

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "checkpoints"

# 训练参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
TOTAL_STEPS = 150000
WARMUP_STEPS = 10000
MASK_RATIO = 0.9

# DataLoader 参数
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

# 保存频率
SAVE_FREQ_STEPS = 10000
PRINT_FREQ = 100

# 数据格式
USE_NPZ = True  # True: NPZ 格式, False: PNG 图像

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 预训练
# =============================================================================

def main():
    # Setup logging
    log_path = setup_logging(OUTPUT_DIR)

    log("=" * 60)
    log("YaTC Pre-training (MAE)")
    log("=" * 60)
    log(f"Data path: {DATA_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Device: {DEVICE}")
    log(f"Data format: {'NPZ' if USE_NPZ else 'PNG'}")
    log(f"Log file: {log_path}")
    log()
    log("Training Configuration:")
    log(f"  Batch size:    {BATCH_SIZE}")
    log(f"  Learning rate: {LEARNING_RATE}")
    log(f"  Weight decay:  {WEIGHT_DECAY}")
    log(f"  Total steps:   {TOTAL_STEPS}")
    log(f"  Warmup steps:  {WARMUP_STEPS}")
    log(f"  Mask ratio:    {MASK_RATIO}")
    log("=" * 60)

    device = torch.device(DEVICE)

    # Enable cuDNN optimization
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Create model
    model = mae_yatc()
    model = model.to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    # Create dataloader
    if USE_NPZ:
        dataloader = build_npz_pretrain_dataloader(
            DATA_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True
        )
    else:
        dataloader = build_pretrain_dataloader(
            DATA_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True
        )

    log(f"Dataset size: {len(dataloader.dataset)}")
    log(f"Steps per epoch: {len(dataloader)}")

    # Training history
    history = {
        'step': [],
        'loss': [],
        'lr': [],
    }

    # Training loop
    steps_per_epoch = len(dataloader)
    num_epochs = (TOTAL_STEPS + steps_per_epoch - 1) // steps_per_epoch

    log(f"\nStarting training for {num_epochs} epochs ({TOTAL_STEPS} steps)...")
    log("-" * 60)

    step = 0
    for epoch in range(num_epochs):
        metrics = pretrain_one_epoch(
            model=model,
            data_loader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            step_offset=step,
            total_steps=TOTAL_STEPS,
            warmup_steps=WARMUP_STEPS,
            base_lr=LEARNING_RATE,
            mask_ratio=MASK_RATIO,
            print_freq=PRINT_FREQ
        )

        step += steps_per_epoch

        # Update history
        history['step'].append(step)
        history['loss'].append(metrics['loss'])
        history['lr'].append(metrics.get('lr', LEARNING_RATE))

        log(f"Epoch {epoch+1:3d} | Step {step:6d} | Loss: {metrics['loss']:.4f}")

        # Save checkpoint periodically
        if step % SAVE_FREQ_STEPS < steps_per_epoch:
            checkpoint_path = OUTPUT_DIR / f'pretrain_step{step:06d}.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                loss=metrics['loss'],
                output_path=checkpoint_path
            )
            log(f"  -> Saved checkpoint: {checkpoint_path.name}")

        if step >= TOTAL_STEPS:
            break

    # Save final model
    log("\n" + "=" * 60)
    log("Training Completed!")
    log("=" * 60)

    final_path = OUTPUT_DIR / 'pretrained.pth'
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        loss=metrics['loss'],
        output_path=final_path
    )
    log(f"Final model saved to: {final_path}")

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = OUTPUT_DIR / f"pretrain_history_{timestamp}.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log(f"Training history saved to: {history_path}")

    log(f"\nFinal Loss: {metrics['loss']:.4f}")
    log(f"Total Steps: {step}")


def wait_for_process(pid: int, check_interval: int = 60):
    """等待指定进程结束后再继续执行。

    Args:
        pid: 要等待的进程 ID
        check_interval: 检查间隔（秒）
    """
    import time
    import psutil

    print(f"等待进程 {pid} 结束...")
    print(f"检查间隔: {check_interval} 秒")

    while True:
        if not psutil.pid_exists(pid):
            print(f"进程 {pid} 已结束，开始执行训练...")
            break
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进程 {pid} 仍在运行，{check_interval} 秒后重新检查...")
            time.sleep(check_interval)


if __name__ == '__main__':
    # 等待指定进程结束后再执行
    WAIT_FOR_PID = 456936  # 要等待的进程 ID，设为 None 则不等待
    if WAIT_FOR_PID is not None:
        wait_for_process(WAIT_FOR_PID, check_interval=60)

    main()
