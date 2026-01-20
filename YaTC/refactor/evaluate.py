#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC 评估脚本

加载已训练的模型进行验证和测试。

Usage:
    python evaluate.py

修改下方配置后运行即可。
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models import traformer_yatc
from data import build_npz_finetune_dataloader, build_split_dataloader
from engine import evaluate


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path) -> str:
    """Setup logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# 配置 - 请根据需要修改
# =============================================================================

# 模型路径
MODEL_PATH = Path("/home/pcz/DL/ML_DL/YaTC/checkpoints/yatc_best.pth")

# 数据路径
DATA_PATH = Path(__file__).parent.parent / "data" / "vpn_unified_output_split"

# 是否使用已划分的目录结构 (train/val/test 子目录)
USE_SPLIT_DIR = True
TRAIN_PATH = None
VAL_PATH = None
TEST_PATH = DATA_PATH / "test"

# 数据格式
USE_NPZ = True

# 数据划分比例 (仅在 USE_SPLIT_DIR=False 时使用)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_SAMPLES = 10

# DataLoader 参数
BATCH_SIZE = 64
NUM_WORKERS = 4

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型参数
DROP_PATH_RATE = 0.1


# =============================================================================
# 评估函数
# =============================================================================

def load_model(model_path: Path, num_classes: int, device: torch.device):
    """加载模型"""
    model = traformer_yatc(
        num_classes=num_classes,
        drop_path_rate=DROP_PATH_RATE
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    # 支持多种保存格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def get_predictions(model, data_loader, device):
    """获取所有预测结果"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            outputs = model(samples)
            _, preds = outputs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    return np.array(all_preds), np.array(all_labels)


def print_results(name: str, metrics: dict, preds, labels, id2label: dict, num_classes: int):
    """打印评估结果"""
    log(f"\n{'=' * 60}")
    log(f"{name} Results")
    log('=' * 60)
    log(f"Accuracy:  {metrics['accuracy']:.4f}")
    log(f"Precision: {metrics['precision']:.4f}")
    log(f"Recall:    {metrics['recall']:.4f}")
    log(f"F1 Score:  {metrics['f1']:.4f}")

    # Classification Report
    log(f"\n{'-' * 60}")
    log("Classification Report:")
    log('-' * 60)
    labels_list = list(range(num_classes))
    target_names = [id2label[i] for i in labels_list]
    log(classification_report(labels, preds, labels=labels_list,
                                target_names=target_names, zero_division=0))


# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "evaluate"


def main():
    # 初始化日志
    log_path = setup_logging(OUTPUT_DIR)

    log("=" * 60)
    log("YaTC Model Evaluation")
    log("=" * 60)
    log(f"Model path: {MODEL_PATH}")
    log(f"Data path:  {DATA_PATH}")
    log(f"Device:     {DEVICE}")
    log(f"Data format: {'NPZ' if USE_NPZ else 'PNG'}")
    log(f"Log file:   {log_path}")

    device = torch.device(DEVICE)

    # 构建测试集数据加载器
    if USE_SPLIT_DIR and TEST_PATH:
        test_loader, num_classes = build_split_dataloader(
            TEST_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
        )
        id2label = test_loader.dataset.id2label
    elif USE_NPZ:
        test_loader, num_classes = build_npz_finetune_dataloader(
            DATA_PATH, split='test', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            shuffle=False, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, min_samples=MIN_SAMPLES
        )
        id2label = test_loader.dataset.id2label
    else:
        log("Error: Non-NPZ format requires manual configuration")
        sys.exit(1)

    log(f"\nNum classes: {num_classes}")
    log(f"Test samples: {len(test_loader.dataset)}")

    # 加载模型
    log(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, num_classes, device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试集评估
    log("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, num_classes)
    test_preds, test_labels = get_predictions(model, test_loader, device)
    print_results("Test", test_metrics, test_preds, test_labels, id2label, num_classes)

    log("\n" + "=" * 60)
    log("Evaluation Complete!")
    log(f"Log saved to: {log_path}")
    log("=" * 60)


if __name__ == '__main__':
    main()
