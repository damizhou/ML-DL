"""
FS-Net Training Script (Hardcoded Parameters)

Usage:
    python run_train.py
"""

import os
import json
from pathlib import Path

import torch
import numpy as np

from models import create_fsnet, create_fsnet_nd
from data import build_dataloader, get_dataset_info
from engine import train_one_epoch, evaluate, save_checkpoint, load_checkpoint, EarlyStopping


# =============================================================================
# Configuration - Hardcoded Parameters
# =============================================================================

# Data
DATA_PATH = "/home/dev/DL/FS-Net/data/iscx_fsnet"
NUM_CLASSES = 12          # ISCX: 12 classes

# Model (Paper parameters: Section V-B-2)
EMBED_DIM = 128           # Embedding dimension (paper: 128)
HIDDEN_DIM = 128          # Hidden dimension (paper: 128)
NUM_LAYERS = 2            # Number of GRU layers (paper: 2)
DROPOUT = 0.3             # Dropout rate (paper: 0.3)
ALPHA = 1.0               # Reconstruction loss weight (paper: 1)
USE_NO_DECODER = False    # True: FS-Net-ND, False: FS-Net
USE_CLASS_WEIGHT = False  # Disable class weights

# Training (Paper parameters)
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0005    # Learning rate (paper: 0.0005)
PATIENCE = 20             # Early stopping patience
NUM_WORKERS = 4           # Data loading workers

# Output
OUTPUT_DIR = "/home/dev/DL/FS-Net/checkpoints"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable cuDNN for RNN (more stable)
torch.backends.cudnn.enabled = False


# =============================================================================
# Helper Functions
# =============================================================================

def compute_class_weights(data_path: str, classes: list) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.

    Uses inverse frequency weighting: weight = total / (num_classes * class_count)
    """
    # Load dataset info
    info_path = Path(data_path) / 'dataset_info.json'
    with open(info_path) as f:
        info = json.load(f)

    flow_counts = info['flow_counts']
    total = sum(flow_counts.values())
    num_classes = len(classes)

    weights = []
    for cls in classes:
        count = flow_counts.get(cls, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Training
# =============================================================================

def main():
    print("=" * 60)
    print("FS-Net Training")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Get dataset info
    info = get_dataset_info(DATA_PATH)
    num_classes = NUM_CLASSES or info['num_classes']
    print(f"Dataset: {info['num_classes']} classes, {info['total_samples']} samples")
    print(f"Classes: {info['classes']}")

    # Build data loaders
    train_loader = build_dataloader(
        DATA_PATH,
        split='train',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )

    val_split = 'val' if (Path(DATA_PATH) / 'val').exists() else 'test'
    val_loader = build_dataloader(
        DATA_PATH,
        split=val_split,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    test_loader = build_dataloader(
        DATA_PATH,
        split='test',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    # Compute class weights for imbalanced data
    class_weight = None
    if USE_CLASS_WEIGHT:
        class_weight = compute_class_weights(DATA_PATH, info['classes'])
        print(f"Class weights: {class_weight.tolist()}")

    # Create model
    if USE_NO_DECODER:
        model = create_fsnet_nd(
            num_classes=num_classes,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        print("Using FS-Net-ND (no decoder)")
    else:
        model = create_fsnet(
            num_classes=num_classes,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            alpha=ALPHA,
            class_weight=class_weight
        )
        print("Using FS-Net")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Training loop
    best_f1 = 0.0

    print("\nTraining Configuration:")
    print(f"  Data path:     {DATA_PATH}")
    print(f"  Num classes:   {num_classes}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Embed dim:     {EMBED_DIM}")
    print(f"  Hidden dim:    {HIDDEN_DIM}")
    print(f"  Num layers:    {NUM_LAYERS}")
    print(f"  Dropout:       {DROPOUT}")
    print(f"  Alpha:         {ALPHA}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, num_classes)
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, TPR: {val_metrics['tpr_avg']:.4f}, "
            f"FPR: {val_metrics['fpr_avg']:.4f}"
        )

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(OUTPUT_DIR, 'best.pth')
            )
            print(f"  -> Saved best model (F1: {best_f1:.4f})")

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
        os.path.join(OUTPUT_DIR, 'best.pth'),
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

    print(f"\nModel saved to: {OUTPUT_DIR}/best.pth")


if __name__ == '__main__':
    main()
