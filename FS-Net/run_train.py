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
from data import load_pickle_dataset, create_dataloaders
from engine import train_one_epoch, evaluate, save_checkpoint, load_checkpoint


# =============================================================================
# Configuration - Hardcoded Parameters
# =============================================================================

# Data (pickle file from iscx_processor.py)
DATA_PATH = "/home/pcz/DL/ML&DL/FS-Net/data/iscxvpn/iscxvpn_fsnet.pkl"
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
EPOCHS = 10
BATCH_SIZE = 2048         # Increased for RTX 4090
LEARNING_RATE = 0.0005    # Learning rate (paper: 0.0005)
# PATIENCE = 20           # Early stopping (disabled)
NUM_WORKERS = 0           # Windows compatibility (avoid multiprocessing issues)
SEED = 42                 # Random seed for reproducibility

# Dataset split ratio (8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Output
OUTPUT_DIR = "/home/pcz/DL/ML&DL/FS-Net/checkpoints"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable cuDNN for better GPU utilization
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Auto-tune for best performance


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

    # Load dataset from pickle
    print(f"\nLoading dataset from: {DATA_PATH}")
    sequences, labels, label_map = load_pickle_dataset(DATA_PATH)
    num_classes = NUM_CLASSES or len(label_map)

    print(f"Total samples: {len(labels)}")
    print(f"Num classes: {num_classes}")

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        class_name = label_map.get(label_id, f"Unknown({label_id})")
        print(f"  [{label_id:2d}] {class_name:15s}: {count:6d} ({count/len(labels)*100:5.1f}%)")

    # Create dataloaders with 8:1:1 split
    print(f"\nCreating dataloaders with {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO} split...")
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences, labels,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        use_direction=True
    )

    # Compute class weights for imbalanced data
    class_weight = None
    if USE_CLASS_WEIGHT:
        total = len(labels)
        weights = []
        for i in range(num_classes):
            count = (labels == i).sum()
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 1.0
            weights.append(weight)
        class_weight = torch.tensor(weights, dtype=torch.float32)
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
        print("\nUsing FS-Net-ND (no decoder)")
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
        print("\nUsing FS-Net")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # Print overall results
    print(f"Overall Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    print(f"  FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    # Print per-class results
    print("\n" + "-" * 80)
    print("Per-Class Results:")
    print("-" * 80)
    print(f"{'Class':<20} {'Count':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TPR':>10} {'FPR':>10}")
    print("-" * 80)

    for i in range(num_classes):
        class_name = label_map.get(i, f"Class_{i}")
        count = test_metrics['per_class_count'][i]
        precision = test_metrics['per_class_precision'][i]
        recall = test_metrics['per_class_recall'][i]
        f1 = test_metrics['per_class_f1'][i]
        tpr = test_metrics['per_class_tpr'][i]
        fpr = test_metrics['per_class_fpr'][i]
        print(f"{class_name:<20} {count:>8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tpr:>10.4f} {fpr:>10.4f}")

    print("-" * 80)
    print(f"\nModel saved to: {OUTPUT_DIR}/best.pth")


if __name__ == '__main__':
    main()
