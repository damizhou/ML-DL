#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
example_usage.py

Demonstrates how to use the train_npz_varlen.py script for training DFNoDefNet
on variable-length flow sequences stored in NPZ files.
"""

import subprocess
import sys
from pathlib import Path

def run_training_example():
    """
    Example 1: Basic training with default parameters
    """
    print("="*70)
    print("Example 1: Basic Training")
    print("="*70)
    
    cmd = [
        "python3", "train_npz_varlen.py",
        "--npz_root", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz",
        "--labels_json", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json",
        "--output_dir", "./runs/basic_experiment",
        "--epochs", "20",
        "--batch_size", "128",
        "--lr", "1e-3",
        "--seed", "42"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("This will:")
    print("- Train for 20 epochs")
    print("- Use batch size of 128")
    print("- Use learning rate of 1e-3")
    print("- Save checkpoints to ./runs/basic_experiment")
    print()

def run_gpu_optimized_example():
    """
    Example 2: GPU-optimized training with mixed precision
    """
    print("="*70)
    print("Example 2: GPU-Optimized Training with Mixed Precision")
    print("="*70)
    
    cmd = [
        "python3", "train_npz_varlen.py",
        "--npz_root", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz",
        "--labels_json", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json",
        "--output_dir", "./runs/gpu_experiment",
        "--epochs", "50",
        "--batch_size", "256",
        "--grad_accum_steps", "2",
        "--lr", "1e-3",
        "--use_amp",
        "--num_workers", "4",
        "--cache_size", "20",
        "--seed", "42"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("This will:")
    print("- Enable mixed precision training (faster on GPU)")
    print("- Use effective batch size of 512 (256 * 2 accum steps)")
    print("- Use 4 workers for parallel data loading")
    print("- Cache 20 NPZ files in memory")
    print()

def run_memory_efficient_example():
    """
    Example 3: Memory-efficient training for limited RAM
    """
    print("="*70)
    print("Example 3: Memory-Efficient Training (Limited RAM)")
    print("="*70)
    
    cmd = [
        "python3", "train_npz_varlen.py",
        "--npz_root", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz",
        "--labels_json", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json",
        "--output_dir", "./runs/memory_efficient",
        "--epochs", "30",
        "--batch_size", "64",
        "--grad_accum_steps", "4",
        "--lr", "1e-3",
        "--num_workers", "2",
        "--cache_size", "5",
        "--seed", "42"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("This will:")
    print("- Use smaller batch size (64) to reduce memory")
    print("- Use gradient accumulation for effective batch size of 256")
    print("- Cache only 5 files to minimize RAM usage")
    print("- Use 2 workers for data loading")
    print()

def run_resume_example():
    """
    Example 4: Resume training from checkpoint
    """
    print("="*70)
    print("Example 4: Resume Training from Checkpoint")
    print("="*70)
    
    cmd = [
        "python3", "train_npz_varlen.py",
        "--npz_root", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz",
        "--labels_json", "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json",
        "--output_dir", "./runs/resumed_experiment",
        "--resume", "./runs/resumed_experiment/last.pt",
        "--epochs", "100",
        "--batch_size", "128",
        "--lr", "1e-3",
        "--seed", "42"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("This will:")
    print("- Resume training from the last checkpoint")
    print("- Continue training until epoch 100")
    print("- Preserve optimizer state and training progress")
    print()

def main():
    """Display all usage examples."""
    print("\n" + "="*70)
    print(" Training Script Usage Examples for train_npz_varlen.py")
    print("="*70 + "\n")
    
    run_training_example()
    print()
    
    run_gpu_optimized_example()
    print()
    
    run_memory_efficient_example()
    print()
    
    run_resume_example()
    print()
    
    print("="*70)
    print(" Additional Tips")
    print("="*70)
    print()
    print("1. Monitor GPU usage:")
    print("   watch -n 1 nvidia-smi")
    print()
    print("2. Check training progress:")
    print("   tail -f runs/experiment_name/train.log  # If logging is enabled")
    print()
    print("3. Adjust hyperparameters based on dataset size:")
    print("   - Small dataset (<100 files): batch_size=64, lr=1e-4")
    print("   - Medium dataset (100-500 files): batch_size=128, lr=1e-3")
    print("   - Large dataset (>500 files): batch_size=256, lr=1e-3")
    print()
    print("4. For best GPU performance:")
    print("   - Use --use_amp for mixed precision")
    print("   - Increase --num_workers (4-8 on multi-core systems)")
    print("   - Use larger --batch_size if memory allows")
    print("   - Enable pinned memory (default on GPU)")
    print()
    print("="*70)
    print()

if __name__ == "__main__":
    main()
