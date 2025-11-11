#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_npz_varlen.py
GPU-optimized, memory-efficient training script for DFNoDefNet using variable-length NPZ flows.

Features:
- Lazy loading of NPZ files to avoid high RAM usage
- Mixed precision training for GPU performance
- Variable-length sequence handling (padding/truncation)
- Label mapping from labels.json
- Efficient batching with custom collate function
- Checkpointing and logging
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path for model import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Model_NoDef_pytorch import DFNoDefNet


class LRUCache:
    """Simple LRU cache for NPZ file contents to balance memory and I/O."""
    
    def __init__(self, capacity: int = 10):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


class LazyNPZDataset(IterableDataset):
    """
    Memory-efficient iterable dataset that lazily loads NPZ files.
    
    Each NPZ contains:
    - flows: array of variable-length int8 arrays (values: -1, 1)
    - labels: array of string labels
    """
    
    def __init__(
        self,
        npz_paths: List[Path],
        label2id: Dict[str, int],
        max_len: int = 5000,
        cache_size: int = 10,
        shuffle: bool = True,
        seed: int = 0
    ):
        super().__init__()
        self.npz_paths = npz_paths
        self.label2id = label2id
        self.max_len = max_len
        self.cache = LRUCache(capacity=cache_size)
        self.shuffle = shuffle
        self.seed = seed
        
    def _load_npz(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load NPZ file with caching."""
        key = str(path)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        with np.load(path, allow_pickle=True) as data:
            flows = data['flows']
            labels = data['labels']
        
        # Don't cache to save memory - load on demand
        return flows, labels
    
    def _process_flow(self, flow: np.ndarray) -> np.ndarray:
        """Process variable-length flow to fixed length."""
        flow_len = len(flow)
        
        if flow_len >= self.max_len:
            # Truncate
            return flow[:self.max_len].astype(np.float32)
        else:
            # Pad with zeros
            padded = np.zeros(self.max_len, dtype=np.float32)
            padded[:flow_len] = flow.astype(np.float32)
            return padded
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Shuffle paths if needed
        paths = self.npz_paths[:]
        if self.shuffle:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(paths)
        
        # Split paths among workers if using multiple workers
        if worker_info is not None:
            per_worker = len(paths) // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker if worker_info.id < worker_info.num_workers - 1 else len(paths)
            paths = paths[start_idx:end_idx]
        
        # Iterate through NPZ files
        for npz_path in paths:
            try:
                flows, labels = self._load_npz(npz_path)
                
                # Iterate through samples in this NPZ
                for flow, label in zip(flows, labels):
                    # Convert label to ID
                    label_str = str(label)
                    if label_str not in self.label2id:
                        continue  # Skip unknown labels
                    
                    label_id = self.label2id[label_str]
                    
                    # Process flow
                    processed_flow = self._process_flow(flow)
                    
                    yield processed_flow, label_id
                    
            except Exception as e:
                print(f"Warning: Failed to load {npz_path}: {e}", file=sys.stderr)
                continue


def collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to batch variable-length flows.
    Flows are already padded to max_len by the dataset.
    """
    flows, labels = zip(*batch)
    
    # Stack flows and convert to tensor
    flows_tensor = torch.from_numpy(np.stack(flows, axis=0))  # (B, L)
    labels_tensor = torch.tensor(labels, dtype=torch.long)    # (B,)
    
    return flows_tensor, labels_tensor


def load_label_mapping(json_path: Path) -> Tuple[Dict[str, int], Dict[str, str], int]:
    """Load label mapping from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    label2id = data['label2id']
    id2label = data['id2label']
    num_classes = len(label2id)
    
    return label2id, id2label, num_classes


def collect_npz_files(root: Path) -> List[Path]:
    """Recursively collect all NPZ files under root directory."""
    npz_files = sorted(root.rglob("*.npz"))
    return npz_files


def split_dataset(
    npz_paths: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split NPZ files into train/val/test sets.
    Note: This is file-level split, not sample-level.
    """
    rng = random.Random(seed)
    paths = npz_paths[:]
    rng.shuffle(paths)
    
    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]
    
    return train_paths, val_paths, test_paths


def create_model(num_classes: int, seq_len: int = 5000) -> nn.Module:
    """Create DFNoDefNet model with correct number of output classes."""
    model = DFNoDefNet()
    
    # Replace classifier layer with correct number of classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Evaluate model on given dataset."""
    model.eval()
    
    correct = 0
    total = 0
    
    for flows, labels in loader:
        flows = flows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast():
                outputs = model(flows)
        else:
            outputs = model(flows)
        
        _, predicted = outputs.max(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    grad_accum_steps: int = 1
) -> float:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (flows, labels) in enumerate(loader):
        flows = flows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(flows)
                loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(flows)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
    
    # Handle remaining gradients
    if num_batches % grad_accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train DFNoDefNet on NPZ dataset')
    
    # Data arguments
    parser.add_argument('--npz_root', type=str, 
                        default='/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz',
                        help='Root directory containing NPZ files')
    parser.add_argument('--labels_json', type=str,
                        default='/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/labels.json',
                        help='Path to labels.json file')
    parser.add_argument('--output_dir', type=str, default='./runs',
                        help='Directory to save checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--max_len', type=int, default=5000,
                        help='Maximum sequence length (pad/truncate to this length)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cache_size', type=int, default=10,
                        help='Number of NPZ files to cache in memory')
    
    # Optimization arguments
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pinned memory for data loading')
    
    # Split arguments
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of files for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of files for validation')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label mapping
    labels_json_path = Path(args.labels_json)
    if not labels_json_path.exists():
        print(f"Error: labels.json not found at {labels_json_path}")
        return
    
    label2id, id2label, num_classes = load_label_mapping(labels_json_path)
    print(f"Loaded {num_classes} classes from {labels_json_path}")
    
    # Collect NPZ files
    npz_root = Path(args.npz_root)
    if not npz_root.exists():
        print(f"Error: NPZ root directory not found at {npz_root}")
        return
    
    npz_files = collect_npz_files(npz_root)
    if not npz_files:
        print(f"Error: No NPZ files found in {npz_root}")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(
        npz_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print(f"Split: Train={len(train_paths)} Val={len(val_paths)} Test={len(test_paths)}")
    
    # Create datasets
    train_dataset = LazyNPZDataset(
        train_paths,
        label2id,
        max_len=args.max_len,
        cache_size=args.cache_size,
        shuffle=True,
        seed=args.seed
    )
    
    val_dataset = LazyNPZDataset(
        val_paths,
        label2id,
        max_len=args.max_len,
        cache_size=args.cache_size,
        shuffle=False,
        seed=args.seed
    )
    
    # Create dataloaders
    pin_memory = not args.no_pin_memory and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    # Create model
    model = create_model(num_classes, seq_len=args.max_len)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input shape: (batch_size, 1, {args.max_len})")
    print(f"  Output classes: {num_classes}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create GradScaler for mixed precision
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Mixed precision: {args.use_amp and torch.cuda.is_available()}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=args.use_amp, grad_accum_steps=args.grad_accum_steps
        )
        
        # Validate
        val_acc, _ = evaluate(model, val_loader, device, use_amp=args.use_amp)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'max_len': args.max_len,
        }
        
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save last checkpoint
        torch.save(checkpoint, output_dir / 'last.pt')
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"*** New best validation accuracy: {best_val_acc:.4f} ***")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Evaluate on test set if available
    if test_paths:
        print("\nEvaluating on test set...")
        test_dataset = LazyNPZDataset(
            test_paths,
            label2id,
            max_len=args.max_len,
            cache_size=args.cache_size,
            shuffle=False,
            seed=args.seed
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
        
        # Load best model if it exists, otherwise use last model
        best_checkpoint_path = output_dir / 'best.pt'
        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print("Using best model checkpoint")
        else:
            print("Using last model checkpoint (best.pt not found)")
        
        test_acc, _ = evaluate(model, test_loader, device, use_amp=args.use_amp)
        print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
