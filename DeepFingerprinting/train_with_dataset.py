"""
Deep Fingerprinting Training Script

Paper: Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning
Conference: CCS 2018

Usage:
    python train.py
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from Model_NoDef_pytorch import DFNoDefNet


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)

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

    return log_path


def log(message: str = ""):
    """Log message to both console and file."""
    logging.info(message)


# =============================================================================
# Configuration (Paper: Table 1)
# =============================================================================

@dataclass
class TrainArgs:
    """Training arguments - all parameters from the paper."""

    # Mode: 'train', 'eval'
    mode: str = 'train'

    # Data path (supports unified_dir and single_npz formats)
    # data_path: str = '/root/autodl-tmp/data/iscxvpn'         # single_npz: data.npz + labels.json
    # data_path: str = '/root/autodl-tmp/data/iscxtor'         # single_npz: data.npz + labels.json
    # data_path: str = '/root/autodl-tmp/data/ustc'            # single_npz: data.npz + labels.json
    # data_path: str = '/root/autodl-tmp/data/cic_iot_2022'    # single_npz: data.npz + labels.json
    # data_path: str = '/root/autodl-tmp/data/cross_platform'  # single_npz: data.npz + labels.json
    data_path: str = '/root/autodl-tmp/data/vpn'             # unified_dir: 多个 .npz 文件
    # data_path: str = '/root/autodl-tmp/data/novpn'           # unified_dir: 多个 .npz 文件

    # Model configuration (Paper Section 5.1, Table 1)
    input_len: int = 5000                       # Fixed input length (Paper: 5000)
    num_classes: Optional[int] = None           # Auto-detect from data

    # Training parameters (Paper Table 1)
    epochs: int = 100                            # Paper: 30
    batch_size: int = 128                       # 论文现实的代码中的数据值
    accum_steps: int = 1                        # Gradient accumulation steps (effective batch = 2048/16 = 128)
    lr: float = 0.002                           # Paper: 0.002
    optimizer: str = 'adamax'                   # Paper: Adamax

    # Dataset split ratio (Paper: 8:1:1)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Data filtering
    min_samples: int = 10                       # Minimum samples per class

    # Paths
    output_dir: str = './output'
    checkpoint: Optional[str] = None

    # Device
    device: str = 'auto'

    # Misc
    seed: int = 42
    num_workers: int = 4                        # Windows compatible


def get_args() -> TrainArgs:
    """Get training arguments with optional command line override for data_path."""
    import argparse

    parser = argparse.ArgumentParser(description='Deep Fingerprinting Training Script')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory (overrides default)')

    args = parser.parse_args()

    # Create TrainArgs with defaults
    train_args = TrainArgs()

    # Override data_path if provided
    if args.data_path is not None:
        train_args.data_path = args.data_path

    return train_args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset
# =============================================================================

class DFDataset(Dataset):
    """Deep Fingerprinting Dataset - 惰性加载模式，支持大数据集。"""

    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray, max_len: int = 5000):
        """
        Args:
            sequences: 变长序列列表 (保持原始格式，不预处理)
            labels: 标签数组
            max_len: 固定序列长度
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = min(len(seq), self.max_len)

        # 即时处理：pad/truncate + 转换
        x = np.zeros(self.max_len, dtype=np.float32)
        x[:seq_len] = seq[:seq_len].astype(np.float32)

        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


def load_unified_dir(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """Load data from unified directory format (multiple .npz files)."""
    sequences = []
    labels = []
    label_map = {}

    npz_files = sorted(data_path.glob('*.npz'))
    log(f"Found {len(npz_files)} .npz files")

    for label_id, npz_file in enumerate(npz_files):
        data = np.load(npz_file, allow_pickle=True)
        flows = data['flows']
        class_name = npz_file.stem

        for flow in flows:
            sequences.append(np.array(flow, dtype=np.int8))
            labels.append(label_id)

        label_map[label_id] = class_name

    labels = np.array(labels, dtype=np.int64)
    return sequences, labels, label_map


def load_single_npz(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """Load data from single NPZ format (data.npz + labels.json)."""
    npz_file = data_path / 'data.npz'
    label_file = data_path / 'labels.json'

    data = np.load(npz_file, allow_pickle=True)
    X = data['X']
    y = data['y']

    sequences = [np.array(x, dtype=np.int8) for x in X]
    labels = np.array(y, dtype=np.int64)

    # Load label map
    label_map = {}
    if label_file.exists():
        with open(label_file, 'r') as f:
            label_info = json.load(f)
            if 'id2label' in label_info:
                label_map = {int(k): v for k, v in label_info['id2label'].items()}
            else:
                unique_labels = np.unique(labels)
                label_map = {int(i): str(i) for i in unique_labels}
    else:
        unique_labels = np.unique(labels)
        label_map = {int(i): str(i) for i in unique_labels}

    return sequences, labels, label_map


def load_dataset(data_path: str) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """Auto-detect format and load dataset."""
    path = Path(data_path)

    if (path / 'data.npz').exists():
        log("Data format: single_npz")
        return load_single_npz(path)
    elif list(path.glob('*.npz')):
        log("Data format: unified_dir")
        return load_unified_dir(path)
    else:
        raise ValueError(f"No valid data found in {data_path}")


def filter_classes(sequences: List, labels: np.ndarray, label_map: Dict,
                   min_samples: int) -> Tuple[List, np.ndarray, Dict[int, str]]:
    """Filter out classes with fewer than min_samples."""
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= min_samples]

    if len(valid_classes) == len(unique):
        return sequences, labels, label_map

    log(f"Filtering: {len(unique)} -> {len(valid_classes)} classes (min_samples={min_samples})")

    # Create new mapping
    new_label_map = {}
    old_to_new = {}
    for new_id, old_id in enumerate(valid_classes):
        old_to_new[old_id] = new_id
        new_label_map[new_id] = label_map.get(old_id, str(old_id))

    # Filter data
    new_sequences = []
    new_labels = []
    for seq, label in zip(sequences, labels):
        if label in old_to_new:
            new_sequences.append(seq)
            new_labels.append(old_to_new[label])

    return new_sequences, np.array(new_labels, dtype=np.int64), new_label_map


def create_dataloaders(sequences: List, labels: np.ndarray, args: TrainArgs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with stratified split."""
    # First split: train vs (val+test)
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        sequences, labels,
        test_size=(args.val_ratio + args.test_ratio),
        random_state=args.seed,
        stratify=labels
    )

    # Second split: val vs test
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(
        temp_seqs, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=args.seed,
        stratify=temp_labels
    )

    log(f"Split: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")

    # 创建数据集 (惰性加载模式)
    log("Creating datasets (lazy loading mode for large dataset)...")
    train_dataset = DFDataset(train_seqs, train_labels, args.input_len)
    val_dataset = DFDataset(val_seqs, val_labels, args.input_len)
    test_dataset = DFDataset(test_seqs, test_labels, args.input_len)

    # 使用多 workers 加速数据加载
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# =============================================================================
# Training Engine
# =============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, scaler, accum_steps: int = 1) -> Tuple[float, float]:
    """Train for one epoch with gradient accumulation.

    Args:
        accum_steps: Number of gradient accumulation steps.
                     Effective batch size = batch_size / accum_steps
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
            # Scale loss by accumulation steps
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every accum_steps or at the last step
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * x.size(0)  # Undo the scaling for logging
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> Dict:
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class metrics
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_count = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_tpr = []
    per_class_fpr = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        count = cm[i, :].sum()

        per_class_count.append(count)
        per_class_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        per_class_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        per_class_tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        per_class_fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        p = per_class_precision[-1]
        r = per_class_recall[-1]
        per_class_f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

    tpr_avg = np.mean(per_class_tpr)
    fpr_avg = np.mean(per_class_fpr)

    return {
        'loss': total_loss / total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tpr_avg': tpr_avg,
        'fpr_avg': fpr_avg,
        'per_class_count': per_class_count,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_tpr': per_class_tpr,
        'per_class_fpr': per_class_fpr,
    }


# =============================================================================
# Training Mode
# =============================================================================

def mode_train(args: TrainArgs):
    """Training mode."""
    log("=" * 70)
    log("Deep Fingerprinting Training")
    log("Paper: CCS 2018")
    log("=" * 70)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log(f"Device: {device}")

    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Load data
    log(f"\nData path: {args.data_path}")
    sequences, labels, label_map = load_dataset(args.data_path)
    log(f"Total samples: {len(labels)}")
    log(f"Original classes: {len(np.unique(labels))}")

    # Filter classes
    sequences, labels, label_map = filter_classes(sequences, labels, label_map, args.min_samples)
    num_classes = len(label_map)
    args.num_classes = num_classes
    log(f"Kept classes: {num_classes}")

    # Print class distribution
    # log("\nClass distribution:")
    # unique, counts = np.unique(labels, return_counts=True)
    # for label_id, count in zip(unique, counts):
    #     class_name = label_map.get(label_id, f"Class_{label_id}")
    #     log(f"  [{label_id:3d}] {class_name:25s}: {count:6d} ({count/len(labels)*100:5.1f}%)")

    # Create dataloaders (惰性加载模式)
    log(f"\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(sequences, labels, args)

    # Create model
    model = DFNoDefNet(num_classes=num_classes)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"\nModel: DFNoDefNet")
    log(f"Parameters: {total_params:,}")

    # Optimizer (Paper: Adamax)
    if args.optimizer.lower() == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training configuration
    effective_batch = args.batch_size // args.accum_steps
    log("\nTraining Configuration:")
    log(f"  Epochs:          {args.epochs}")
    log(f"  Batch size:      {args.batch_size} (effective: {effective_batch} with {args.accum_steps} accum steps)")
    log(f"  Learning rate:   {args.lr}")
    log(f"  Optimizer:       {args.optimizer}")
    log(f"  Input length:    {args.input_len}")
    log("=" * 70)

    # Training loop
    best_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, args.accum_steps)
        val_metrics = evaluate(model, val_loader, device, num_classes)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'num_classes': num_classes,
                'label_map': label_map,
            }, os.path.join(args.output_dir, 'best_model.pth'))

        log(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"Time: {epoch_time:.1f}s {'*' if is_best else ''}")

    # Load best model for final evaluation
    log("\n" + "=" * 70)
    log("Final Evaluation on Test Set")
    log("=" * 70)

    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, num_classes)

    log("\nOverall Results:")
    log(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    log(f"  Precision: {test_metrics['precision']:.4f}")
    log(f"  Recall:    {test_metrics['recall']:.4f}")
    log(f"  F1 Score:  {test_metrics['f1']:.4f}")
    log(f"  TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    log(f"  FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    # Per-class results
    log("\n" + "-" * 90)
    log("Per-Class Results:")
    log("-" * 90)
    log(f"{'Class':<25} {'Count':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TPR':>10} {'FPR':>10}")
    log("-" * 90)

    for i in range(num_classes):
        class_name = label_map.get(i, f"Class_{i}")
        if len(class_name) > 24:
            class_name = class_name[:21] + "..."
        log(f"{class_name:<25} {test_metrics['per_class_count'][i]:>8} "
            f"{test_metrics['per_class_precision'][i]:>10.4f} "
            f"{test_metrics['per_class_recall'][i]:>10.4f} "
            f"{test_metrics['per_class_f1'][i]:>10.4f} "
            f"{test_metrics['per_class_tpr'][i]:>10.4f} "
            f"{test_metrics['per_class_fpr'][i]:>10.4f}")

    log("-" * 90)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'label_map': label_map,
        'args': args,
        'history': history,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'tpr_avg': test_metrics['tpr_avg'],
            'fpr_avg': test_metrics['fpr_avg'],
        },
    }, final_path)
    log(f"\nModel saved to {final_path}")

    # Save history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(args.output_dir, f'history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log(f"Training history saved to {history_path}")

    return model, test_metrics


# =============================================================================
# Evaluation Mode
# =============================================================================

def mode_eval(args: TrainArgs):
    """Evaluation mode."""
    log("=" * 70)
    log("Deep Fingerprinting Evaluation")
    log("=" * 70)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log(f"Device: {device}")

    # Checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, 'best_model.pth')

    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_classes = checkpoint['num_classes']
    label_map = checkpoint['label_map']

    # Load data
    sequences, labels, _ = load_dataset(args.data_path)
    sequences, labels, label_map = filter_classes(sequences, labels, label_map, args.min_samples)

    # Use all data for testing
    test_dataset = DFDataset(sequences, labels, args.input_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create and load model
    model = DFNoDefNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    test_metrics = evaluate(model, test_loader, device, num_classes)

    log("\n" + "=" * 70)
    log("TEST RESULTS")
    log("=" * 70)
    log(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    log(f"Precision: {test_metrics['precision']:.4f}")
    log(f"Recall:    {test_metrics['recall']:.4f}")
    log(f"F1 Score:  {test_metrics['f1']:.4f}")
    log(f"TPR_AVE:   {test_metrics['tpr_avg']:.4f}")
    log(f"FPR_AVE:   {test_metrics['fpr_avg']:.4f}")

    return test_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = datetime.now()

    args = get_args()
    set_seed(args.seed)

    # Create dataset-specific output directory to avoid conflicts
    dataset_name = Path(args.data_path).name
    args.output_dir = os.path.join(args.output_dir, dataset_name)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logging(args.output_dir)

    log("\nConfiguration:")
    log(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Mode: {args.mode}")
    log(f"  Data: {args.data_path}")
    log(f"  Device: {args.device}")
    log(f"  Log file: {log_path}")
    log()

    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'eval':
        mode_eval(args)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log()
    log(f"Start time:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"End time:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == '__main__':
    main()
