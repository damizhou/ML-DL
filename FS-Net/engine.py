"""
FS-Net Training Engine

Training and evaluation utilities for FS-Net:
- Training loop
- Evaluation metrics (TPR, FPR, FTF)
- Model checkpointing
"""

import time
import logging
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEFAULT_TRAIN_CONFIG, TrainConfig


def log(message: str = ""):
    """Log message using configured logger."""
    logging.info(message)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 50
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model: FS-Net model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency

    Returns:
        Dictionary of training metrics
    """
    model.train()

    loss_meter = AverageMeter()
    class_loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    start_time = time.time()

    for batch_idx, (sequences, lengths, labels) in enumerate(dataloader):
        # Move to device
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # Forward pass
        class_logits, recon_logits = model(sequences, lengths)

        # Compute loss
        total_loss, class_loss, recon_loss = model.compute_loss(
            class_logits, recon_logits, labels, sequences, lengths
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = class_logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        # Update meters
        batch_size = sequences.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        class_loss_meter.update(class_loss.item(), batch_size)
        recon_loss_meter.update(recon_loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        # Print progress
        if (batch_idx + 1) % print_freq == 0:
            log(f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} (C: {class_loss_meter.avg:.4f}, "
                f"R: {recon_loss_meter.avg:.4f}) Acc: {acc_meter.avg:.4f}")

    elapsed = time.time() - start_time

    return {
        'loss': loss_meter.avg,
        'class_loss': class_loss_meter.avg,
        'recon_loss': recon_loss_meter.avg,
        'accuracy': acc_meter.avg,
        'time': elapsed
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Evaluate model on test set.

    Computes metrics from paper:
    - TPR (True Positive Rate) per class and average
    - FPR (False Positive Rate) per class and average
    - FTF (F1-like metric)

    Args:
        model: FS-Net model
        dataloader: Test data loader
        device: Device to evaluate on
        num_classes: Number of classes

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    total_class_loss = 0
    total_recon_loss = 0
    num_samples = 0

    for sequences, lengths, labels in dataloader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # Forward pass
        class_logits, recon_logits = model(sequences, lengths)

        # Compute loss
        total, class_loss, recon_loss = model.compute_loss(
            class_logits, recon_logits, labels, sequences, lengths
        )

        batch_size = sequences.size(0)
        total_loss += total.item() * batch_size
        total_class_loss += class_loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        num_samples += batch_size

        # Get predictions
        preds = class_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Compute per-class metrics
    tpr_list = []
    fpr_list = []
    flow_counts = []

    for c in range(num_classes):
        # True positives, false positives, true negatives, false negatives
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        tn = ((all_preds != c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()

        # TPR = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        flow_counts.append((all_labels == c).sum().item())

    # Compute weighted averages (as in paper)
    total_flows = sum(flow_counts)
    weights = [fc / total_flows for fc in flow_counts]

    tpr_avg = sum(w * tpr for w, tpr in zip(weights, tpr_list))
    fpr_avg = sum(w * fpr for w, fpr in zip(weights, fpr_list))

    # FTF metric from paper: sum(wi * TPRi / (1 + FPRi))
    ftf = sum(
        w * (tpr / (1 + fpr))
        for w, tpr, fpr in zip(weights, tpr_list, fpr_list)
    )

    # Overall accuracy
    accuracy = (all_preds == all_labels).float().mean().item()

    # Per-class precision, recall, F1
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    return {
        'loss': total_loss / num_samples,
        'class_loss': total_class_loss / num_samples,
        'recon_loss': total_recon_loss / num_samples,
        'accuracy': accuracy,
        'tpr_avg': tpr_avg,
        'fpr_avg': fpr_avg,
        'ftf': ftf,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'per_class_tpr': tpr_list,
        'per_class_fpr': fpr_list,
        'per_class_precision': precision_list,
        'per_class_recall': recall_list,
        'per_class_f1': f1_list,
        'per_class_count': flow_counts
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Evaluation metrics
        path: Path to save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    log(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None
) -> Tuple[nn.Module, int, Dict]:
    """Load model checkpoint.

    Args:
        model: Model to load weights into
        path: Path to checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load to

    Returns:
        Tuple of (model, epoch, metrics)
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    log(f"Loaded checkpoint from {path} (epoch {epoch})")

    return model, epoch, metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
