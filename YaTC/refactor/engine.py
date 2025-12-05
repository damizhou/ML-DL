"""
YaTC Training Engine

This module implements training and evaluation loops for the YaTC model:
- pretrain_one_epoch: One epoch of MAE pre-training
- train_one_epoch: One epoch of fine-tuning
- evaluate: Model evaluation on test set

Training hyperparameters are consistent with the paper.
"""

import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEFAULT_PRETRAIN_CONFIG, DEFAULT_FINETUNE_CONFIG


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


def adjust_learning_rate_pretrain(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0
) -> float:
    """Adjust learning rate with warmup and cosine decay for pre-training.

    Args:
        optimizer: Optimizer
        step: Current step
        total_steps: Total number of steps
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate

    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        # Linear warmup
        lr = base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_finetune(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6
) -> float:
    """Adjust learning rate with warmup and cosine decay for fine-tuning.

    Args:
        optimizer: Optimizer
        epoch: Current epoch
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate

    Returns:
        Current learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group['lr'] = lr * param_group["lr_scale"]
        else:
            param_group['lr'] = lr

    return lr


def get_layer_id_for_traformer(name: str, num_layers: int) -> int:
    """Get layer id for layer-wise learning rate decay.

    Args:
        name: Parameter name
        num_layers: Total number of layers

    Returns:
        Layer id (0 for embedding, 1 to num_layers for blocks, num_layers+1 for head)
    """
    if name.startswith('patch_embed') or name.startswith('cls_token') or name.startswith('pos_embed'):
        return 0
    elif name.startswith('blocks'):
        # Extract block number from name like 'blocks.0.xxx'
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        # Classification head
        return num_layers + 1


def get_param_groups_with_layer_decay(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    num_layers: int
) -> list:
    """Create parameter groups with layer-wise learning rate decay.

    Args:
        model: Model
        base_lr: Base learning rate
        weight_decay: Weight decay
        layer_decay: Layer decay rate (0.65 in paper)
        num_layers: Number of transformer layers

    Returns:
        List of parameter groups
    """
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias and LayerNorm
        if param.ndim == 1 or 'bias' in name:
            wd = 0.0
            group_name = "no_decay"
        else:
            wd = weight_decay
            group_name = "decay"

        layer_id = get_layer_id_for_traformer(name, num_layers)

        # Learning rate scale based on layer
        lr_scale = layer_decay ** (num_layers + 1 - layer_id)

        group_key = f"{group_name}_layer{layer_id}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "weight_decay": wd,
                "lr_scale": lr_scale,
            }

        param_groups[group_key]["params"].append(param)

    return list(param_groups.values())


def pretrain_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    step_offset: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
    mask_ratio: float = 0.9,
    print_freq: int = 100
) -> Dict[str, float]:
    """One epoch of MAE pre-training.

    Args:
        model: MAE model
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        step_offset: Step offset from previous epochs
        total_steps: Total training steps
        warmup_steps: Warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        mask_ratio: Ratio of patches to mask
        print_freq: Print frequency

    Returns:
        Dictionary of training metrics
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, samples in enumerate(data_loader):
        step = step_offset + batch_idx

        # Adjust learning rate
        lr = adjust_learning_rate_pretrain(
            optimizer, step, total_steps, warmup_steps, base_lr, min_lr
        )

        # Move to device
        samples = samples.to(device, non_blocking=True)

        # Forward pass
        loss, _, _ = model(samples, mask_ratio=mask_ratio)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), samples.size(0))

        # Print progress
        if batch_idx % print_freq == 0:
            print(
                f"Epoch [{epoch}] Step [{step}/{total_steps}] "
                f"Loss: {loss_meter.avg:.4f} LR: {lr:.6f}"
            )

    return {
        "loss": loss_meter.avg,
        "lr": lr
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
    print_freq: int = 100,
    mixup_fn: Optional[callable] = None
) -> Dict[str, float]:
    """One epoch of fine-tuning.

    Args:
        model: Traffic Transformer model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs
        warmup_epochs: Warmup epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        print_freq: Print frequency
        mixup_fn: Optional mixup function

    Returns:
        Dictionary of training metrics
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # Adjust learning rate
    lr = adjust_learning_rate_finetune(
        optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr
    )

    for batch_idx, (samples, targets) in enumerate(data_loader):
        # Move to device
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup if provided
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy (only when not using mixup)
        if mixup_fn is None:
            _, preds = outputs.max(dim=1)
            correct = preds.eq(targets).sum().item()
            acc = correct / samples.size(0)
            acc_meter.update(acc, samples.size(0))

        loss_meter.update(loss.item(), samples.size(0))

        # Print progress
        if batch_idx % print_freq == 0:
            acc_str = f"Acc: {acc_meter.avg:.4f}" if mixup_fn is None else ""
            print(
                f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {loss_meter.avg:.4f} {acc_str} LR: {lr:.6f}"
            )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg if mixup_fn is None else 0.0,
        "lr": lr
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Traffic Transformer model
        data_loader: Test data loader
        device: Device to evaluate on
        num_classes: Number of classes

    Returns:
        Dictionary of evaluation metrics including:
        - accuracy: Overall accuracy
        - precision: Macro precision
        - recall: Macro recall
        - f1: Macro F1 score
    """
    model.eval()

    all_preds = []
    all_targets = []

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)

        outputs = model(samples)
        _, preds = outputs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)

    # Calculate accuracy
    correct = all_preds.eq(all_targets).sum().item()
    accuracy = correct / len(all_targets)

    # Calculate per-class metrics
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0

    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((all_preds == cls) & (all_targets == cls)).sum().item()
        fp = ((all_preds == cls) & (all_targets != cls)).sum().item()
        fn = ((all_preds != cls) & (all_targets == cls)).sum().item()

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    # Macro averages
    macro_precision = precision_sum / num_classes
    macro_recall = recall_sum / num_classes
    macro_f1 = f1_sum / num_classes

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall: {macro_recall:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False
) -> nn.Module:
    """Load pre-trained weights into model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce matching keys

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove decoder weights for fine-tuning
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('decoder') and not k.startswith('mask_token')
    }

    # Load weights
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded pre-trained weights from {checkpoint_path}")
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    output_path: str
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        output_path: Path to save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")
