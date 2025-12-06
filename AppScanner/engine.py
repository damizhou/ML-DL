"""
AppScanner Training Engine

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

This module provides training and evaluation functionality for AppScanner.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    loss: float
    accuracy: float
    epoch: int
    learning_rate: float


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confidence_accuracy: float  # Accuracy for confident predictions
    confidence_ratio: float     # Ratio of confident predictions
    confusion_matrix: np.ndarray
    per_class_accuracy: np.ndarray


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> TrainingMetrics:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scheduler: Learning rate scheduler

    Returns:
        Training metrics
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total
    accuracy = correct / total
    current_lr = optimizer.param_groups[0]['lr']

    return TrainingMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        epoch=0,  # Set by caller
        learning_rate=current_lr,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    prediction_threshold: float = 0.9,
    num_classes: Optional[int] = None,
) -> EvaluationMetrics:
    """
    Evaluate model on dataset.

    Args:
        model: Neural network model
        dataloader: Evaluation dataloader
        device: Device to use
        prediction_threshold: Confidence threshold (from paper Section V-C)
        num_classes: Number of classes for confusion matrix

    Returns:
        Evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        probs = F.softmax(outputs, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Confident predictions (paper's main metric)
    confident_mask = all_confidences >= prediction_threshold
    confidence_ratio = confident_mask.sum() / len(confident_mask)

    if confident_mask.sum() > 0:
        confidence_accuracy = accuracy_score(
            all_labels[confident_mask],
            all_predictions[confident_mask]
        )
    else:
        confidence_accuracy = 0.0

    # Confusion matrix
    if num_classes is None:
        num_classes = max(all_labels.max(), all_predictions.max()) + 1
    conf_matrix = confusion_matrix(
        all_labels, all_predictions,
        labels=list(range(num_classes))
    )

    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-10)

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confidence_accuracy=confidence_accuracy,
        confidence_ratio=confidence_ratio,
        confusion_matrix=conf_matrix,
        per_class_accuracy=per_class_acc,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Any,
    save_dir: str = './output',
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Full training loop.

    Args:
        model: Neural network model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration object
        save_dir: Directory to save checkpoints

    Returns:
        Trained model and training history
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(config.device)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01,
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode='max')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'confidence_acc': [],
        'learning_rate': [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')

    print(f"Training AppScanner on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        train_metrics.epoch = epoch

        # Validate
        val_metrics = evaluate(
            model, val_loader, device,
            prediction_threshold=config.prediction_threshold,
            num_classes=config.num_classes,
        )

        # Update history
        history['train_loss'].append(train_metrics.loss)
        history['train_acc'].append(train_metrics.accuracy)
        history['val_acc'].append(val_metrics.accuracy)
        history['val_f1'].append(val_metrics.f1)
        history['confidence_acc'].append(val_metrics.confidence_accuracy)
        history['learning_rate'].append(train_metrics.learning_rate)

        # Save best model
        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics.accuracy,
                'config': config,
            }, best_model_path)

        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Loss: {train_metrics.loss:.4f} | "
              f"Train Acc: {train_metrics.accuracy:.4f} | "
              f"Val Acc: {val_metrics.accuracy:.4f} | "
              f"Conf Acc: {val_metrics.confidence_accuracy:.4f} "
              f"({val_metrics.confidence_ratio:.1%}) | "
              f"LR: {train_metrics.learning_rate:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Early stopping disabled - run all epochs
        # if early_stopping(val_metrics.accuracy):
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("-" * 60)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

    return model, history


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    prediction_threshold: float = 0.9,
    label_map: Optional[Dict[int, str]] = None,
) -> EvaluationMetrics:
    """
    Test model and print detailed report.

    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to use
        prediction_threshold: Confidence threshold
        label_map: Mapping from label index to class name

    Returns:
        Evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = F.softmax(outputs, dim=-1)
            confidences, predictions = probs.max(dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Metrics
    num_classes = max(all_labels.max(), all_predictions.max()) + 1
    metrics = evaluate(
        model, test_loader, device,
        prediction_threshold=prediction_threshold,
        num_classes=num_classes,
    )

    # Print report
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:     {metrics.accuracy:.4f}")
    print(f"Precision (weighted): {metrics.precision:.4f}")
    print(f"Recall (weighted):    {metrics.recall:.4f}")
    print(f"F1 Score (weighted):  {metrics.f1:.4f}")
    print("-" * 60)
    print(f"Confidence Threshold: {prediction_threshold}")
    print(f"Confident Predictions: {metrics.confidence_ratio:.1%}")
    print(f"Confidence Accuracy:  {metrics.confidence_accuracy:.4f}")
    print("-" * 60)

    # Classification report
    if label_map is not None:
        target_names = [label_map[i] for i in range(num_classes)]
    else:
        target_names = [f"Class_{i}" for i in range(num_classes)]

    report = classification_report(
        all_labels, all_predictions,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    return metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    prediction_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Train and evaluate Random Forest classifier (original paper approach).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_estimators: Number of trees
        prediction_threshold: Confidence threshold

    Returns:
        Dictionary with model and metrics
    """
    from models import AppScannerRF

    print("Training Random Forest classifier...")
    rf = AppScannerRF(n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    # Predictions
    predictions, confidences, confident_mask = rf.predict_with_threshold(
        X_test, threshold=prediction_threshold
    )

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    confidence_accuracy = accuracy_score(
        y_test[confident_mask],
        predictions[confident_mask]
    ) if confident_mask.sum() > 0 else 0.0
    confidence_ratio = confident_mask.sum() / len(confident_mask)

    print(f"Random Forest Results:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Confidence Accuracy: {confidence_accuracy:.4f} ({confidence_ratio:.1%})")

    # Feature importance
    importance = rf.feature_importance()
    top_features = np.argsort(importance)[::-1][:10]
    print(f"  Top 10 features: {top_features}")

    return {
        'model': rf,
        'accuracy': accuracy,
        'confidence_accuracy': confidence_accuracy,
        'confidence_ratio': confidence_ratio,
        'feature_importance': importance,
    }


def compare_approaches(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Any,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different classification approaches (as in paper Section V).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration object

    Returns:
        Dictionary with results for each approach
    """
    from models import AppScannerNN, AppScannerRF

    results = {}

    # 1. Neural Network
    print("\n" + "=" * 40)
    print("Approach: Neural Network")
    print("=" * 40)

    from data import create_dataloaders
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples
    train_ratio = 1.0 - test_ratio
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        np.vstack([X_train, X_test]),
        np.hstack([y_train, y_test]),
        batch_size=config.batch_size,
        train_ratio=train_ratio,
        val_ratio=0.0,
        test_ratio=test_ratio,
    )

    model = AppScannerNN(
        input_dim=X_train.shape[1],
        num_classes=len(np.unique(y_train)),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    model, history = train(model, train_loader, val_loader, config)
    nn_metrics = test(model, test_loader, torch.device(config.device))

    results['neural_network'] = {
        'accuracy': nn_metrics.accuracy,
        'confidence_accuracy': nn_metrics.confidence_accuracy,
        'f1': nn_metrics.f1,
    }

    # 2. Random Forest
    print("\n" + "=" * 40)
    print("Approach: Random Forest")
    print("=" * 40)

    rf_results = train_random_forest(
        X_train, y_train, X_test, y_test,
        n_estimators=config.n_estimators,
        prediction_threshold=config.prediction_threshold,
    )

    results['random_forest'] = {
        'accuracy': rf_results['accuracy'],
        'confidence_accuracy': rf_results['confidence_accuracy'],
    }

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name:20s}: Acc={metrics['accuracy']:.4f}, "
              f"ConfAcc={metrics.get('confidence_accuracy', 0):.4f}")

    return results


if __name__ == '__main__':
    # Test training engine with dummy data
    from models import AppScannerNN
    from data import create_dataloaders
    from config import get_config

    config = get_config()
    config.epochs = 5
    config.num_classes = 10

    # Create dummy data
    n_samples = 1000
    n_features = 54

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, config.num_classes, n_samples)

    # Create dataloaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        features, labels,
        batch_size=config.batch_size,
    )

    # Create model
    model = AppScannerNN(
        input_dim=n_features,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    # Train
    model, history = train(model, train_loader, val_loader, config)

    # Test
    metrics = test(model, test_loader, torch.device(config.device))

    print("\nEngine tests passed!")
