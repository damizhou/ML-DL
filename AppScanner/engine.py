"""
AppScanner Training Engine

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

This module provides training and evaluation functionality for AppScanner.
"""

import os
import time
import gc
import logging
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


def log(message: str = ""):
    """Log message using configured logger."""
    logging.info(message)


def _get_process_memory_usage() -> Tuple[float, float]:
    """
    Get current process RSS memory usage.

    Returns:
        (rss_gb, rss_percent_of_total_ram)
    """
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        rss_bytes = proc.memory_info().rss
        total_bytes = psutil.virtual_memory().total
        rss_gb = rss_bytes / (1024 ** 3)
        rss_pct = (rss_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        return rss_gb, rss_pct
    except Exception:
        # Linux fallback when psutil is unavailable.
        try:
            rss_kb = None
            total_kb = None

            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_kb = float(parts[1])
                        break

            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb = float(parts[1])
                        break

            if rss_kb is not None and total_kb is not None and total_kb > 0:
                rss_gb = (rss_kb * 1024.0) / (1024 ** 3)
                rss_pct = (rss_kb / total_kb) * 100.0
                return rss_gb, rss_pct
        except Exception:
            pass

    return float("nan"), float("nan")


def _trim_process_memory() -> None:
    """Best-effort heap trim on glibc systems after large temporary allocations."""
    try:
        if os.name != "posix":
            return
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


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

    log(f"Training AppScanner on {device}")
    log(f"Training samples: {len(train_loader.dataset)}")
    log(f"Validation samples: {len(val_loader.dataset)}")
    log("-" * 60)

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
        log(f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"Loss: {train_metrics.loss:.4f} | "
            f"Train Acc: {train_metrics.accuracy:.4f} | "
            f"Val Acc: {val_metrics.accuracy:.4f} | "
            f"Val F1: {val_metrics.f1:.4f} | "
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

    log("-" * 60)
    log(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

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
    log("\n" + "=" * 60)
    log("TEST RESULTS")
    log("=" * 60)
    log(f"Overall Accuracy:     {metrics.accuracy:.4f}")
    log(f"Precision (weighted): {metrics.precision:.4f}")
    log(f"Recall (weighted):    {metrics.recall:.4f}")
    log(f"F1 Score (weighted):  {metrics.f1:.4f}")
    log("-" * 60)
    log(f"Confidence Threshold: {prediction_threshold}")
    log(f"Confident Predictions: {metrics.confidence_ratio:.1%}")
    log(f"Confidence Accuracy:  {metrics.confidence_accuracy:.4f}")
    log("-" * 60)

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
    log("\nClassification Report:")
    log(report)

    return metrics


def _auto_eval_batch_size(
    n_classes: int,
    prob_buffer_mb: int = 256,
    max_batch_size: int = 100_000,
) -> int:
    """
    Compute an evaluation batch size based on probability-buffer memory budget.

    The main evaluation buffer is (batch_size, n_classes) float32.
    """
    if n_classes <= 0:
        return 1

    bytes_per_value = np.dtype(np.float32).itemsize
    bytes_per_row = n_classes * bytes_per_value
    target_bytes = max(32, int(prob_buffer_mb)) * 1024 * 1024

    batch_size = target_bytes // bytes_per_row
    if batch_size <= 0:
        batch_size = 1
    return max(1, min(max_batch_size, int(batch_size)))


def predict_disk_forest(
    X: np.ndarray,
    tree_dir: str,
    n_estimators: int,
    n_classes: int,
    batch_size: Optional[int] = None,
    prob_buffer_mb: int = 256,
    trees_per_batch: int = 1,
    eval_strategy: str = "auto",
    tree_first_max_prob_mb: int = 2048,
    tree_prefetch: int = 1,
    tree_eval_workers: int = 1,
    log_each_tree_time: bool = False,
    desc: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with disk-saved trees using soft voting (mean predict_proba).

    This matches sklearn RandomForestClassifier prediction semantics.

    Args:
        X: Feature matrix to evaluate
        tree_dir: Directory containing tree_*.joblib files
        n_estimators: Number of trees to use
        n_classes: Number of classes (max_label + 1)
        batch_size: Sample batch size for evaluation
        prob_buffer_mb: Memory budget for auto sample batch size
        trees_per_batch: Number of trees to evaluate in parallel
        eval_strategy: 'auto', 'batch_first', or 'tree_first'
        tree_first_max_prob_mb: Max full prob-buffer size for tree_first
        tree_prefetch: Tree prefetch queue size for tree_first pipeline
        tree_eval_workers: Parallel tree workers in tree_first (shared merge)
        log_each_tree_time: Whether to log elapsed time for each tree (tree_first)
        desc: Dataset description for logs

    Returns:
        predictions: int32 array
        confidences: float32 array (max averaged class probability)
    """
    import joblib
    from joblib import Parallel, delayed

    n = len(X)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    if batch_size is None:
        batch_size = min(n, _auto_eval_batch_size(n_classes, prob_buffer_mb=prob_buffer_mb))
    else:
        batch_size = max(1, min(int(batch_size), n))
    trees_per_batch = max(1, int(trees_per_batch))

    tree_paths = [
        os.path.join(tree_dir, f'tree_{ti:04d}.joblib')
        for ti in range(n_estimators)
    ]
    for p in tree_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing tree file: {p}")

    bytes_per_value = np.dtype(np.float32).itemsize
    full_prob_mb = (n * n_classes * bytes_per_value) / (1024 ** 2)
    batch_prob_mb = (batch_size * n_classes * bytes_per_value) / (1024 ** 2)
    tree_first_cap_mb = float(tree_first_max_prob_mb)
    strategy = str(eval_strategy).strip().lower()
    if strategy not in {"auto", "batch_first", "tree_first"}:
        raise ValueError(
            f"Invalid eval_strategy={eval_strategy!r}, expected one of "
            f"['auto', 'batch_first', 'tree_first']"
        )
    if strategy == "auto":
        strategy = "tree_first" if full_prob_mb <= tree_first_cap_mb else "batch_first"
    elif strategy == "tree_first" and full_prob_mb > tree_first_cap_mb:
        log(
            "  Requested eval_strategy=tree_first but full probability buffer "
            f"needs ~{full_prob_mb:.1f}MB (> tree_first_max_prob_mb={tree_first_cap_mb:.1f}MB). "
            "Proceeding with tree_first because it was explicitly requested."
        )

    log(
        f"  Evaluating on {desc} ({n:,} samples) "
        f"[strategy={strategy}, batch_size={batch_size}, trees_per_batch={trees_per_batch}, "
        f"tree_prefetch={max(1, int(tree_prefetch))}, tree_eval_workers={max(1, int(tree_eval_workers))}, "
        f"log_each_tree_time={bool(log_each_tree_time)}, soft-vote, "
        f"batch_prob_buffer~{batch_prob_mb:.1f}MB, full_prob_buffer~{full_prob_mb:.1f}MB]"
    )

    if strategy == "tree_first":
        # Keep one full probability buffer in memory so each tree is loaded exactly once.
        all_probs = np.zeros((n, n_classes), dtype=np.float32)
        tree_progress_every = max(1, n_estimators // 20)
        n_batches = (n + batch_size - 1) // batch_size
        tree_batch_progress_every = max(1, n_batches // 10)
        prefetch = max(1, int(tree_prefetch))
        eval_workers = max(1, int(tree_eval_workers))

        def _accumulate_single_tree(
            tree_idx: int,
            tree: Any,
            classes: np.ndarray,
        ) -> None:
            elapsed_total = 0.0

            def _predict_and_accumulate(start: int, end: int) -> Tuple[float, int]:
                x_batch = X[start:end]
                t0 = time.time()
                tree_probs = tree.predict_proba(x_batch).astype(np.float32, copy=False)
                all_probs[start:end, classes] += tree_probs
                elapsed = time.time() - t0
                del x_batch, tree_probs
                return elapsed, end

            batch_ranges = [
                (start, min(start + batch_size, n))
                for start in range(0, n, batch_size)
            ]

            for batch_group_start in range(0, n_batches, eval_workers):
                current_ranges = batch_ranges[
                    batch_group_start:batch_group_start + eval_workers
                ]
                if len(current_ranges) == 1:
                    elapsed, group_end = _predict_and_accumulate(*current_ranges[0])
                    outputs = [(elapsed, group_end)]
                else:
                    outputs = Parallel(
                        n_jobs=len(current_ranges),
                        prefer="threads",
                    )(
                        delayed(_predict_and_accumulate)(start, end)
                        for start, end in current_ranges
                    )

                elapsed_total += sum(elapsed for elapsed, _ in outputs)
                batch_idx = batch_group_start + len(current_ranges)
                group_end = current_ranges[-1][1]
                if (
                    batch_group_start == 0
                    or batch_idx % tree_batch_progress_every == 0
                    or batch_idx == n_batches
                ):
                    log(
                        f"    {desc} tree {tree_idx}/{n_estimators} batch "
                        f"{batch_idx}/{n_batches} "
                        f"({group_end:,}/{n:,} samples, {100.0 * group_end / n:.1f}%)"
                    )

            if log_each_tree_time:
                log(
                    f"    {desc} tree {tree_idx}/{n_estimators} elapsed: "
                    f"{elapsed_total:.2f}s"
                )
            if (
                tree_idx == 1
                or tree_idx % tree_progress_every == 0
                or tree_idx == n_estimators
            ):
                log(
                    f"    {desc} progress: tree {tree_idx}/{n_estimators} "
                    f"({100.0 * tree_idx / n_estimators:.1f}%)"
                )

        from queue import Queue
        from threading import Thread

        # Keep the queue aligned with tree_prefetch; buffered trees already hold the active group.
        queue_capacity = min(prefetch, n_estimators)
        load_queue: Queue = Queue(maxsize=queue_capacity)

        def _loader():
            try:
                for tree_idx, tree_path in enumerate(tree_paths, start=1):
                    # Load full tree object into RAM (disable mmap paging).
                    tree = joblib.load(tree_path)
                    classes = np.asarray(tree.classes_, dtype=np.int64)
                    load_queue.put(("tree", tree_idx, tree, classes))
            except Exception as exc:
                load_queue.put(("error", exc, None, None))
            finally:
                load_queue.put(("done", None, None, None))

        loader = Thread(target=_loader, daemon=True)
        loader.start()

        processed = 0
        done_seen = False
        buffered: List[Tuple[int, Any, np.ndarray]] = []

        while processed < n_estimators:
            while len(buffered) < prefetch and not done_seen:
                msg, a, b, c = load_queue.get()
                if msg == "error":
                    raise a
                if msg == "done":
                    done_seen = True
                    break
                buffered.append((int(a), b, c))

            if not buffered:
                break

            tree_idx, tree, classes = buffered.pop(0)
            _accumulate_single_tree(tree_idx, tree, classes)
            processed += 1
            del tree, classes
            gc.collect()

        loader.join(timeout=1.0)

        all_probs /= float(n_estimators)
        predictions = np.argmax(all_probs, axis=1).astype(np.int32, copy=False)
        confidences = np.max(all_probs, axis=1).astype(np.float32, copy=False)
        del all_probs
        gc.collect()
        return predictions, confidences

    predictions = np.empty(n, dtype=np.int32)
    confidences = np.empty(n, dtype=np.float32)
    n_batches = (n + batch_size - 1) // batch_size
    progress_every = max(1, n_batches // 20)

    for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)
        batch_len = end - start
        x_batch = X[start:end]
        batch_probs = np.zeros((batch_len, n_classes), dtype=np.float32)

        def _predict_tree_probs(tree_path: str):
            # Load full tree object into RAM (disable mmap paging).
            tree = joblib.load(tree_path)
            tree_probs = tree.predict_proba(x_batch).astype(np.float32, copy=False)
            classes = np.asarray(tree.classes_, dtype=np.int64)
            del tree
            return classes, tree_probs

        for tree_start in range(0, n_estimators, trees_per_batch):
            tree_end = min(n_estimators, tree_start + trees_per_batch)
            chunk_paths = tree_paths[tree_start:tree_end]

            if len(chunk_paths) == 1:
                classes, tree_probs = _predict_tree_probs(chunk_paths[0])
                batch_probs[:, classes] += tree_probs
                del classes, tree_probs
            else:
                chunk_outputs = Parallel(n_jobs=len(chunk_paths), prefer="threads")(
                    delayed(_predict_tree_probs)(p) for p in chunk_paths
                )
                for classes, tree_probs in chunk_outputs:
                    batch_probs[:, classes] += tree_probs
                del chunk_outputs

        batch_probs /= float(n_estimators)
        predictions[start:end] = np.argmax(batch_probs, axis=1).astype(np.int32, copy=False)
        confidences[start:end] = np.max(batch_probs, axis=1).astype(np.float32, copy=False)
        del batch_probs, x_batch
        gc.collect()
        if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == n_batches:
            log(
                f"    {desc} progress: batch {batch_idx}/{n_batches} "
                f"({end:,}/{n:,} samples, {100.0 * end / n:.1f}%)"
            )

    return predictions, confidences


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    prediction_threshold: float = 0.9,
    n_jobs: int = 1,
    max_depth: Optional[int] = 30,
    progress_tree_step: int = 1,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    label_map: Dict = None,
    save_dir: str = './output',
    seed: int = 42,
    compute_train_metrics: bool = True,
    compute_feature_importance: bool = True,
    eval_batch_size: Optional[int] = None,
    eval_prob_buffer_mb: int = 256,
) -> Dict[str, Any]:
    """
    Train and evaluate Random Forest classifier (original paper approach).

    Memory-optimized: trains trees in small parallel batches and saves to disk,
    avoiding keeping the full forest in memory simultaneously.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_estimators: Number of trees
        prediction_threshold: Confidence threshold
        n_jobs: Trees trained in parallel per batch
        max_depth: Maximum depth of each tree
        progress_tree_step: Log progress every N trees (0 to disable)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        label_map: Label mapping for classification report (optional)
        save_dir: Directory to save tree files
        compute_train_metrics: Whether to evaluate full train set after fitting
        compute_feature_importance: Whether to compute averaged tree feature importances
        eval_batch_size: Fixed evaluation batch size (None = auto)
        eval_prob_buffer_mb: Probability buffer memory budget for auto batch size

    Returns:
        Dictionary with model path and metrics
    """
    import joblib
    from joblib import Parallel, delayed
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils import check_random_state

    tree_dir = os.path.join(save_dir, 'rf_trees')
    os.makedirs(tree_dir, exist_ok=True)

    n_samples = len(X_train)
    n_features = X_train.shape[1]
    # n_classes from label_map (covers all classes including train-only ones)
    if label_map is not None:
        n_classes = max(int(k) for k in label_map.keys()) + 1
    else:
        max_label = int(y_train.max())
        if y_test is not None:
            max_label = max(max_label, int(y_test.max()))
        if y_val is not None:
            max_label = max(max_label, int(y_val.max()))
        n_classes = max_label + 1

    log(f"Training Random Forest (disk-based, memory-optimized)")
    log(f"  Trees: {n_estimators}, max_depth: {max_depth}")
    log(f"  Samples: {n_samples:,}, Features: {n_features}, Classes: {n_classes:,}")
    effective_n_jobs = max(1, int(n_jobs))
    log(f"  Trees per batch: {effective_n_jobs}")
    log(f"  Tree directory: {tree_dir}")

    rng = check_random_state(seed)
    train_start = time.time()

    # =====================================================================
    # Phase 1: Train trees in parallel batches, save each to disk immediately
    # Peak memory roughly scales with trees_per_batch
    # =====================================================================
    tree_seeds = [int(rng.randint(np.iinfo(np.int32).max)) for _ in range(n_estimators)]

    def _fit_and_save_tree(tree_idx: int, tree_seed: int) -> None:
        tree_rng = check_random_state(tree_seed)

        # Bootstrap via sample_weight (avoids copying X_train)
        sample_idx = tree_rng.randint(0, n_samples, n_samples)
        sample_weight = np.bincount(sample_idx, minlength=n_samples).astype(np.float64)

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features='sqrt',
            random_state=tree_seed,
        )
        tree.fit(X_train, y_train, sample_weight=sample_weight)

        # Log process memory right before persisting this tree.
        rss_gb, rss_pct = _get_process_memory_usage()
        if np.isfinite(rss_gb) and np.isfinite(rss_pct):
            log(
                f"    Tree {tree_idx + 1}/{n_estimators} fitted | "
                f"Process RSS: {rss_gb:.2f} GB ({rss_pct:.1f}%)"
            )
        else:
            log(
                f"    Tree {tree_idx + 1}/{n_estimators} fitted | "
                f"Process RSS: unavailable"
            )

        joblib.dump(tree, os.path.join(tree_dir, f'tree_{tree_idx:04d}.joblib'))

        del tree, sample_idx, sample_weight

    built = 0
    for batch_start in range(0, n_estimators, effective_n_jobs):
        batch_end = min(n_estimators, batch_start + effective_n_jobs)
        batch_indices = range(batch_start, batch_end)
        batch_jobs = batch_end - batch_start

        Parallel(n_jobs=batch_jobs, prefer="threads")(
            delayed(_fit_and_save_tree)(idx, tree_seeds[idx])
            for idx in batch_indices
        )

        built = batch_end
        gc.collect()
        _trim_process_memory()

        if progress_tree_step > 0 and (
            built % progress_tree_step == 0 or built == n_estimators
        ):
            elapsed = time.time() - train_start
            log(
                f"  Tree {built}/{n_estimators} ({100.0 * built / n_estimators:.1f}%) "
                f"[{elapsed:.0f}s elapsed]"
            )

    train_elapsed = time.time() - train_start
    log(f"Training complete: {n_estimators} trees in {train_elapsed:.1f}s")
    _trim_process_memory()

    # --- Train metrics ---
    if compute_train_metrics:
        train_preds, _ = predict_disk_forest(
            X_train,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=eval_prob_buffer_mb,
            trees_per_batch=effective_n_jobs,
            desc="train set",
        )
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)
        log(f"Train Accuracy: {train_acc:.4f}")
        log(f"Train F1 (weighted): {train_f1:.4f}")
        del train_preds
        gc.collect()
    else:
        train_acc = None
        train_f1 = None
        log("Skipping full train-set evaluation to reduce memory/time.")

    # --- Val metrics ---
    if X_val is not None and y_val is not None:
        val_preds, _ = predict_disk_forest(
            X_val,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=eval_prob_buffer_mb,
            trees_per_batch=effective_n_jobs,
            desc="val set",
        )
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
        log(f"Val Accuracy: {val_acc:.4f}")
        log(f"Val F1 (weighted): {val_f1:.4f}")
        del val_preds
        gc.collect()
    else:
        val_acc = None
        val_f1 = None

    # --- Test metrics ---
    if X_test is not None and y_test is not None:
        test_preds, test_confidences = predict_disk_forest(
            X_test,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=eval_prob_buffer_mb,
            trees_per_batch=effective_n_jobs,
            desc="test set",
        )
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

        confident_mask = test_confidences >= prediction_threshold
        confidence_accuracy = accuracy_score(
            y_test[confident_mask], test_preds[confident_mask]
        ) if confident_mask.sum() > 0 else 0.0
        confidence_ratio = confident_mask.sum() / len(confident_mask)

        log(f"Test Accuracy: {test_acc:.4f}")
        log(f"Test F1 (weighted): {test_f1:.4f}")
        log(f"Confidence Accuracy: {confidence_accuracy:.4f} ({confidence_ratio:.1%})")

        # Classification report
        if label_map is not None:
            report_labels = sorted(label_map.keys())
            target_names = [label_map[i] for i in report_labels]
            report = classification_report(
                y_test,
                test_preds,
                labels=report_labels,
                target_names=target_names,
                zero_division=0,
            )
            log(f"\nClassification Report:\n{report}")
    else:
        test_acc = None
        test_f1 = None
        confidence_accuracy = None
        confidence_ratio = None

    # Feature importance (averaged across all trees)
    importance = None
    if compute_feature_importance:
        log("Computing feature importance...")
        _trim_process_memory()
        importance = np.zeros(n_features, dtype=np.float64)
        for i in range(n_estimators):
            tree = joblib.load(os.path.join(tree_dir, f'tree_{i:04d}.joblib'))
            importance += tree.feature_importances_
            del tree
            gc.collect()
            _trim_process_memory()
            if progress_tree_step > 0 and (
                (i + 1) % progress_tree_step == 0 or (i + 1) == n_estimators
            ):
                log(f"  Feature importance: {i + 1}/{n_estimators}")
        importance /= n_estimators
        gc.collect()
        _trim_process_memory()

        top_features = np.argsort(importance)[::-1][:10]
        log(f"Top 10 features: {top_features}")
    else:
        log("Skipping feature importance to reduce memory/time.")

    return {
        'tree_dir': tree_dir,
        'n_estimators': n_estimators,
        'n_classes': n_classes,
        'train_accuracy': train_acc,
        'train_f1': train_f1,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
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
    log("\n" + "=" * 40)
    log("Approach: Neural Network")
    log("=" * 40)

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
    log("\n" + "=" * 40)
    log("Approach: Random Forest")
    log("=" * 40)

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
    log("\n" + "=" * 60)
    log("COMPARISON SUMMARY")
    log("=" * 60)
    for name, metrics in results.items():
        log(f"{name:20s}: Acc={metrics['accuracy']:.4f}, "
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

    log("\nEngine tests passed!")
