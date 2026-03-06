"""
Resume evaluation for disk-based Random Forest trees.

This script reuses trees saved under `rf_trees` and evaluates on val/test splits
without retraining.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Ensure local AppScanner modules are importable when invoked from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data import load_dataset
from engine import predict_disk_forest


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"resume_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


def log(message: str = "") -> None:
    """Log message to both console and file."""
    logging.info(message)


# =============================================================================
# Configuration (edit these values directly)
# =============================================================================

DATA_PATH = "/home/pcz/code/DL/AppScanner/data/novpn/novpn_appscanner.pkl"
TREE_DIR = "/home/pcz/code/DL/AppScanner/output/novpn/rf_trees"
SEED = 42  # 随机种子：用于可复现的数据划分（train/val/test）。
TRAIN_RATIO = 0.8  # 训练集比例：总样本中划给训练集的占比。
VAL_RATIO = 0.1  # 验证集比例：总样本中划给验证集的占比。
THRESHOLD = 0.9  # 置信度阈值：高于该概率的预测计入 confidence_accuracy。
VAL_TREES_PER_BATCH = 10  # 验证集评估时 batch_first 的并行树数（与 tree_eval_workers 对齐）。
TEST_TREES_PER_BATCH = 10  # 测试集评估时 batch_first 的并行树数（与 tree_eval_workers 对齐）。
EVAL_BATCH_SIZE = None  # 样本批大小：None 表示按 PROB_BUFFER_MB 自动估算。
PROB_BUFFER_MB = 256  # 自动批大小的概率缓冲预算（MB），控制每批内存占用。
EVAL_STRATEGY = "tree_first"  # 评估策略：固定 tree_first（树外层，一次加载整棵树）。
TREE_FIRST_MAX_PROB_MB = 4096  # auto 选择 tree_first 的上限：全量概率矩阵估算内存不超过该值才启用。
TREE_PREFETCH = 1  # tree_first 预加载队列长度：评估当前树时后台预加载后续树（如 25）。
TREE_EVAL_WORKERS = 10  # tree_first 评估并行树数（每批最多同时计算 K 棵树）。
LOG_EACH_TREE_TIME = True  # 是否打印每棵树的评估耗时（tree_first 模式）。
COMBINE_VAL_TEST = True
OUTPUT_JSON = "/home/pcz/code/DL/AppScanner/output/novpn/resume_eval_metrics.json"


def _find_tree_indices(tree_dir: str) -> List[int]:
    """Return sorted tree indices from files named tree_XXXX.joblib."""
    indices: List[int] = []
    for name in os.listdir(tree_dir):
        if not name.startswith("tree_") or not name.endswith(".joblib"):
            continue
        idx_part = name[len("tree_") : -len(".joblib")]
        if idx_part.isdigit():
            indices.append(int(idx_part))
    return sorted(indices)


def _validate_trees(tree_dir: str) -> int:
    """Validate tree files are contiguous from 0..N-1 and return N."""
    if not os.path.isdir(tree_dir):
        raise FileNotFoundError(f"Tree directory does not exist: {tree_dir}")

    indices = _find_tree_indices(tree_dir)
    if not indices:
        raise FileNotFoundError(f"No tree_*.joblib files found in: {tree_dir}")

    n_estimators = len(indices)
    expected = list(range(n_estimators))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        raise RuntimeError(
            f"Tree files are not contiguous 0..{n_estimators - 1}. "
            f"Missing={missing[:10]}, Extra={extra[:10]}"
        )
    return n_estimators


def _label_map_num_classes(label_map: Dict, labels: np.ndarray) -> int:
    """Compute num classes robustly from label_map and labels."""
    if label_map:
        try:
            keys = [int(k) for k in label_map.keys()]
            return max(keys) + 1
        except Exception:
            pass
    return int(labels.max()) + 1


def _classification_targets(label_map: Dict) -> Tuple[List[int], List[str]]:
    """Build sorted integer labels and target names from label_map keys."""
    pairs: List[Tuple[int, str]] = []
    for key, value in label_map.items():
        try:
            pairs.append((int(key), str(value)))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    labels = [x[0] for x in pairs]
    names = [x[1] for x in pairs]
    return labels, names


def evaluate_saved_forest_splits(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    tree_dir: str,
    n_estimators: int,
    n_classes: int,
    threshold: float,
    eval_batch_size: Optional[int] = None,
    prob_buffer_mb: int = 256,
    val_trees_per_batch: int = 10,
    test_trees_per_batch: int = 10,
    eval_strategy: str = "tree_first",
    tree_first_max_prob_mb: int = 4096,
    tree_prefetch: int = 1,
    tree_eval_workers: int = 10,
    log_each_tree_time: bool = True,
    combine_val_test: bool = True,
    label_map: Optional[Dict] = None,
    logger: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Evaluate saved RF trees on val/test splits using shared logic."""
    if features.dtype != np.float32:
        logger(f"Casting features to float32 from {features.dtype} for faster RF inference...")
        features = features.astype(np.float32, copy=False)
    if not features.flags.c_contiguous:
        features = np.ascontiguousarray(features)

    results: Dict[str, Any] = {}
    has_val = len(val_idx) > 0
    X_test, y_test = features[test_idx], labels[test_idx]
    test_preds = None
    test_conf = None

    if has_val and combine_val_test:
        X_val, y_val = features[val_idx], labels[val_idx]
        X_eval = np.concatenate([X_val, X_test], axis=0)
        combined_trees_per_batch = max(1, min(int(val_trees_per_batch), int(test_trees_per_batch)))
        eval_preds, eval_conf = predict_disk_forest(
            X_eval,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=prob_buffer_mb,
            trees_per_batch=combined_trees_per_batch,
            eval_strategy=eval_strategy,
            tree_first_max_prob_mb=tree_first_max_prob_mb,
            tree_prefetch=tree_prefetch,
            tree_eval_workers=tree_eval_workers,
            log_each_tree_time=log_each_tree_time,
            desc="val+test set",
        )
        val_size = len(X_val)
        val_preds = eval_preds[:val_size]
        test_preds = eval_preds[val_size:]
        test_conf = eval_conf[val_size:]
        del X_eval, eval_preds, eval_conf

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        logger(f"Val Accuracy: {val_acc:.4f}")
        logger(f"Val F1 (weighted): {val_f1:.4f}")
        results["val_accuracy"] = float(val_acc)
        results["val_f1"] = float(val_f1)
    elif has_val:
        X_val, y_val = features[val_idx], labels[val_idx]
        val_preds, _ = predict_disk_forest(
            X_val,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=prob_buffer_mb,
            trees_per_batch=val_trees_per_batch,
            eval_strategy=eval_strategy,
            tree_first_max_prob_mb=tree_first_max_prob_mb,
            tree_prefetch=tree_prefetch,
            tree_eval_workers=tree_eval_workers,
            log_each_tree_time=log_each_tree_time,
            desc="val set",
        )
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        logger(f"Val Accuracy: {val_acc:.4f}")
        logger(f"Val F1 (weighted): {val_f1:.4f}")
        results["val_accuracy"] = float(val_acc)
        results["val_f1"] = float(val_f1)
    else:
        logger("Validation split is empty; skipping val evaluation.")

    if test_preds is None or test_conf is None:
        test_preds, test_conf = predict_disk_forest(
            X_test,
            tree_dir=tree_dir,
            n_estimators=n_estimators,
            n_classes=n_classes,
            batch_size=eval_batch_size,
            prob_buffer_mb=prob_buffer_mb,
            trees_per_batch=test_trees_per_batch,
            eval_strategy=eval_strategy,
            tree_first_max_prob_mb=tree_first_max_prob_mb,
            tree_prefetch=tree_prefetch,
            tree_eval_workers=tree_eval_workers,
            log_each_tree_time=log_each_tree_time,
            desc="test set",
        )
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    results["test_accuracy"] = float(test_acc)
    results["test_f1"] = float(test_f1)

    confident_mask = test_conf >= threshold
    results["confidence_ratio"] = float(confident_mask.mean())
    if confident_mask.sum() > 0:
        results["confidence_accuracy"] = float(
            accuracy_score(y_test[confident_mask], test_preds[confident_mask])
        )
    else:
        results["confidence_accuracy"] = 0.0

    logger(f"Test Accuracy: {results['test_accuracy']:.4f}")
    logger(f"Test F1 (weighted): {results['test_f1']:.4f}")
    logger(
        "Confidence Accuracy: "
        f"{results['confidence_accuracy']:.4f} ({results['confidence_ratio']:.1%})"
    )

    if label_map is not None:
        report_labels, target_names = _classification_targets(label_map)
        if report_labels:
            report = classification_report(
                y_test,
                test_preds,
                labels=report_labels,
                target_names=target_names,
                zero_division=0,
            )
            logger(f"\nClassification Report:\n{report}")

    return results


def main() -> None:
    if OUTPUT_JSON:
        log_output_dir = str(Path(OUTPUT_JSON).resolve().parent)
    else:
        log_output_dir = str(Path(TREE_DIR).resolve().parent)
    log_path = setup_logging(log_output_dir)

    log("=" * 60)
    log("AppScanner Resume RF Evaluation")
    log("=" * 60)
    log(f"Log file: {log_path}")

    n_estimators = _validate_trees(TREE_DIR)

    log(f"Loading dataset: {DATA_PATH}")
    features, labels, label_map = load_dataset(DATA_PATH)
    n_samples = len(labels)
    n_classes = _label_map_num_classes(label_map, labels)
    log(f"Dataset shape: {features.shape}, classes: {n_classes}, trees: {n_estimators}")

    np.random.seed(SEED)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    log(f"Training samples: {len(train_idx)}")
    log(f"Validation samples: {len(val_idx)}")
    log(f"Test samples: {len(test_idx)}")

    results = {
        "n_estimators_used": n_estimators,
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "seed": int(SEED),
        "train_ratio": float(TRAIN_RATIO),
        "val_ratio": float(VAL_RATIO),
        "test_ratio": float(1.0 - TRAIN_RATIO - VAL_RATIO),
        "threshold": float(THRESHOLD),
        "eval_strategy": str(EVAL_STRATEGY),
        "tree_first_max_prob_mb": int(TREE_FIRST_MAX_PROB_MB),
        "tree_prefetch": int(TREE_PREFETCH),
        "tree_eval_workers": int(TREE_EVAL_WORKERS),
        "log_each_tree_time": bool(LOG_EACH_TREE_TIME),
        "combine_val_test": bool(COMBINE_VAL_TEST),
        "val_trees_per_batch": int(VAL_TREES_PER_BATCH),
        "test_trees_per_batch": int(TEST_TREES_PER_BATCH),
        "log_file": str(log_path),
    }
    eval_results = evaluate_saved_forest_splits(
        features,
        labels,
        val_idx=val_idx,
        test_idx=test_idx,
        tree_dir=TREE_DIR,
        n_estimators=n_estimators,
        n_classes=n_classes,
        threshold=THRESHOLD,
        eval_batch_size=EVAL_BATCH_SIZE,
        prob_buffer_mb=PROB_BUFFER_MB,
        val_trees_per_batch=VAL_TREES_PER_BATCH,
        test_trees_per_batch=TEST_TREES_PER_BATCH,
        eval_strategy=EVAL_STRATEGY,
        tree_first_max_prob_mb=TREE_FIRST_MAX_PROB_MB,
        tree_prefetch=TREE_PREFETCH,
        tree_eval_workers=TREE_EVAL_WORKERS,
        log_each_tree_time=LOG_EACH_TREE_TIME,
        combine_val_test=COMBINE_VAL_TEST,
        label_map=label_map,
        logger=log,
    )
    results.update(eval_results)

    if OUTPUT_JSON:
        out_path = Path(OUTPUT_JSON)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"Saved metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
