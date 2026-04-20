"""
AppScanner Ablation Study Training Script

训练/评估主干与 `train_with_dataset.py` 保持一致，
仅保留消融实验自定义的数据集读取与划分逻辑。

实现三个消融实验：

实验1 (Baseline: 首页指纹):
    训练集: 数据集B (仅首页) 80%
    验证集: 数据集B (仅首页) 10%
    测试集: 数据集B (子页面) + 数据集A (连续会话) 10%

实验2 (Extended: 全站单页面):
    训练集: 数据集B (首页 + 子页面) 80%
    验证集: 数据集B (首页 + 子页面) 10%
    测试集: 数据集A (连续会话) 100%

实验3 (Session-based: 连续会话):
    训练集: 数据集A 80%
    验证集: 数据集A 10%
    测试集: 数据集A 10%

Usage:
    python train_ablation.py --experiment 1 --model nn
    python train_ablation.py --experiment 2 --model rf
    python train_ablation.py --experiment 3 --model deep --epochs 100
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from engine import train_random_forest
from resume_rf_evaluation import evaluate_saved_forest_splits
from train_args import TrainArgs, create_config_from_args, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "ablation_study"
OUTPUT_ROOT = SCRIPT_DIR / "checkpoints" / "ablation_study"

# 按批次划分比例 (避免数据泄露)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EXPERIMENT_NAMES = {
    1: "Baseline: Homepage Fingerprint",
    2: "Extended: Full-Site Atomic Pages",
    3: "Session-based: Aggregate Sessions",
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


def log(message: str = "") -> None:
    """Log message to both console and file."""
    logging.info(message)


def _int_or_none(value: str):
    """Parse an int argument that may also be `none`."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def get_args():
    """Build ablation-study args using shared runtime defaults."""
    defaults = TrainArgs()

    parser = argparse.ArgumentParser(description="AppScanner Ablation Study")
    parser.add_argument(
        "--experiment",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Experiment number (1: baseline, 2: extended, 3: session-based)",
    )
    parser.add_argument(
        "--model",
        "--model_type",
        dest="model_type",
        type=str,
        default="nn",
        choices=["nn", "deep", "rf"],
        help="Model type",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay, help="Weight decay")
    parser.add_argument("--patience", type=int, default=defaults.patience, help="Early-stop patience")
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=defaults.prediction_threshold,
        help="Confidence threshold",
    )
    parser.add_argument("--dropout", type=float, default=defaults.dropout, help="Dropout rate")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=list(defaults.hidden_dims),
        help="Hidden layer dimensions",
    )
    parser.add_argument("--min_flow_length", type=int, default=defaults.min_flow_length)
    parser.add_argument("--max_flow_length", type=int, default=defaults.max_flow_length)
    parser.add_argument("--n_estimators", type=int, default=defaults.n_estimators)
    parser.add_argument("--rf_max_depth", type=_int_or_none, default=defaults.rf_max_depth)
    parser.add_argument("--rf_trees_per_batch", type=int, default=defaults.rf_trees_per_batch)
    parser.add_argument("--rf_val_trees_per_batch", type=int, default=defaults.rf_val_trees_per_batch)
    parser.add_argument("--rf_test_trees_per_batch", type=int, default=defaults.rf_test_trees_per_batch)
    parser.add_argument("--rf_eval_batch_size", type=_int_or_none, default=defaults.rf_eval_batch_size)
    parser.add_argument("--rf_eval_prob_buffer_mb", type=int, default=defaults.rf_eval_prob_buffer_mb)
    parser.add_argument(
        "--rf_eval_strategy",
        type=str,
        default=defaults.rf_eval_strategy,
        choices=["auto", "batch_first", "tree_first"],
    )
    parser.add_argument(
        "--rf_tree_first_max_prob_mb",
        type=int,
        default=defaults.rf_tree_first_max_prob_mb,
    )
    parser.add_argument("--rf_tree_prefetch", type=int, default=defaults.rf_tree_prefetch)
    parser.add_argument("--rf_tree_eval_workers", type=int, default=defaults.rf_tree_eval_workers)
    parser.add_argument(
        "--rf_log_each_tree_time",
        action=argparse.BooleanOptionalAction,
        default=defaults.rf_log_each_tree_time,
    )
    parser.add_argument(
        "--rf_combine_val_test",
        action=argparse.BooleanOptionalAction,
        default=defaults.rf_combine_val_test,
    )
    parser.add_argument(
        "--rf_compute_feature_importance",
        action=argparse.BooleanOptionalAction,
        default=defaults.rf_compute_feature_importance,
    )
    parser.add_argument("--rf_progress_tree_step", type=int, default=defaults.rf_progress_tree_step)
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 4 for nn/deep, 0 for rf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )

    args = parser.parse_args()
    args.mode = "train"
    args.csv_path = None
    args.features_path = None
    args.features_paths = []
    args.checkpoint = None
    args.num_classes = None
    args.data_dir = str(DATA_DIR)

    if args.num_workers is None:
        args.num_workers = 4 if args.model_type in {"nn", "deep"} else 0

    return args


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_b(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset B (single) with homepage and subpage separation.

    Returns:
        homepage_features, homepage_labels, subpage_features, subpage_labels, label_map
    """
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    return (
        data["homepage_features"],
        data["homepage_labels"],
        data["subpage_features"],
        data["subpage_labels"],
        data["label_map"],
    )


def load_dataset_a(data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Dataset A (batch) - aggregate session data.

    Returns:
        features, labels, label_map
    """
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    return data["features"], data["labels"], data["label_map"]


def sanitize_features(features: np.ndarray, name: str) -> np.ndarray:
    """Replace NaN/Inf values and log counts."""
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    if nan_count > 0 or inf_count > 0:
        log(f"{name}: found {nan_count} NaN and {inf_count} Inf values, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def align_labels(
    dataset_b_label_map: Dict,
    dataset_a_label_map: Dict,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Align label mappings between datasets B and A.

    Returns:
        unified_label_map: Unified label mapping
        b_to_unified: Mapping from B's labels to unified labels
        a_to_unified: Mapping from A's labels to unified labels
    """
    all_websites = set(dataset_b_label_map.values()) | set(dataset_a_label_map.values())
    unified_label_map = {i: website for i, website in enumerate(sorted(all_websites))}
    website_to_unified = {website: i for i, website in unified_label_map.items()}

    b_to_unified = np.array(
        [website_to_unified[dataset_b_label_map[i]] for i in range(len(dataset_b_label_map))]
    )
    a_to_unified = np.array(
        [website_to_unified[dataset_a_label_map[i]] for i in range(len(dataset_a_label_map))]
    )

    return unified_label_map, b_to_unified, a_to_unified


def split_by_batch(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    按批次划分数据集，避免数据泄露。

    这里简化处理：假设每个类别的样本已经按时间顺序排列。
    实际应用中，应根据实际的批次ID进行划分。
    """
    np.random.seed(seed)

    unique_labels = np.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_samples = len(label_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    train_data = (features[train_indices], labels[train_indices])
    val_data = (features[val_indices], labels[val_indices])
    test_data = (features[test_indices], labels[test_indices])

    return train_data, val_data, test_data


# =============================================================================
# Experiment Data Builders
# =============================================================================

def _load_cross_domain_datasets():
    """Load dataset A/B used by experiments 1 and 2."""
    dataset_b_path = DATA_DIR / "dataset_b_single.pkl"
    dataset_a_path = DATA_DIR / "dataset_a_batch.pkl"

    if not dataset_b_path.exists():
        raise FileNotFoundError(f"Dataset B not found: {dataset_b_path}")
    if not dataset_a_path.exists():
        raise FileNotFoundError(f"Dataset A not found: {dataset_a_path}")

    log("Loading ablation datasets...")
    homepage_features, homepage_labels, subpage_features, subpage_labels, label_map_b = load_dataset_b(
        str(dataset_b_path)
    )
    aggregate_features, aggregate_labels, label_map_a = load_dataset_a(str(dataset_a_path))

    homepage_features = sanitize_features(homepage_features, "homepage_features")
    subpage_features = sanitize_features(subpage_features, "subpage_features")
    aggregate_features = sanitize_features(aggregate_features, "aggregate_features")

    unified_label_map, b_to_unified, a_to_unified = align_labels(label_map_b, label_map_a)
    homepage_labels = b_to_unified[homepage_labels]
    subpage_labels = b_to_unified[subpage_labels]
    aggregate_labels = a_to_unified[aggregate_labels]

    return (
        homepage_features,
        homepage_labels,
        subpage_features,
        subpage_labels,
        aggregate_features,
        aggregate_labels,
        unified_label_map,
    )


def build_experiment_data(experiment: int, seed: int) -> Dict[str, Any]:
    """Build train/val/test splits for the selected ablation experiment."""
    if experiment == 1:
        (
            homepage_features,
            homepage_labels,
            subpage_features,
            subpage_labels,
            aggregate_features,
            aggregate_labels,
            label_map,
        ) = _load_cross_domain_datasets()

        train_data, val_data, _ = split_by_batch(
            homepage_features,
            homepage_labels,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=seed,
        )

        test_data = (
            np.concatenate([subpage_features, aggregate_features], axis=0),
            np.concatenate([subpage_labels, aggregate_labels], axis=0),
        )

        return {
            "name": EXPERIMENT_NAMES[experiment],
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "label_map": label_map,
            "cross_domain_test_norm": True,
            "train_desc": "homepage only",
            "val_desc": "homepage only",
            "test_desc": f"subpage ({len(subpage_labels)}) + aggregate ({len(aggregate_labels)})",
        }

    if experiment == 2:
        (
            homepage_features,
            homepage_labels,
            subpage_features,
            subpage_labels,
            aggregate_features,
            aggregate_labels,
            label_map,
        ) = _load_cross_domain_datasets()

        all_b_features = np.concatenate([homepage_features, subpage_features], axis=0)
        all_b_labels = np.concatenate([homepage_labels, subpage_labels], axis=0)

        train_data, val_data, _ = split_by_batch(
            all_b_features,
            all_b_labels,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=seed,
        )

        return {
            "name": EXPERIMENT_NAMES[experiment],
            "train_data": train_data,
            "val_data": val_data,
            "test_data": (aggregate_features, aggregate_labels),
            "label_map": label_map,
            "cross_domain_test_norm": True,
            "train_desc": "homepage + subpage",
            "val_desc": "homepage + subpage",
            "test_desc": "aggregate session",
        }

    if experiment == 3:
        dataset_a_path = DATA_DIR / "dataset_a_batch.pkl"
        if not dataset_a_path.exists():
            raise FileNotFoundError(f"Dataset A not found: {dataset_a_path}")

        log("Loading ablation dataset A...")
        aggregate_features, aggregate_labels, label_map = load_dataset_a(str(dataset_a_path))
        aggregate_features = sanitize_features(aggregate_features, "aggregate_features")

        train_data, val_data, test_data = split_by_batch(
            aggregate_features,
            aggregate_labels,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=seed,
        )

        return {
            "name": EXPERIMENT_NAMES[experiment],
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "label_map": label_map,
            "cross_domain_test_norm": False,
            "train_desc": "aggregate session",
            "val_desc": "aggregate session",
            "test_desc": "aggregate session",
        }

    raise ValueError(f"Unknown experiment: {experiment}")


# =============================================================================
# Training Helpers
# =============================================================================

def _json_safe(value: Any):
    """Convert numpy/tensor-heavy objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_run_summary(
    output_dir: str,
    summary: Dict[str, Any],
) -> str:
    """Persist a compact JSON run summary."""
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(summary), file, indent=2, ensure_ascii=False)
    log(f"Run summary saved to {summary_path}")
    return summary_path


def run_neural_training(
    args,
    config,
    experiment_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run NN/Deep training using the shared `train_with_dataset.py` logic."""
    import torch

    from data import create_dataloaders_from_split
    from engine import test, train
    from models import AppScannerDeep, AppScannerNN

    train_data = experiment_data["train_data"]
    val_data = experiment_data["val_data"]
    test_data = experiment_data["test_data"]
    label_map = experiment_data["label_map"]

    train_loader, val_loader, test_loader, norm_params = create_dataloaders_from_split(
        train_data,
        val_data,
        test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        independent_test_norm=experiment_data["cross_domain_test_norm"],
    )

    if args.model_type == "nn":
        model = AppScannerNN(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
    elif args.model_type == "deep":
        model = AppScannerDeep(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dims[0],
            num_layers=4,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown neural model type: {args.model_type}")

    log(f"\nModel: {args.model_type}")
    log(f"Parameters: {sum(param.numel() for param in model.parameters()):,}")

    model, history = train(
        model,
        train_loader,
        val_loader,
        config,
        save_dir=config.output_dir,
    )

    device = torch.device(config.device)
    metrics = test(
        model,
        test_loader,
        device,
        prediction_threshold=config.prediction_threshold,
        label_map=label_map,
    )

    final_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "label_map": label_map,
            "norm_params": norm_params,
            "history": history,
            "metrics": {
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "confidence_accuracy": metrics.confidence_accuracy,
                "confidence_ratio": metrics.confidence_ratio,
            },
        },
        final_path,
    )
    log(f"\nModel saved to {final_path}")

    history_path = os.path.join(
        config.output_dir,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    log(f"Training history saved to {history_path}")

    return {
        "final_model_path": final_path,
        "history_path": history_path,
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "confidence_accuracy": metrics.confidence_accuracy,
            "confidence_ratio": metrics.confidence_ratio,
        },
    }


def run_random_forest_training(
    args,
    config,
    experiment_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run RF training using the same disk-based workflow as `train_with_dataset.py`."""
    train_features, train_labels = experiment_data["train_data"]
    val_features, val_labels = experiment_data["val_data"]
    test_features, test_labels = experiment_data["test_data"]
    label_map = experiment_data["label_map"]

    log(f"\nModel: rf (n_estimators={config.n_estimators}, max_depth={args.rf_max_depth})")
    log(f"RF train trees_per_batch: {args.rf_trees_per_batch}")
    log(f"RF val trees_per_batch: {args.rf_val_trees_per_batch}")
    log(f"RF test trees_per_batch: {args.rf_test_trees_per_batch}")
    log(f"RF eval batch_size: {args.rf_eval_batch_size}")
    log(f"RF eval prob_buffer_mb: {args.rf_eval_prob_buffer_mb}")
    log(f"RF eval strategy: {args.rf_eval_strategy}")
    log(f"RF tree_first max_prob_mb: {args.rf_tree_first_max_prob_mb}")
    log(f"RF tree prefetch: {args.rf_tree_prefetch}")
    log(f"RF tree eval workers: {args.rf_tree_eval_workers}")
    log(f"RF log each tree time: {args.rf_log_each_tree_time}")
    log(f"RF combine val+test: {args.rf_combine_val_test}")
    log(f"RF compute feature importance: {args.rf_compute_feature_importance}")

    results = train_random_forest(
        train_features,
        train_labels,
        X_test=None,
        y_test=None,
        n_estimators=config.n_estimators,
        prediction_threshold=config.prediction_threshold,
        n_jobs=args.rf_trees_per_batch,
        max_depth=args.rf_max_depth,
        progress_tree_step=args.rf_progress_tree_step,
        X_val=None,
        y_val=None,
        label_map=label_map,
        save_dir=config.output_dir,
        seed=config.seed,
        compute_train_metrics=False,
        compute_feature_importance=args.rf_compute_feature_importance,
        eval_batch_size=args.rf_eval_batch_size,
        eval_prob_buffer_mb=args.rf_eval_prob_buffer_mb,
    )

    eval_features = np.concatenate([val_features, test_features], axis=0)
    eval_labels = np.concatenate([val_labels, test_labels], axis=0)
    val_size = len(val_labels)
    val_idx = np.arange(val_size, dtype=np.int64)
    test_idx = np.arange(val_size, len(eval_labels), dtype=np.int64)

    eval_results = evaluate_saved_forest_splits(
        eval_features,
        eval_labels,
        val_idx=val_idx,
        test_idx=test_idx,
        train_features=train_features,
        train_labels=train_labels,
        tree_dir=results["tree_dir"],
        n_estimators=results["n_estimators"],
        n_classes=results["n_classes"],
        threshold=config.prediction_threshold,
        eval_batch_size=args.rf_eval_batch_size,
        prob_buffer_mb=args.rf_eval_prob_buffer_mb,
        train_trees_per_batch=args.rf_val_trees_per_batch,
        val_trees_per_batch=args.rf_val_trees_per_batch,
        test_trees_per_batch=args.rf_test_trees_per_batch,
        eval_strategy=args.rf_eval_strategy,
        tree_first_max_prob_mb=args.rf_tree_first_max_prob_mb,
        tree_prefetch=args.rf_tree_prefetch,
        tree_eval_workers=args.rf_tree_eval_workers,
        log_each_tree_time=args.rf_log_each_tree_time,
        combine_val_test=args.rf_combine_val_test,
        label_map=label_map,
        logger=log,
    )
    results.update(eval_results)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = datetime.now()
    args = get_args()

    set_seed(args.seed)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(OUTPUT_ROOT / f"experiment{args.experiment}_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logging(args.output_dir)
    experiment_data = build_experiment_data(args.experiment, args.seed)

    train_features, train_labels = experiment_data["train_data"]
    val_features, val_labels = experiment_data["val_data"]
    test_features, test_labels = experiment_data["test_data"]

    args.input_dim = int(train_features.shape[1])
    args.num_classes = len(experiment_data["label_map"])

    config = create_config_from_args(args)

    log("=" * 70)
    log("AppScanner Ablation Study")
    log("=" * 70)
    log(f"Experiment: {args.experiment} - {experiment_data['name']}")
    log(f"Model: {args.model_type}")
    log(f"Device: {config.device}")
    log(f"Output dir: {config.output_dir}")
    log(f"Data dir: {DATA_DIR}")
    log(f"Input dim: {config.input_dim}")
    log(f"Number of classes: {config.num_classes}")
    log(f"Prediction threshold: {config.prediction_threshold}")
    log(f"Split ratio target: train {TRAIN_RATIO:.2f} | val {VAL_RATIO:.2f} | test {TEST_RATIO:.2f}")
    log(f"Train samples: {len(train_labels)} ({experiment_data['train_desc']})")
    log(f"Validation samples: {len(val_labels)} ({experiment_data['val_desc']})")
    log(f"Test samples: {len(test_labels)} ({experiment_data['test_desc']})")
    log(f"Train shape: {train_features.shape}")
    log(f"Validation shape: {val_features.shape}")
    log(f"Test shape: {test_features.shape}")
    log(f"Cross-domain test normalization: {experiment_data['cross_domain_test_norm']}")
    log(f"Log file: {log_path}")

    if args.model_type == "rf":
        results = run_random_forest_training(args, config, experiment_data)
    else:
        results = run_neural_training(args, config, experiment_data)

    summary = {
        "experiment": args.experiment,
        "experiment_name": experiment_data["name"],
        "model_type": args.model_type,
        "output_dir": config.output_dir,
        "log_file": log_path,
        "input_dim": config.input_dim,
        "num_classes": config.num_classes,
        "prediction_threshold": config.prediction_threshold,
        "cross_domain_test_norm": experiment_data["cross_domain_test_norm"],
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "results": results,
    }
    save_run_summary(config.output_dir, summary)

    elapsed_time = datetime.now() - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log()
    log("=" * 70)
    log("Experiment Complete!")
    log(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    log("=" * 70)


if __name__ == "__main__":
    main()
