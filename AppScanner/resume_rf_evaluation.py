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
from train_args import create_config_from_args, get_args


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"resume_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)
    log_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
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


DEFAULT_OUTPUT_JSON_NAME = "resume_eval_metrics.json"


def _dataset_name(data_path: str) -> str:
    return Path(data_path).stem.replace("_appscanner", "")


def _build_run_specs() -> Tuple[Any, Any, List[Dict[str, str]]]:
    """Reuse the shared train/runtime defaults so resume-eval stays in sync."""
    args = get_args()
    config = create_config_from_args(args)
    base_output_dir = Path(config.output_dir)
    run_specs: List[Dict[str, str]] = []

    for data_path in args.features_paths:
        dataset_name = _dataset_name(data_path)
        dataset_output_dir = base_output_dir / dataset_name
        run_specs.append(
            {
                "dataset_name": dataset_name,
                "data_path": data_path,
                "output_dir": str(dataset_output_dir),
                "tree_dir": str(dataset_output_dir / "rf_trees"),
                "output_json": str(dataset_output_dir / DEFAULT_OUTPUT_JSON_NAME),
            }
        )

    return args, config, run_specs


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
    start_time = datetime.now()
    args, config, run_specs = _build_run_specs()

    for i, run in enumerate(run_specs, start=1):
        dataset_start = datetime.now()
        log_path = setup_logging(run["output_dir"])

        log(f"\n{'=' * 70}")
        log(f"[{i}/{len(run_specs)}] Dataset: {run['dataset_name']}")
        log(f"{'=' * 70}")
        log("Configuration:")
        log(f"  Start time: {dataset_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"  Data: {run['data_path']}")
        log(f"  Tree dir: {run['tree_dir']}")
        log(f"  Output dir: {run['output_dir']}")
        log(f"  Seed: {config.seed}")
        log(f"  Train ratio: {config.train_ratio}")
        log(f"  Val ratio: {config.val_ratio}")
        log(f"  Test ratio: {config.test_ratio}")
        log(f"  Prediction threshold: {config.prediction_threshold}")
        log(f"  RF val trees_per_batch: {args.rf_val_trees_per_batch}")
        log(f"  RF test trees_per_batch: {args.rf_test_trees_per_batch}")
        log(f"  RF eval batch_size: {args.rf_eval_batch_size}")
        log(f"  RF eval prob_buffer_mb: {args.rf_eval_prob_buffer_mb}")
        log(f"  RF eval strategy: {args.rf_eval_strategy}")
        log(f"  RF tree_first max_prob_mb: {args.rf_tree_first_max_prob_mb}")
        log(f"  RF tree prefetch: {args.rf_tree_prefetch}")
        log(f"  RF tree eval workers: {args.rf_tree_eval_workers}")
        log(f"  RF log each tree time: {args.rf_log_each_tree_time}")
        log(f"  RF combine val+test: {args.rf_combine_val_test}")
        log(f"  Metrics JSON: {run['output_json']}")
        log(f"  Log file: {log_path}")

        try:
            n_estimators = _validate_trees(run["tree_dir"])

            log(f"Loading dataset: {run['data_path']}")
            features, labels, label_map = load_dataset(run["data_path"])
            n_samples = len(labels)
            n_classes = _label_map_num_classes(label_map, labels)
            log(f"Dataset shape: {features.shape}, classes: {n_classes}, trees: {n_estimators}")

            np.random.seed(config.seed)
            indices = np.random.permutation(n_samples)
            n_train = int(n_samples * config.train_ratio)
            n_val = int(n_samples * config.val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            log(f"Training samples: {len(train_idx)}")
            log(f"Validation samples: {len(val_idx)}")
            log(f"Test samples: {len(test_idx)}")

            results = {
                "dataset_name": run["dataset_name"],
                "data_path": run["data_path"],
                "tree_dir": run["tree_dir"],
                "n_estimators_used": n_estimators,
                "n_samples": int(n_samples),
                "n_classes": int(n_classes),
                "seed": int(config.seed),
                "train_ratio": float(config.train_ratio),
                "val_ratio": float(config.val_ratio),
                "test_ratio": float(config.test_ratio),
                "threshold": float(config.prediction_threshold),
                "eval_strategy": str(args.rf_eval_strategy),
                "tree_first_max_prob_mb": int(args.rf_tree_first_max_prob_mb),
                "tree_prefetch": int(args.rf_tree_prefetch),
                "tree_eval_workers": int(args.rf_tree_eval_workers),
                "log_each_tree_time": bool(args.rf_log_each_tree_time),
                "combine_val_test": bool(args.rf_combine_val_test),
                "val_trees_per_batch": int(args.rf_val_trees_per_batch),
                "test_trees_per_batch": int(args.rf_test_trees_per_batch),
                "eval_batch_size": (
                    None if args.rf_eval_batch_size is None else int(args.rf_eval_batch_size)
                ),
                "eval_prob_buffer_mb": int(args.rf_eval_prob_buffer_mb),
                "metrics_json": run["output_json"],
                "log_file": str(log_path),
            }
            eval_results = evaluate_saved_forest_splits(
                features,
                labels,
                val_idx=val_idx,
                test_idx=test_idx,
                tree_dir=run["tree_dir"],
                n_estimators=n_estimators,
                n_classes=n_classes,
                threshold=config.prediction_threshold,
                eval_batch_size=args.rf_eval_batch_size,
                prob_buffer_mb=args.rf_eval_prob_buffer_mb,
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

            out_path = Path(run["output_json"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            log(f"Saved metrics JSON: {out_path}")

            dataset_end = datetime.now()
            elapsed = dataset_end - dataset_start
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            log(f"\n[{run['dataset_name']}] Completed in {h:02d}:{m:02d}:{s:02d}")
        except Exception as e:
            log(f"\n[{run['dataset_name']}] FAILED: {e}")
            import traceback

            log(traceback.format_exc())

    end_time = datetime.now()
    elapsed = end_time - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    log()
    log("=" * 70)
    log(f"All {len(run_specs)} datasets completed.")
    log(f"Total time: {h:02d}:{m:02d}:{s:02d}")
    log("=" * 70)


if __name__ == "__main__":
    main()
