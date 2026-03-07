"""
AppScanner Main Training Script

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

Usage:
    python train.py
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)
    log_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


def log(message: str = ""):
    """Log message to both console and file."""
    logging.info(message)


def trim_process_memory() -> None:
    """Best-effort heap trim on glibc systems after releasing large arrays."""
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


from models import AppScannerNN, AppScannerDeep, build_model
from data import (
    create_dataset_from_directory,
    create_dataset_from_csv,
    create_dataloaders,
    save_dataset,
    load_dataset,
)
from engine import (
    train,
    test,
    train_random_forest,
    compare_approaches,
)
from resume_rf_evaluation import evaluate_saved_forest_splits
from train_args import create_config_from_args, get_args, set_seed


def load_data(args, config):
    """Load data based on arguments."""
    if args.features_path is not None:
        log(f"Loading pre-extracted features from {args.features_path}")
        features, labels, label_map = load_dataset(args.features_path)
    elif args.csv_path is not None:
        log(f"Loading features from CSV: {args.csv_path}")
        features, labels, label_map = create_dataset_from_csv(args.csv_path)
    else:
        log(f"Extracting features from PCAP files in {args.data_dir}")
        features, labels, label_map = create_dataset_from_directory(
            args.data_dir,
            min_flow_length=config.min_flow_length,
            max_flow_length=config.max_flow_length,
        )

    # Update num_classes if not specified
    if args.num_classes is None:
        config.num_classes = len(label_map)

    return features, labels, label_map


def mode_train(args, config):
    """Training mode."""
    log("=" * 60)
    log("AppScanner Training")
    log("=" * 60)

    # Load data
    features, labels, label_map = load_data(args, config)
    log(f"Features shape: {features.shape}")
    log(f"Number of classes: {config.num_classes}")

    if args.model_type == 'rf':
        # --- Random Forest branch (memory-optimized) ---
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

        # Split indices only (no data copy)
        np.random.seed(config.seed)
        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * config.train_ratio)
        n_val = int(n_samples * config.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        log(f"Training samples: {n_train}")
        log(f"Validation samples: {len(val_idx)}")
        log(f"Test samples: {len(test_idx)}")

        # Phase 1: Build X_train only, release features
        import gc
        X_train = features[train_idx]
        y_train = labels[train_idx]
        # Save split info for later phases
        data_path = args.features_path
        del features, labels, indices
        gc.collect()

        # Train (trees saved to disk in parallel batches)
        results = train_random_forest(
            X_train, y_train,
            X_test=None, y_test=None,  # defer evaluation
            n_estimators=config.n_estimators,
            prediction_threshold=config.prediction_threshold,
            n_jobs=args.rf_trees_per_batch,
            max_depth=args.rf_max_depth,
            progress_tree_step=args.rf_progress_tree_step,
            label_map=label_map,
            save_dir=config.output_dir,
            seed=config.seed,
            compute_train_metrics=False,
            compute_feature_importance=args.rf_compute_feature_importance,
            eval_batch_size=args.rf_eval_batch_size,
            eval_prob_buffer_mb=args.rf_eval_prob_buffer_mb,
        )

        # Release training data
        del X_train, y_train
        gc.collect()
        trim_process_memory()

        # Phase 2: Reload data for val/test evaluation
        log("\nReloading data for evaluation...")
        features, labels, _ = load_dataset(data_path)

        # Evaluate using disk-saved trees (soft voting)
        tree_dir = results['tree_dir']
        n_est = results['n_estimators']
        n_classes = results['n_classes']
        results['rf_eval_strategy'] = args.rf_eval_strategy
        results['rf_tree_first_max_prob_mb'] = args.rf_tree_first_max_prob_mb
        results['rf_tree_prefetch'] = args.rf_tree_prefetch
        results['rf_tree_eval_workers'] = args.rf_tree_eval_workers
        results['rf_log_each_tree_time'] = args.rf_log_each_tree_time
        results['rf_combine_val_test'] = args.rf_combine_val_test
        results['rf_val_trees_per_batch'] = args.rf_val_trees_per_batch
        results['rf_test_trees_per_batch'] = args.rf_test_trees_per_batch
        eval_results = evaluate_saved_forest_splits(
            features,
            labels,
            val_idx=val_idx,
            test_idx=test_idx,
            tree_dir=tree_dir,
            n_estimators=n_est,
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
        del features, labels, val_idx, test_idx
        gc.collect()

        return results

    else:
        # --- NN / Deep branch ---
        # Create dataloaders with 8:1:1 split (train:val:test)
        train_loader, val_loader, test_loader, norm_params = create_dataloaders(
            features, labels,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        input_dim = features.shape[1]

        # Create model
        if args.model_type == 'nn':
            model = AppScannerNN(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        elif args.model_type == 'deep':
            model = AppScannerDeep(
                input_dim=input_dim,
                num_classes=config.num_classes,
                hidden_dim=config.hidden_dims[0],
                num_layers=4,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        log(f"\nModel: {args.model_type}")
        log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        model, history = train(
            model, train_loader, val_loader, config,
            save_dir=config.output_dir,
        )

        # Test
        device = torch.device(config.device)
        metrics = test(
            model, test_loader, device,
            prediction_threshold=config.prediction_threshold,
            label_map=label_map,
        )

        # Save final model and metadata
        final_path = os.path.join(config.output_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'label_map': label_map,
            'norm_params': norm_params,
            'history': history,
            'metrics': {
                'accuracy': metrics.accuracy,
                'f1': metrics.f1,
                'confidence_accuracy': metrics.confidence_accuracy,
            },
        }, final_path)
        log(f"\nModel saved to {final_path}")

        # Save training history
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(config.output_dir, f'history_{time}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        log(f"Training history saved to {history_path}")

        return model, metrics


def mode_eval(args, config):
    """Evaluation mode."""
    log("=" * 60)
    log("AppScanner Evaluation")
    log("=" * 60)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(config.output_dir, 'best_model.pth')

    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)

    # Load config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.num_classes = saved_config.num_classes
        config.hidden_dims = saved_config.hidden_dims

    # Load data
    features, labels, label_map = load_data(args, config)

    # Create test loader (use all data for testing)
    _, _, test_loader, _ = create_dataloaders(
        features, labels,
        batch_size=config.batch_size,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        num_workers=config.num_workers,
    )

    # Create and load model
    model = AppScannerNN(
        input_dim=features.shape[1],
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    device = torch.device(config.device)
    metrics = test(
        model, test_loader, device,
        prediction_threshold=config.prediction_threshold,
        label_map=label_map,
    )

    return metrics


def mode_extract(args, config):
    """Feature extraction mode."""
    log("=" * 60)
    log("AppScanner Feature Extraction")
    log("=" * 60)

    # Extract features
    features, labels, label_map = create_dataset_from_directory(
        args.data_dir,
        min_flow_length=config.min_flow_length,
        max_flow_length=config.max_flow_length,
    )

    # Save
    output_path = args.features_path or os.path.join(config.output_dir, 'features.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dataset(features, labels, label_map, output_path)

    log(f"Features shape: {features.shape}")
    log(f"Number of classes: {len(label_map)}")
    log(f"Saved to: {output_path}")

    return features, labels, label_map


def mode_compare(args, config):
    """Compare different approaches."""
    log("=" * 60)
    log("AppScanner Approach Comparison")
    log("=" * 60)

    # Load data
    features, labels, label_map = load_data(args, config)

    # Split data
    np.random.seed(config.seed)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * config.test_ratio)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    log(f"Train samples: {len(y_train)}")
    log(f"Test samples: {len(y_test)}")

    # Compare approaches
    results = compare_approaches(
        X_train, y_train, X_test, y_test, config
    )

    # Save results
    results_path = os.path.join(config.output_dir, 'comparison_results.json')
    os.makedirs(config.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")

    return results


def main():
    # 记录开始时间
    start_time = datetime.now()

    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Create base config
    config = create_config_from_args(args)
    base_output_dir = config.output_dir

    log("=" * 70)
    log("AppScanner Multi-Dataset Training")
    log("=" * 70)
    log(f"Datasets to run: {len(args.features_paths)}")
    log(f"Model: {args.model_type}")
    log()

    for i, data_path in enumerate(args.features_paths, 1):
        dataset_start = datetime.now()
        dataset_name = Path(data_path).stem.replace('_appscanner', '')

        # Set current dataset
        args.features_path = data_path
        args.num_classes = None  # Reset for auto-detect
        config.output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup per-dataset logging
        log_path = setup_logging(config.output_dir)

        log(f"\n{'=' * 70}")
        log(f"[{i}/{len(args.features_paths)}] Dataset: {dataset_name}")
        log(f"{'=' * 70}")
        log(f"\nConfiguration:")
        log(f"  Start time: {dataset_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"  Mode: {args.mode}")
        log(f"  Model: {args.model_type}")
        log(f"  Data: {data_path}")
        log(f"  Device: {config.device}")
        log(f"  Prediction threshold: {config.prediction_threshold}")
        if args.model_type == 'rf':
            log(f"  RF max_depth: {args.rf_max_depth}")
            log(f"  RF train trees_per_batch: {args.rf_trees_per_batch}")
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
            log(f"  RF progress_tree_step: {args.rf_progress_tree_step}")
            log(f"  RF compute feature importance: {args.rf_compute_feature_importance}")
        log(f"  Log file: {log_path}")
        log()

        try:
            # Run mode
            if args.mode == 'train':
                mode_train(args, config)
            elif args.mode == 'eval':
                mode_eval(args, config)
            elif args.mode == 'extract':
                mode_extract(args, config)
            elif args.mode == 'compare':
                mode_compare(args, config)

            dataset_end = datetime.now()
            elapsed = dataset_end - dataset_start
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            log(f"\n[{dataset_name}] Completed in {h:02d}:{m:02d}:{s:02d}")

        except Exception as e:
            log(f"\n[{dataset_name}] FAILED: {e}")
            import traceback
            log(traceback.format_exc())

    # 总计用时
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    log()
    log("=" * 70)
    log(f"All {len(args.features_paths)} datasets completed.")
    log(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    log("=" * 70)

if __name__ == '__main__':
    main()
