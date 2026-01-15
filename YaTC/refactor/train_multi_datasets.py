"""
YaTC Multi-Dataset Concurrent Training Script

并发训练多个 Top-K 数据集，最多同时运行 MAX_WORKERS 个训练任务。
使用 split_topk_classes.py 处理后的数据集进行微调。

训练参数（epochs, batch_size, lr 等）在 refactor/finetune.py 中硬编码配置。

Usage:
    python train_multi_datasets.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Single dataset configuration."""
    name: str                    # Dataset name (for output directory)
    train_path: str              # Path to train directory (contains NPZ files)
    val_path: str                # Path to val directory
    test_path: str               # Path to test directory
    num_classes: Optional[int] = None  # Auto-detect from labels.json if None
    pretrained: Optional[str] = None   # Pre-trained model path (override global)


# 数据根目录
DATA_ROOT = '/root/autodl-tmp/YaTc/data'

# 预训练模型路径（全局默认）
PRETRAINED_MODEL = './output_dir/pretrained.pth'

# 配置要训练的数据集列表
# 使用 split_topk_classes.py 处理后的数据集目录
DATASETS: List[DatasetConfig] = [
    # VPN Top-K datasets
    DatasetConfig(
        name='novpn_top1000',
        train_path=f'{DATA_ROOT}/novpn_top1000_split/train',
        val_path=f'{DATA_ROOT}/novpn_top1000_split/val', 
        test_path=f'{DATA_ROOT}/novpn_top1000_split/test', 
    ),
    DatasetConfig(
        name='vpn_top1000',
        train_path=f'{DATA_ROOT}/vpn_top1000_split/train',
        val_path=f'{DATA_ROOT}/vpn_top1000_split/val', 
        test_path=f'{DATA_ROOT}/vpn_top1000_split/test', 
    ),
    DatasetConfig(
        name='novpn_top500',
        train_path=f'{DATA_ROOT}/novpn_top500_split/train',
        val_path=f'{DATA_ROOT}/novpn_top500_split/val', 
        test_path=f'{DATA_ROOT}/novpn_top500_split/test', 
    ),
    DatasetConfig(
        name='vpn_top500',
        train_path=f'{DATA_ROOT}/vpn_top500_split/train',
        val_path=f'{DATA_ROOT}/vpn_top500_split/val', 
        test_path=f'{DATA_ROOT}/vpn_top500_split/test', 
    ),
    DatasetConfig(
        name='novpn_top100',
        train_path=f'{DATA_ROOT}/novpn_top100_split/train',
        val_path=f'{DATA_ROOT}/novpn_top100_split/val', 
        test_path=f'{DATA_ROOT}/novpn_top100_split/test', 
    ),
    DatasetConfig(
        name='vpn_top100',
        train_path=f'{DATA_ROOT}/vpn_top100_split/train',
        val_path=f'{DATA_ROOT}/vpn_top100_split/val', 
        test_path=f'{DATA_ROOT}/vpn_top100_split/test', 
    ),
    DatasetConfig(
        name='novpn_top50',
        train_path=f'{DATA_ROOT}/novpn_top50_split/train',
        val_path=f'{DATA_ROOT}/novpn_top50_split/val', 
        test_path=f'{DATA_ROOT}/novpn_top50_split/test', 
    ),
    DatasetConfig(
        name='vpn_top50',
        train_path=f'{DATA_ROOT}/vpn_top50_split/train',
        val_path=f'{DATA_ROOT}/vpn_top50_split/val', 
        test_path=f'{DATA_ROOT}/vpn_top50_split/test', 
    ),
    DatasetConfig(
        name='novpn_top10',
        train_path=f'{DATA_ROOT}/novpn_top10_split/train',
        val_path=f'{DATA_ROOT}/novpn_top10_split/val', 
        test_path=f'{DATA_ROOT}/novpn_top10_split/test', 
    ),
    DatasetConfig(
        name='vpn_top10',
        train_path=f'{DATA_ROOT}/vpn_top10_split/train',
        val_path=f'{DATA_ROOT}/vpn_top10_split/val', 
        test_path=f'{DATA_ROOT}/vpn_top10_split/test', 
    ),
]

# 并发数量（根据 GPU 数量和显存调整）
MAX_WORKERS = 3

# 输出根目录
OUTPUT_ROOT = './YaTC/output_multi'


# =============================================================================
# Helper Functions
# =============================================================================

def get_num_classes_from_labels(data_path: str) -> Optional[int]:
    """
    从 labels.json 读取类别数量。

    Args:
        data_path: 数据目录路径（包含 labels.json）

    Returns:
        类别数量，如果读取失败返回 None
    """
    import json
    labels_json = Path(data_path) / 'labels.json'
    if labels_json.exists():
        try:
            with open(labels_json, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta.get('num_classes') or len(meta.get('label2id', {}))
        except Exception:
            pass
    return None


def check_dataset_exists(config: DatasetConfig) -> bool:
    """检查数据集目录是否存在。"""
    return (
        Path(config.train_path).exists() and
        Path(config.val_path).exists() and
        Path(config.test_path).exists()
    )


# =============================================================================
# Training Function
# =============================================================================

def train_single_dataset(config: DatasetConfig) -> dict:
    """
    Train a single dataset by calling refactor/finetune.py.

    Returns dict with status and metrics.
    """
    start_time = datetime.now()
    output_dir = os.path.join(OUTPUT_ROOT, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result
    result = {
        'name': config.name,
        'status': 'failed',
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'elapsed': '00:00:00',
        'error': None,
        'num_classes': 0
    }

    # Check if dataset exists
    if not check_dataset_exists(config):
        result['error'] = f'Dataset not found: {config.train_path}'
        return result

    # Get num_classes from train directory
    num_classes = config.num_classes
    if num_classes is None:
        num_classes = get_num_classes_from_labels(config.train_path)

    if num_classes is None:
        result['error'] = 'Could not determine num_classes'
        return result

    result['num_classes'] = num_classes

    # Build command line arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'finetune.py')

    # Use dataset-specific or global pretrained model
    pretrained = config.pretrained or PRETRAINED_MODEL

    # 从 train_path 推导父目录（去掉 /train 后缀）
    data_root = os.path.dirname(config.train_path)

    cmd = [
        sys.executable, train_script,
        '--data_path', data_root
    ]

    try:
        process = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Save full output to log file
        log_path = os.path.join(output_dir, 'subprocess_output.log')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")

        # Parse output for metrics
        output = process.stdout
        if process.returncode == 0:
            # Parse metrics from output
            # Look for lines like "Accuracy:  0.9876" or "Test Accuracy: 0.9876"
            for line in output.split('\n'):
                line = line.strip()
                if 'Accuracy:' in line and 'Best' not in line:
                    try:
                        result['accuracy'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Precision:' in line:
                    try:
                        result['precision'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Recall:' in line:
                    try:
                        result['recall'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'F1 Score:' in line or 'F1:' in line:
                    try:
                        result['f1'] = float(line.split(':')[-1].strip())
                    except:
                        pass

            result['status'] = 'success'
            if result['accuracy'] == 0 and result['f1'] == 0:
                result['error'] = 'Could not parse metrics from output'
        else:
            result['error'] = process.stderr or f'Process exited with code {process.returncode}'

    except Exception as e:
        result['error'] = str(e)
        # Save error log
        with open(os.path.join(output_dir, 'error.log'), 'w', encoding='utf-8') as f:
            f.write(f"Exception: {str(e)}")

    # Calculate elapsed time
    end_time = datetime.now()
    elapsed = end_time - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    result['elapsed'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = datetime.now()

    print("=" * 80)
    print("YaTC Multi-Dataset Concurrent Training")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Pre-trained model: {PRETRAINED_MODEL}")
    print()

    # Filter datasets that exist
    valid_datasets = []
    print("Checking datasets...")
    for ds in DATASETS:
        if check_dataset_exists(ds):
            num_classes = ds.num_classes or get_num_classes_from_labels(ds.train_path)
            print(f"  [OK] {ds.name}: {num_classes} classes")
            valid_datasets.append(ds)
        else:
            print(f"  [SKIP] {ds.name}: not found")

    if not valid_datasets:
        print("\nNo valid datasets found. Exiting.")
        return

    print(f"\nDatasets to train: {len(valid_datasets)}")
    print(f"Training parameters are configured in refactor/finetune.py")
    print()

    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Run training concurrently
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(train_single_dataset, config): config
            for config in valid_datasets
        }

        # Process completed tasks
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)

                status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
                print(f"{status_icon} {result['name']:20s} | "
                      f"Classes: {result['num_classes']:4d} | "
                      f"Acc: {result['accuracy']:.4f} | "
                      f"F1: {result['f1']:.4f} | "
                      f"Time: {result['elapsed']}")

            except Exception as e:
                print(f"[ERROR] {config.name}: {e}")
                results.append({
                    'name': config.name,
                    'status': 'error',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'elapsed': '00:00:00',
                    'error': str(e),
                    'num_classes': 0
                })

    # Summary
    end_time = datetime.now()
    elapsed = end_time - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Sort results by name for consistent output
    results.sort(key=lambda x: x['name'])

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"{'Dataset':<20} {'Classes':>8} {'Status':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>12}")
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<20} {r['num_classes']:>8} {r['status']:<10} "
              f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['elapsed']:>12}")

    print("-" * 100)
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")

    # Save summary
    summary_path = os.path.join(OUTPUT_ROOT, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("YaTC Multi-Dataset Training Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        f.write(f"Pre-trained model: {PRETRAINED_MODEL}\n")
        f.write(f"Training parameters: configured in refactor/finetune.py\n\n")

        f.write(f"{'Dataset':<20} {'Classes':>8} {'Status':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>12}\n")
        f.write("-" * 100 + "\n")
        for r in results:
            f.write(f"{r['name']:<20} {r['num_classes']:>8} {r['status']:<10} "
                    f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                    f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['elapsed']:>12}\n")
        f.write("-" * 100 + "\n")

        # Write errors if any
        errors = [r for r in results if r.get('error')]
        if errors:
            f.write("\nErrors:\n")
            for r in errors:
                f.write(f"  {r['name']}: {r['error']}\n")

    print(f"\nSummary saved to: {summary_path}")

    # Save results as CSV for easy analysis
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("dataset,num_classes,status,accuracy,precision,recall,f1,elapsed,error\n")
        for r in results:
            error_str = r.get('error', '').replace(',', ';').replace('\n', ' ') if r.get('error') else ''
            f.write(f"{r['name']},{r['num_classes']},{r['status']},"
                    f"{r['accuracy']:.4f},{r['precision']:.4f},"
                    f"{r['recall']:.4f},{r['f1']:.4f},"
                    f"{r['elapsed']},{error_str}\n")

    print(f"Results CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
