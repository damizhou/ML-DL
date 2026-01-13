"""
AppScanner Multi-Dataset Concurrent Training Script

并发训练多个数据集，最多同时运行 3 个训练任务。

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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Single dataset configuration."""
    name: str                    # Dataset name (for display)
    data_path: str               # Path to pickle file


# 配置要训练的数据集列表
DATASETS: List[DatasetConfig] = [
    DatasetConfig(name='novpn_top1000', data_path='/root/autodl-tmp/AppScanner/data/novpn_top1000/novpn_top1000_appscanner.pkl'),
    DatasetConfig(name='vpn_top1000', data_path='/root/autodl-tmp/AppScanner/data/vpn_top1000/vpn_top1000_appscanner.pkl'),
    DatasetConfig(name='novpn_top500', data_path='/root/autodl-tmp/AppScanner/data/novpn_top500/novpn_top500_appscanner.pkl'),
    DatasetConfig(name='vpn_top500', data_path='/root/autodl-tmp/AppScanner/data/vpn_top500/vpn_top500_appscanner.pkl'),
    DatasetConfig(name='novpn_top100', data_path='/root/autodl-tmp/AppScanner/data/novpn_top100/novpn_top100_appscanner.pkl'),
    DatasetConfig(name='vpn_top100', data_path='/root/autodl-tmp/AppScanner/data/vpn_top100/vpn_top100_appscanner.pkl'),
    DatasetConfig(name='novpn_top50', data_path='/root/autodl-tmp/AppScanner/data/novpn_top50/novpn_top50_appscanner.pkl'),
    DatasetConfig(name='vpn_top50', data_path='/root/autodl-tmp/AppScanner/data/vpn_top50/vpn_top50_appscanner.pkl'),
    DatasetConfig(name='novpn_top10', data_path='/root/autodl-tmp/AppScanner/data/novpn_top10/novpn_top10_appscanner.pkl'),
    DatasetConfig(name='vpn_top10', data_path='/root/autodl-tmp/AppScanner/data/vpn_top10/vpn_top10_appscanner.pkl'),
]


# 并发数量
MAX_WORKERS = 5

# 输出根目录
OUTPUT_ROOT = './output_multi'


# =============================================================================
# Training Function
# =============================================================================

def train_single_dataset(config: DatasetConfig) -> dict:
    """
    Train a single dataset by calling train_with_dataset.py.

    Returns dict with status and metrics.
    """
    start_time = datetime.now()
    output_dir = os.path.join(OUTPUT_ROOT, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Build command line arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train_with_dataset.py')

    cmd = [
        sys.executable, train_script,
        '--data_path', config.data_path,
    ]

    # Initialize result
    result = {
        'name': config.name,
        'status': 'failed',
        'accuracy': 0.0,
        'f1': 0.0,
        'elapsed': '00:00:00',
        'error': None
    }

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

        # Parse output for metrics (look for final results in output)
        output = process.stdout
        if process.returncode == 0:
            # Try to parse metrics from final model output
            # Look for lines like "Accuracy:  0.9876"
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('Accuracy:'):
                    try:
                        result['accuracy'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('F1 Score:') or line.startswith('F1:'):
                    try:
                        result['f1'] = float(line.split(':')[1].strip())
                    except:
                        pass

            if result['accuracy'] > 0 or result['f1'] > 0:
                result['status'] = 'success'
            else:
                result['status'] = 'success'  # Process completed but couldn't parse metrics
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

    print("=" * 70)
    print("AppScanner Multi-Dataset Concurrent Training")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets to train: {len(DATASETS)}")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print()

    # Print dataset list
    print("Datasets:")
    for i, ds in enumerate(DATASETS, 1):
        print(f"  {i}. {ds.name}: {ds.data_path}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Run training concurrently
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(train_single_dataset, config): config
            for config in DATASETS
        }

        # Process completed tasks
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)

                status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
                print(f"{status_icon} {result['name']:20s} | "
                      f"Acc: {result['accuracy']:.4f} | "
                      f"F1: {result['f1']:.4f} | "
                      f"Time: {result['elapsed']}")

            except Exception as e:
                print(f"[ERROR] {config.name}: {e}")
                results.append({
                    'name': config.name,
                    'status': 'error',
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'elapsed': '00:00:00',
                    'error': str(e)
                })

    # Summary
    end_time = datetime.now()
    elapsed = end_time - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Status':<10} {'Accuracy':>10} {'F1':>10} {'Time':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<20} {r['status']:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['elapsed']:>12}")

    print("-" * 70)
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")

    # Save summary
    summary_path = os.path.join(OUTPUT_ROOT, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"AppScanner Multi-Dataset Training Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")
        f.write(f"{'Dataset':<20} {'Status':<10} {'Accuracy':>10} {'F1':>10} {'Time':>12}\n")
        f.write(f"{'-'*70}\n")
        for r in results:
            f.write(f"{r['name']:<20} {r['status']:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['elapsed']:>12}\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
