#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AppScanner NOVPN 数据处理器

用于加载 unified_novpn_processor.py 生成的数据并转换为 AppScanner 训练格式。

输入格式 (from unified_novpn_processor.py):
    AppScanner/novpn_unified_output/
    ├── <label1>.npz    # contains 'features' (N×54), 'label', 'label_id'
    ├── <label2>.npz
    ├── ...
    └── labels.json     # {'label2id': {...}, 'id2label': {...}}

输出格式 (for train_with_dataset.py):
    appscanner_novpn.pkl
    {
        'features': np.ndarray,  # (N, 54)
        'labels': np.ndarray,    # (N,)
        'label_map': Dict[int, str]
    }

Usage:
    python novpn_processor.py --input ./novpn_unified_output --output ./data/novpn/novpn_appscanner.pkl
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


def load_npz_directory(
    npz_dir: Path,
    min_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load dataset from unified_novpn_processor output directory.

    Args:
        npz_dir: Directory containing NPZ files and labels.json
        min_samples: Minimum samples per class (skip classes with fewer)

    Returns:
        features: Feature matrix (N, 54)
        labels: Label array (N,)
        label_map: Mapping from label_id to class name
    """
    # Load labels.json
    labels_json_path = npz_dir / "labels.json"
    if not labels_json_path.exists():
        raise FileNotFoundError(f"labels.json not found in {npz_dir}")

    with open(labels_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta.get("label2id", {})
    id2label_raw = meta.get("id2label", {})
    id2label = {int(k): v for k, v in id2label_raw.items()}

    all_features = []
    all_labels = []
    skipped_classes = []
    loaded_classes = []

    print(f"\n加载目录: {npz_dir}")
    print(f"发现 {len(label2id)} 个类别")
    print("-" * 50)

    # Load each NPZ file
    for label_name, label_id in sorted(label2id.items(), key=lambda x: x[1]):
        npz_path = npz_dir / f"{label_name}.npz"
        if not npz_path.exists():
            print(f"  [跳过] {label_name}: NPZ 文件不存在")
            skipped_classes.append((label_name, 0, "文件不存在"))
            continue

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if 'features' not in data:
                    print(f"  [跳过] {label_name}: NPZ 中无 'features' 键")
                    skipped_classes.append((label_name, 0, "无 features 键"))
                    continue

                features = data['features']

                if len(features) < min_samples:
                    skipped_classes.append((label_name, len(features), f"样本数 < {min_samples}"))
                    continue

                all_features.append(features)
                all_labels.extend([label_id] * len(features))
                loaded_classes.append((label_name, label_id, len(features)))
                print(f"  [加载] {label_name}: {len(features):>6} 条")

        except Exception as e:
            print(f"  [错误] {label_name}: {e}")
            skipped_classes.append((label_name, 0, str(e)))

    print("-" * 50)

    if skipped_classes:
        print(f"\n跳过的类别 ({len(skipped_classes)} 个):")
        for name, count, reason in skipped_classes:
            print(f"  - {name}: {reason}")

    if not all_features:
        raise ValueError(f"No valid data found in {npz_dir}")

    # Concatenate all features
    features = np.concatenate(all_features, axis=0)
    labels_raw = np.array(all_labels, dtype=np.int64)

    # Clean data: replace NaN/Inf with 0
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"\n[Warning] 发现 {nan_count} 个 NaN 和 {inf_count} 个 Inf，已替换为 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Remap labels to consecutive integers (0, 1, 2, ...)
    unique_labels = sorted(set(all_labels))
    old_to_new = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([old_to_new[l] for l in labels_raw], dtype=np.int64)

    # Update label_map (new_id -> class_name)
    label_map = {new: id2label[old] for old, new in old_to_new.items()}

    return features.astype(np.float32), labels, label_map


def save_pickle(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_path: Path
):
    """Save dataset as pickle file for train_with_dataset.py"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'features': features,
        'labels': labels,
        'label_map': label_map,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n保存到: {output_path}")


def print_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str]
):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"总样本数: {len(labels)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"类别数量: {len(label_map)}")

    print("\n类别分布:")
    print("-" * 60)
    print(f"{'类别':<30} {'样本数':>10} {'占比':>10}")
    print("-" * 60)

    unique, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        class_name = label_map.get(label_id, f"Unknown({label_id})")
        ratio = count / len(labels) * 100
        print(f"{class_name:<30} {count:>10} {ratio:>9.1f}%")

    print("-" * 60)

    # Feature statistics
    print("\n特征统计:")
    print(f"  Min:  {features.min():.4f}")
    print(f"  Max:  {features.max():.4f}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std:  {features.std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="将 unified_novpn_processor 输出转换为 AppScanner 训练格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./data/vpn_unified_output",
        help="输入目录 (unified_novpn_processor 输出)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/vpn/vpn_appscanner.pkl",
        help="输出 pickle 文件路径"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="每类最少样本数 (默认: 10)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("AppScanner NOVPN 数据处理器")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")
    print(f"最少样本: {args.min_samples}")

    # Check input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # Load data
    features, labels, label_map = load_npz_directory(
        input_dir,
        min_samples=args.min_samples
    )

    # Print statistics
    print_statistics(features, labels, label_map)

    # Save pickle
    save_pickle(features, labels, label_map, output_path)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  1. 编辑 train_with_dataset.py 中的 features_path:")
    print(f"     features_path: str = '{output_path}'")
    print(f"  2. 运行训练:")
    print(f"     python train_with_dataset.py")


if __name__ == "__main__":
    main()
