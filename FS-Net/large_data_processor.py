#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS-Net NOVPN 数据处理器

用于加载 unified_novpn_processor.py 生成的数据并转换为 FS-Net 训练格式。

输入格式 (from unified_novpn_processor.py):
    FS-Net/novpn_unified_output/
    ├── <label1>.npz    # contains 'sequences' (object array of int16), 'label', 'label_id'
    ├── <label2>.npz
    ├── ...
    └── labels.json     # {'label2id': {...}, 'id2label': {...}}

输出格式 (for run_train.py):
    fsnet_novpn.pkl
    {
        'sequences': List[np.ndarray],  # 变长序列列表 (int16, ±值表示方向)
        'labels': np.ndarray,           # (N,) int64
        'label_map': Dict[int, str]     # {0: 'class1', 1: 'class2', ...}
    }

Usage:
    python novpn_processor.py --input ./novpn_unified_output --output ./data/novpn/novpn_fsnet.pkl
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
) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """Load dataset from unified_novpn_processor output directory.

    Args:
        npz_dir: Directory containing NPZ files and labels.json
        min_samples: Minimum samples per class (skip classes with fewer)

    Returns:
        sequences: List of variable-length sequences (int16, signed)
        labels: Label array (N,) int64
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

    all_sequences = []
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
                if 'sequences' not in data:
                    print(f"  [跳过] {label_name}: NPZ 中无 'sequences' 键")
                    skipped_classes.append((label_name, 0, "无 sequences 键"))
                    continue

                sequences = data['sequences']

                # Convert to list if needed
                if sequences.dtype == object:
                    sequences = list(sequences)
                else:
                    sequences = [sequences[i] for i in range(len(sequences))]

                if len(sequences) < min_samples:
                    skipped_classes.append((label_name, len(sequences), f"样本数 < {min_samples}"))
                    continue

                # Ensure each sequence is numpy array
                sequences = [np.asarray(s, dtype=np.int16) for s in sequences]

                all_sequences.extend(sequences)
                all_labels.extend([label_id] * len(sequences))
                loaded_classes.append((label_name, label_id, len(sequences)))

                # 计算序列长度统计
                seq_lens = [len(s) for s in sequences]
                avg_len = np.mean(seq_lens)
                print(f"  [加载] {label_name}: {len(sequences):>6} 条, 平均长度: {avg_len:.1f}")

        except Exception as e:
            print(f"  [错误] {label_name}: {e}")
            skipped_classes.append((label_name, 0, str(e)))

    print("-" * 50)

    if skipped_classes:
        print(f"\n跳过的类别 ({len(skipped_classes)} 个):")
        for name, count, reason in skipped_classes:
            print(f"  - {name}: {reason}")

    if not all_sequences:
        raise ValueError(f"No valid data found in {npz_dir}")

    labels_raw = np.array(all_labels, dtype=np.int64)

    # Remap labels to consecutive integers (0, 1, 2, ...)
    unique_labels = sorted(set(all_labels))
    old_to_new = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([old_to_new[l] for l in labels_raw], dtype=np.int64)

    # Update label_map (new_id -> class_name)
    label_map = {new: id2label[old] for old, new in old_to_new.items()}

    return all_sequences, labels, label_map


def save_pickle(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_path: Path
):
    """Save dataset as pickle file for run_train.py"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'sequences': sequences,
        'labels': labels,
        'label_map': label_map,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n保存到: {output_path}")


def print_statistics(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str]
):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"总样本数: {len(labels)}")
    print(f"类别数量: {len(label_map)}")

    # Sequence length statistics
    seq_lens = [len(s) for s in sequences]
    print(f"\n序列长度统计:")
    print(f"  Min:  {np.min(seq_lens)}")
    print(f"  Max:  {np.max(seq_lens)}")
    print(f"  Mean: {np.mean(seq_lens):.1f}")
    print(f"  Std:  {np.std(seq_lens):.1f}")

    # Packet length statistics (absolute values)
    all_lengths = np.concatenate([np.abs(s) for s in sequences])
    print(f"\n数据包长度统计 (绝对值):")
    print(f"  Min:  {np.min(all_lengths)}")
    print(f"  Max:  {np.max(all_lengths)}")
    print(f"  Mean: {np.mean(all_lengths):.1f}")

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


def main():
    parser = argparse.ArgumentParser(
        description="将 unified_novpn_processor 输出转换为 FS-Net 训练格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        # default="./data/novpn_unified_output",
        default="./data/vpn_unified_output",
        help="输入目录 (unified_novpn_processor 输出)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        # default="./data/novpn/novpn_fsnet.pkl",
        default="./data/novpn/vpn_fsnet.pkl",
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
    print("FS-Net NOVPN 数据处理器")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")
    print(f"最少样本: {args.min_samples}")

    # Check input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # Load data
    sequences, labels, label_map = load_npz_directory(
        input_dir,
        min_samples=args.min_samples
    )

    # Print statistics
    print_statistics(sequences, labels, label_map)

    # Save pickle
    save_pickle(sequences, labels, label_map, output_path)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  1. 编辑 run_train.py 中的 DATA_PATH:")
    print(f"     DATA_PATH = \"{output_path}\"")
    print(f"  2. 运行训练:")
    print(f"     python run_train.py")


if __name__ == "__main__":
    main()
