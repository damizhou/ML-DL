#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFingerprinting 大数据处理器

用于加载 unified_novpn_processor.py 生成的数据并转换为 DeepFingerprinting 训练格式。

输入格式 (from unified_novpn_processor.py):
    data/vpn_unified_output/
    ├── <label1>.npz    # contains 'sequences' (object array of int16), 'label', 'label_id'
    ├── <label2>.npz
    ├── ...
    └── labels.json     # {'label2id': {...}, 'id2label': {...}}

输出格式 (for train.py - single_npz format):
    data/vpn/
    ├── data.npz        # contains 'X' (object array of int8 ±1), 'y' (int64 labels)
    └── labels.json     # {'label2id': {...}, 'id2label': {...}}

特征转换:
    FS-Net: 数据包长度序列 (±1 ~ ±1500)
    DF:     方向序列 (仅 ±1)
    转换方式: sign(seq) -> +1 (出站) / -1 (入站)

Usage:
    python large_data_processor.py --input ./data/vpn_unified_output --output ./data/vpn
"""

import os
import json
import shutil
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
        sequences: List of direction sequences (int8, ±1 only)
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

                # 转换为方向序列 (±1): sign(length) -> direction
                # FS-Net 格式: ±长度 (1~1500)
                # DF 格式: ±1 (方向)
                sequences = [np.sign(s).astype(np.int8) for s in sequences]

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


def save_npz(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_dir: Path
):
    """Save dataset as NPZ file for train.py (single_npz format)"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to object array for variable-length sequences
    X = np.array(sequences, dtype=object)
    y = labels

    # Save data.npz
    npz_path = output_dir / 'data.npz'
    np.savez(npz_path, X=X, y=y)
    print(f"\n保存数据到: {npz_path}")

    # Save labels.json
    labels_json = {
        'label2id': {name: idx for idx, name in label_map.items()},
        'id2label': {str(idx): name for idx, name in label_map.items()}
    }
    labels_path = output_dir / 'labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)
    print(f"保存标签到: {labels_path}")


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

    # Direction distribution statistics
    all_directions = np.concatenate(sequences)
    pos_count = np.sum(all_directions == 1)
    neg_count = np.sum(all_directions == -1)
    total = len(all_directions)
    print(f"\n方向分布统计:")
    print(f"  出站 (+1): {pos_count:>10} ({pos_count/total*100:.1f}%)")
    print(f"  入站 (-1): {neg_count:>10} ({neg_count/total*100:.1f}%)")

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
        description="将 unified_novpn_processor 输出转换为 DeepFingerprinting 训练格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./data/vpn_unified_output",
        help="输入目录 (unified_vpn_processor 输出)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/vpn",
        help="输出目录 (包含 data.npz 和 labels.json)"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="每类最少样本数 (默认: 10)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print("=" * 60)
    print("DeepFingerprinting 大数据处理器")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
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

    # Save as NPZ (single_npz format for train.py)
    save_npz(sequences, labels, label_map, output_dir)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  1. 编辑 train.py 中的 data_path:")
    print(f"     data_path: str = '{output_dir}'")
    print(f"  2. 运行训练:")
    print(f"     python train.py")


if __name__ == "__main__":
    main()
