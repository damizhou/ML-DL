#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AppScanner PKL 文件 Top-K 类别分割脚本

从大型数据集中提取样本数量最多的 K 个类别，生成新的 PKL 文件。

输入: ./data/vpn/vpn_appscanner.pkl (包含数千类别)
输出: ./data/vpn_top10/vpn_top10_appscanner.pkl (仅包含 Top-10 类别)

Usage:
    python split_topk_classes.py
"""

import os
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

# =============================================================================
# 配置参数
# =============================================================================

# 输入文件路径
INPUT_PATH = './data/novpn/novpn_appscanner.pkl'

# 要处理的多个 Top-K 值
TOP_K_LIST = [10, 50, 100, 500, 1000]

# 每个类别的最小样本数 (低于此数的类别不参与排序)
MIN_SAMPLES = 10

# 输出目录 (None 表示自动生成: ./data/<name>_top<K>/)
OUTPUT_DIR = None

# =============================================================================
# 主逻辑
# =============================================================================

def load_pkl(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """加载 PKL 数据集，支持多种格式。"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # 格式1: {'features', 'labels', 'label_map'}
    if 'features' in data:
        features = data['features']
        labels = data['labels']
        label_map = data['label_map']
    # 格式2: {'train_features', 'train_labels', ...}
    elif 'train_features' in data:
        parts_features = [data['train_features']]
        parts_labels = [data['train_labels']]

        if 'val_features' in data:
            parts_features.append(data['val_features'])
            parts_labels.append(data['val_labels'])

        parts_features.append(data['test_features'])
        parts_labels.append(data['test_labels'])

        features = np.concatenate(parts_features, axis=0)
        labels = np.concatenate(parts_labels, axis=0)
        label_map = data['label_map']
    else:
        raise KeyError(f"未知的数据格式，keys: {list(data.keys())}")

    # 清理 NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, labels, label_map


def select_topk_classes(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    top_k: int,
    min_samples: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    选择样本数量最多的 Top-K 类别。

    Returns:
        filtered_features: 过滤后的特征
        filtered_labels: 重新映射的标签 (0 到 K-1)
        new_label_map: 新的标签映射
    """
    # 统计每个类别的样本数
    class_counts = Counter(labels)

    print(f"\n原始数据集:")
    print(f"  总样本数: {len(labels):,}")
    print(f"  总类别数: {len(class_counts):,}")

    # 按样本数排序
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # 过滤最小样本数
    if min_samples > 0:
        sorted_classes = [(c, n) for c, n in sorted_classes if n >= min_samples]
        print(f"  满足最小样本数 ({min_samples}) 的类别: {len(sorted_classes)}")

    # 选择 Top-K
    selected_classes = sorted_classes[:top_k]

    print(f"\nTop-{top_k} 类别:")
    print("-" * 55)
    total_selected = 0
    for i, (class_id, count) in enumerate(selected_classes):
        class_name = label_map.get(class_id, f"Unknown({class_id})")
        # 截断过长的类名
        if len(class_name) > 35:
            class_name = class_name[:32] + "..."
        print(f"  {i+1:2d}. {class_name:<35} {count:>8,} 样本")
        total_selected += count
    print("-" * 55)
    print(f"  合计: {total_selected:,} 样本")

    # 创建新的标签映射
    selected_class_ids = set(c for c, _ in selected_classes)
    old_to_new = {old_id: new_id for new_id, (old_id, _) in enumerate(selected_classes)}

    new_label_map = {
        new_id: label_map.get(old_id, f"Unknown({old_id})")
        for old_id, new_id in old_to_new.items()
    }

    # 过滤数据
    mask = np.array([l in selected_class_ids for l in labels])
    filtered_features = features[mask]
    filtered_labels = np.array([old_to_new[l] for l in labels[mask]])

    print(f"\n过滤后数据集:")
    print(f"  类别数: {len(new_label_map)}")
    print(f"  样本数: {len(filtered_labels):,}")

    return filtered_features, filtered_labels, new_label_map


def save_pkl(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_path: str,
    source_info: dict = None
):
    """保存为 PKL 格式。"""
    data = {
        'features': features,
        'labels': labels,
        'label_map': label_map,
    }

    if source_info:
        data['source_info'] = source_info

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n已保存: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def process_single_topk(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    input_path: Path,
    top_k: int,
    min_samples: int,
    output_dir: str = None
):
    """处理单个 Top-K 值并保存结果。"""
    import json

    print(f"\n{'=' * 60}")
    print(f"处理 Top-{top_k}")
    print("=" * 60)

    # 选择 Top-K 类别
    filtered_features, filtered_labels, new_label_map = select_topk_classes(
        features, labels, label_map,
        top_k=top_k,
        min_samples=min_samples
    )

    # 确定输出路径
    parent_name = input_path.parent.name  # e.g., "vpn"

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = input_path.parent.parent / f"{parent_name}_top{top_k}"

    output_name = f"{parent_name}_top{top_k}_appscanner.pkl"
    output_path = out_dir / output_name

    # 保存 PKL
    save_pkl(
        filtered_features,
        filtered_labels,
        new_label_map,
        str(output_path),
        source_info={
            'source_file': str(input_path),
            'top_k': top_k,
            'min_samples': min_samples,
            'original_classes': len(label_map),
            'original_samples': len(labels),
        }
    )

    # 保存 labels.json
    labels_json_path = out_dir / "labels.json"
    with open(labels_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': {v: k for k, v in new_label_map.items()},
            'id2label': {str(k): v for k, v in new_label_map.items()},
            'num_classes': len(new_label_map),
            'source': str(input_path),
            'top_k': top_k,
        }, f, ensure_ascii=False, indent=2)
    print(f"标签映射: {labels_json_path}")


def main():
    print("=" * 60)
    print("AppScanner Top-K 类别分割 (批量处理)")
    print("=" * 60)
    print(f"输入文件: {INPUT_PATH}")
    print(f"Top-K 列表: {TOP_K_LIST}")
    print(f"最小样本数: {MIN_SAMPLES}")

    # 检查输入文件
    if not os.path.exists(INPUT_PATH):
        print(f"\n[错误] 输入文件不存在: {INPUT_PATH}")
        return

    # 只加载一次数据
    print(f"\n加载数据...")
    features, labels, label_map = load_pkl(INPUT_PATH)
    print(f"特征维度: {features.shape}")
    print(f"总样本数: {len(labels):,}")
    print(f"总类别数: {len(label_map):,}")

    input_path = Path(INPUT_PATH)

    # 批量处理所有 Top-K
    for top_k in TOP_K_LIST:
        process_single_topk(
            features, labels, label_map,
            input_path, top_k, MIN_SAMPLES, OUTPUT_DIR
        )

    print("\n" + "=" * 60)
    print(f"全部完成! 共处理 {len(TOP_K_LIST)} 个 Top-K 配置")
    print("=" * 60)


if __name__ == '__main__':
    main()
