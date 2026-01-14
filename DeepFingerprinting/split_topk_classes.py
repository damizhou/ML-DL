#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS-Net PKL 文件 Top-K 类别分割脚本

从大型数据集中提取样本数量最多的 K 个类别，生成新的 PKL 文件。

输入格式:
    {
        'sequences': List[np.ndarray],  # 变长序列
        'labels': np.ndarray,           # 类别标签
        'label_map': Dict[int, str]     # 标签映射
    }

Usage:
    python split_topk_classes.py
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# =============================================================================
# 配置参数
# =============================================================================

# 输入文件路径
# INPUT_PATH = './data/novpn/novpn_deepfingerprinting.npz'
INPUT_PATH = './data/vpn/data.npz'

# 要处理的多个 Top-K 值
TOP_K_LIST = [10, 50, 100, 500, 1000]

# 每个类别的最小样本数 (低于此数的类别不参与排序)
MIN_SAMPLES = 10

# 输出目录 (None 表示自动生成: ./data/<name>_top<K>/)
OUTPUT_DIR = None


# =============================================================================
# 主逻辑
# =============================================================================

def load_data(path: str) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载数据集，支持 PKL 和 NPZ 格式。

    Returns:
        sequences: 变长序列列表
        labels: 标签数组
        label_map: 标签映射字典
    """
    path_lower = path.lower()
    path_obj = Path(path)

    if path_lower.endswith('.npz'):
        # NPZ 格式 (NumPy 归档)
        data = np.load(path, allow_pickle=True)
        sequences = list(data['sequences']) if 'sequences' in data else list(data['X'])
        labels = data['labels'] if 'labels' in data else data['y']

        # NPZ 中 label_map 可能是 0-d 数组包装的字典
        if 'label_map' in data:
            label_map = data['label_map']
            if isinstance(label_map, np.ndarray):
                label_map = label_map.item()  # 从 0-d 数组提取字典
        else:
            # 尝试读取同目录下的 labels.json
            labels_json_path = path_obj.parent / 'labels.json'
            if labels_json_path.exists():
                with open(labels_json_path, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
                # 支持 id2label 或 label2id 格式
                if 'id2label' in labels_data:
                    label_map = {int(k): v for k, v in labels_data['id2label'].items()}
                elif 'label2id' in labels_data:
                    label_map = {v: k for k, v in labels_data['label2id'].items()}
                else:
                    label_map = {int(k): v for k, v in labels_data.items()}
                print(f"  已加载标签映射: {labels_json_path}")
            else:
                # 如果没有 label_map，自动生成
                unique_labels = np.unique(labels)
                label_map = {int(l): f"class_{l}" for l in unique_labels}
    else:
        # PKL 格式
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sequences = data['sequences']
        labels = data['labels']
        label_map = data['label_map']

    return sequences, labels, label_map


def select_topk_classes(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str],
    top_k: int,
    min_samples: int = 0,
    verbose: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """
    选择样本数量最多的 Top-K 类别。

    Returns:
        filtered_sequences: 过滤后的序列
        filtered_labels: 重新映射的标签 (0 到 K-1)
        new_label_map: 新的标签映射
    """
    # 统计每个类别的样本数
    class_counts = Counter(labels)

    if verbose:
        print(f"\n原始数据集:")
        print(f"  总样本数: {len(labels):,}")
        print(f"  总类别数: {len(class_counts):,}")

    # 按样本数排序
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # 过滤最小样本数
    if min_samples > 0:
        sorted_classes = [(c, n) for c, n in sorted_classes if n >= min_samples]
        if verbose:
            print(f"  满足最小样本数 ({min_samples}) 的类别: {len(sorted_classes)}")

    # 实际可选类别数
    actual_k = min(top_k, len(sorted_classes))
    if actual_k < top_k and verbose:
        print(f"  [注意] 仅有 {actual_k} 个类别满足条件，少于请求的 {top_k}")

    # 选择 Top-K
    selected_classes = sorted_classes[:actual_k]

    if verbose:
        print(f"\nTop-{actual_k} 类别:")
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
    filtered_sequences = [sequences[i] for i in range(len(sequences)) if mask[i]]
    filtered_labels = np.array([old_to_new[l] for l in labels[mask]])

    if verbose:
        print(f"\n过滤后数据集:")
        print(f"  类别数: {len(new_label_map)}")
        print(f"  样本数: {len(filtered_labels):,}")

    return filtered_sequences, filtered_labels, new_label_map


def save_pkl(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_path: str,
    source_info: dict = None
):
    """保存为 FS-Net PKL 格式。"""
    data = {
        'sequences': sequences,
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
    sequences: List[np.ndarray],
    labels: np.ndarray,
    label_map: Dict[int, str],
    input_path: Path,
    top_k: int,
    min_samples: int,
    output_dir: str = None
):
    """处理单个 Top-K 值并保存结果。"""
    print(f"\n{'=' * 60}")
    print(f"处理 Top-{top_k}")
    print("=" * 60)

    # 选择 Top-K 类别
    filtered_sequences, filtered_labels, new_label_map = select_topk_classes(
        sequences, labels, label_map,
        top_k=top_k,
        min_samples=min_samples
    )

    # 如果没有足够的类别，跳过
    if len(new_label_map) == 0:
        print(f"  [跳过] 没有满足条件的类别")
        return

    # 确定输出路径
    parent_name = input_path.parent.name  # e.g., "iscxvpn"

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = input_path.parent.parent / f"{parent_name}_top{top_k}"

    output_name = f"{parent_name}_top{top_k}_deepfingerprinting.npz"
    output_path = out_dir / output_name

    # 保存 PKL
    save_pkl(
        filtered_sequences,
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

    # 保存 labels.json 方便查看
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
    print("FS-Net Top-K 类别分割 (批量处理)")
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
    sequences, labels, label_map = load_data(INPUT_PATH)
    print(f"序列数量: {len(sequences):,}")
    print(f"总类别数: {len(label_map):,}")

    # 统计序列长度
    seq_lengths = [len(seq) for seq in sequences]
    print(f"序列长度: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")

    input_path = Path(INPUT_PATH)

    # 批量处理所有 Top-K
    for top_k in TOP_K_LIST:
        process_single_topk(
            sequences, labels, label_map,
            input_path, top_k, MIN_SAMPLES, OUTPUT_DIR
        )

    print("\n" + "=" * 60)
    print(f"全部完成! 共处理 {len(TOP_K_LIST)} 个 Top-K 配置")
    print("=" * 60)


if __name__ == '__main__':
    main()
