#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFingerprinting NPZ 文件 Top-K 类别分割脚本

从大型数据集中提取样本数量最多的 K 个类别，生成新的 NPZ 文件。

支持三种输入格式:
1. single_npz: data.npz (X, y) + labels.json
2. unified_dir: <label>.npz + labels.json (每个类别一个文件)
3. multi_npz: 多目录 NPZ 文件 (遍历所有子目录)

输出格式: unified_dir (统一输出为每类一个 NPZ)

Usage:
    python split_topk_classes.py
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 配置参数
# =============================================================================

# 输入路径 (目录或单个 NPZ 文件)
INPUT_PATH = './DatasetDealer/VPN/npz_longflows_little'

# 要处理的多个 Top-K 值
TOP_K_LIST = [10, 50, 100, 500, 1000]

# 每个类别的最小样本数 (低于此数的类别不参与排序)
MIN_SAMPLES = 10

# 输出目录 (None 表示自动生成: ./data/<name>_top<K>/)
OUTPUT_DIR = None


# =============================================================================
# 数据加载函数
# =============================================================================

def detect_data_format(data_path: Path) -> str:
    """自动检测数据格式"""
    if data_path.is_file() and data_path.suffix == '.npz':
        return 'single_npz'

    if data_path.is_dir():
        labels_json = data_path / "labels.json"
        if labels_json.exists():
            npz_files = list(data_path.glob("*.npz"))
            if npz_files:
                with np.load(npz_files[0], allow_pickle=True) as data:
                    if 'X' in data.files and 'y' in data.files:
                        return 'single_npz'
                    elif 'flows' in data.files:
                        return 'unified_dir'

        sub_npz = list(data_path.rglob("*.npz"))
        if sub_npz:
            return 'multi_npz'

    raise ValueError(f"无法识别数据格式: {data_path}")


def load_single_npz(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载单个 NPZ 文件 (data.npz 格式)"""
    if data_path.is_file():
        labels_json = data_path.parent / "labels.json"
        npz_path = data_path
    else:
        labels_json = data_path / "labels.json"
        npz_path = data_path / "data.npz"

    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id2label = {int(k): str(v) for k, v in meta.get("id2label", {}).items()}

    with np.load(npz_path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"]

    if X.dtype == object:
        flows = list(X)
    else:
        flows = [X[i] for i in range(len(X))]

    flows = [np.asarray(f, dtype=np.int8) for f in flows]
    labels = y.astype(np.int64)

    return flows, labels, id2label


def load_unified_dir(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载统一目录格式 (<label>.npz + labels.json)"""
    labels_json = data_path / "labels.json"

    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta.get("label2id", {})
    id2label = {int(k): str(v) for k, v in meta.get("id2label", {}).items()}

    all_flows = []
    all_labels = []

    for label_name, label_id in label2id.items():
        npz_path = data_path / f"{label_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            flows = data["flows"]
            all_flows.extend(flows)
            all_labels.extend([int(label_id)] * len(flows))

    flows = [np.asarray(f, dtype=np.int8) for f in all_flows]
    labels = np.array(all_labels, dtype=np.int64)

    return flows, labels, id2label


def load_multi_npz(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """加载多 NPZ 目录 (全部加载到内存)"""
    # 查找 labels.json
    labels_json = data_path / "labels.json"
    if not labels_json.exists():
        for p in data_path.rglob("labels.json"):
            labels_json = p
            break

    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = {str(k): int(v) for k, v in meta.get("label2id", {}).items()}
    id2label = {int(k): str(v) for k, v in meta.get("id2label", {}).items()}

    all_flows = []
    all_labels = []

    npz_files = sorted(data_path.rglob("*.npz"))
    print(f"  扫描 NPZ 文件: {len(npz_files)} 个")

    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "flows" not in data.files:
                    continue

                flows = data["flows"]
                n = len(flows)
                if n == 0:
                    continue

                # 获取标签
                if "labels" in data.files:
                    file_labels = np.asarray(data["labels"]).reshape(-1)
                    if file_labels.size == 1:
                        file_labels = np.full(n, file_labels[0])
                else:
                    label_name = npz_path.stem
                    if label_name not in label2id:
                        continue
                    file_labels = np.full(n, label2id[label_name])

                for i in range(n):
                    all_flows.append(np.asarray(flows[i], dtype=np.int8))

                    label = file_labels[i] if i < len(file_labels) else file_labels[0]
                    if isinstance(label, (bytes, np.bytes_)):
                        label = label.decode('utf-8')
                    if isinstance(label, str):
                        label = label2id.get(label, 0)
                    all_labels.append(int(label))

        except Exception as e:
            print(f"  跳过文件 {npz_path}: {e}")
            continue

    labels = np.array(all_labels, dtype=np.int64)
    return all_flows, labels, id2label


def load_data(data_path: Path) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str], str]:
    """自动检测格式并加载数据"""
    fmt = detect_data_format(data_path)
    print(f"  检测到格式: {fmt}")

    if fmt == 'single_npz':
        flows, labels, id2label = load_single_npz(data_path)
    elif fmt == 'unified_dir':
        flows, labels, id2label = load_unified_dir(data_path)
    elif fmt == 'multi_npz':
        flows, labels, id2label = load_multi_npz(data_path)
    else:
        raise ValueError(f"不支持的格式: {fmt}")

    return flows, labels, id2label, fmt


# =============================================================================
# Top-K 选择与保存
# =============================================================================

def select_topk_classes(
    flows: List[np.ndarray],
    labels: np.ndarray,
    id2label: Dict[int, str],
    top_k: int,
    min_samples: int = 0,
    verbose: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, str]]:
    """
    选择样本数量最多的 Top-K 类别。

    Returns:
        filtered_flows: 过滤后的 flows
        filtered_labels: 重新映射的标签 (0 到 K-1)
        new_id2label: 新的标签映射
    """
    class_counts = Counter(labels)

    if verbose:
        print(f"\n原始数据集:")
        print(f"  总样本数: {len(labels):,}")
        print(f"  总类别数: {len(class_counts):,}")

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    if min_samples > 0:
        sorted_classes = [(c, n) for c, n in sorted_classes if n >= min_samples]
        if verbose:
            print(f"  满足最小样本数 ({min_samples}) 的类别: {len(sorted_classes)}")

    actual_k = min(top_k, len(sorted_classes))
    if actual_k < top_k and verbose:
        print(f"  [注意] 仅有 {actual_k} 个类别满足条件，少于请求的 {top_k}")

    selected_classes = sorted_classes[:actual_k]

    if verbose:
        print(f"\nTop-{actual_k} 类别:")
        print("-" * 55)
        total_selected = 0
        for i, (class_id, count) in enumerate(selected_classes):
            class_name = id2label.get(class_id, f"Unknown({class_id})")
            if len(class_name) > 35:
                class_name = class_name[:32] + "..."
            print(f"  {i+1:2d}. {class_name:<35} {count:>8,} 样本")
            total_selected += count
        print("-" * 55)
        print(f"  合计: {total_selected:,} 样本")

    selected_class_ids = set(c for c, _ in selected_classes)
    old_to_new = {old_id: new_id for new_id, (old_id, _) in enumerate(selected_classes)}

    new_id2label = {
        new_id: id2label.get(old_id, f"Unknown({old_id})")
        for old_id, new_id in old_to_new.items()
    }

    mask = np.array([l in selected_class_ids for l in labels])
    filtered_flows = [flows[i] for i in range(len(flows)) if mask[i]]
    filtered_labels = np.array([old_to_new[l] for l in labels[mask]])

    if verbose:
        print(f"\n过滤后数据集:")
        print(f"  类别数: {len(new_id2label)}")
        print(f"  样本数: {len(filtered_labels):,}")

    return filtered_flows, filtered_labels, new_id2label


def save_unified_dir(
    flows: List[np.ndarray],
    labels: np.ndarray,
    id2label: Dict[int, str],
    output_dir: Path,
    source_info: dict = None
):
    """保存为 unified_dir 格式 (每个类别一个 NPZ)"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按类别分组保存
    label2id = {v: k for k, v in id2label.items()}

    for class_id, class_name in id2label.items():
        mask = labels == class_id
        class_flows = [flows[i] for i in range(len(flows)) if mask[i]]

        if len(class_flows) == 0:
            continue

        # 转换为 object 数组保存变长序列
        flows_array = np.empty(len(class_flows), dtype=object)
        for i, f in enumerate(class_flows):
            flows_array[i] = f

        npz_path = output_dir / f"{class_name}.npz"
        np.savez_compressed(npz_path, flows=flows_array)

    # 保存 labels.json
    labels_json = {
        'label2id': label2id,
        'id2label': {str(k): v for k, v in id2label.items()},
        'num_classes': len(id2label),
    }
    if source_info:
        labels_json['source_info'] = source_info

    with open(output_dir / "labels.json", 'w', encoding='utf-8') as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    # 计算总大小
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.npz"))
    print(f"\n已保存: {output_dir}")
    print(f"  文件数: {len(id2label)} 个 NPZ")
    print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")


def process_single_topk(
    flows: List[np.ndarray],
    labels: np.ndarray,
    id2label: Dict[int, str],
    input_path: Path,
    top_k: int,
    min_samples: int,
    output_dir: str = None
):
    """处理单个 Top-K 值并保存结果。"""
    print(f"\n{'=' * 60}")
    print(f"处理 Top-{top_k}")
    print("=" * 60)

    filtered_flows, filtered_labels, new_id2label = select_topk_classes(
        flows, labels, id2label,
        top_k=top_k,
        min_samples=min_samples
    )

    if len(new_id2label) == 0:
        print(f"  [跳过] 没有满足条件的类别")
        return

    # 确定输出路径
    if input_path.is_file():
        parent_name = input_path.stem
        base_dir = input_path.parent.parent
    else:
        parent_name = input_path.name
        base_dir = input_path.parent

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = base_dir / f"{parent_name}_top{top_k}"

    save_unified_dir(
        filtered_flows,
        filtered_labels,
        new_id2label,
        out_dir,
        source_info={
            'source_path': str(input_path),
            'top_k': top_k,
            'min_samples': min_samples,
            'original_classes': len(id2label),
            'original_samples': len(labels),
        }
    )


def main():
    print("=" * 60)
    print("DeepFingerprinting Top-K 类别分割 (批量处理)")
    print("=" * 60)
    print(f"输入路径: {INPUT_PATH}")
    print(f"Top-K 列表: {TOP_K_LIST}")
    print(f"最小样本数: {MIN_SAMPLES}")

    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        print(f"\n[错误] 输入路径不存在: {INPUT_PATH}")
        return

    # 只加载一次数据
    print(f"\n加载数据...")
    flows, labels, id2label, fmt = load_data(input_path)
    print(f"  流数量: {len(flows):,}")
    print(f"  类别数: {len(id2label):,}")

    # 统计序列长度
    seq_lengths = [len(f) for f in flows]
    print(f"  序列长度: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")

    # 批量处理所有 Top-K
    for top_k in TOP_K_LIST:
        process_single_topk(
            flows, labels, id2label,
            input_path, top_k, MIN_SAMPLES, OUTPUT_DIR
        )

    print("\n" + "=" * 60)
    print(f"全部完成! 共处理 {len(TOP_K_LIST)} 个 Top-K 配置")
    print("=" * 60)


if __name__ == '__main__':
    main()
