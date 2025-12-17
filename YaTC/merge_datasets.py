#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集合并脚本

将多个已划分的数据集合并为一个预训练数据集。
只合并 train split，确保不会泄露任何测试数据。

使用场景:
1. 联合预训练：合并多个数据集的 train split
2. 跨数据集迁移：合并外部数据集用于预训练

输入格式 (split_dataset.py 输出):
    dataset1/
    ├── train/
    │   ├── <label>.npz
    │   └── labels.json
    ├── val/
    └── test/

    dataset2/
    └── ...

输出格式:
    merged_pretrain/
    ├── <dataset1>_<label>.npz    # 添加数据集前缀避免冲突
    ├── <dataset2>_<label>.npz
    └── labels.json

Usage:
    # 合并多个数据集的 train split 用于预训练
    python merge_datasets.py \\
        --inputs YaTC/data/iscxtor_split/train YaTC/data/iscxvpn_split/train \\
        --output YaTC/data/merged_pretrain

    # 合并完整数据集（跨数据集迁移场景）
    python merge_datasets.py \\
        --inputs YaTC/data/ustc YaTC/data/iscxtor \\
        --output YaTC/data/external_pretrain \\
        --full  # 使用全部数据而非只用 train

    # 指定数据集名称前缀
    python merge_datasets.py \\
        --inputs ./data1 ./data2 \\
        --names iscxtor iscxvpn \\
        --output ./merged
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np


# ==================== 配置参数 ====================
# 脚本所在目录（用于解析相对路径）
SCRIPT_DIR = Path(__file__).resolve().parent

# 输入目录列表（多个数据集的 train split，相对于脚本目录）
INPUT_DIRS = [
    "data/iscxtor_split/train",
    "data/cic_iot_2022_split/train",
    "data/cross_platform_split/train",
    "data/novpn_unified_split/train",
    "data/vpn_unified_split/train",
    "data/iscxvpn_split/train",
    "data/ustc_split/train"
]

# 输出目录（相对于脚本目录）
OUTPUT_DIR = "data/merged_pretrain"

# 数据集名称（用作标签前缀，None 表示从目录名推断）
DATASET_NAMES = None

# 是否使用完整数据集（而非只用 train split）
USE_FULL = False
# ================================================


def detect_data_key(npz_path: Path) -> str:
    """检测 NPZ 文件中的数据键名"""
    with np.load(npz_path, allow_pickle=True) as data:
        keys = data.files
        for key in ['images', 'features', 'flows', 'sequences', 'data', 'X']:
            if key in keys:
                return key
    raise ValueError(f"无法识别 NPZ 数据格式: {npz_path}, keys={keys}")


def load_npz_data(npz_path: Path, data_key: str) -> np.ndarray:
    """加载 NPZ 数据"""
    with np.load(npz_path, allow_pickle=True) as data:
        return data[data_key]


def merge_datasets(
    input_dirs: List[Path],
    output_dir: Path,
    dataset_names: Optional[List[str]] = None,
    use_full: bool = False
):
    """
    合并多个数据集

    Args:
        input_dirs: 输入目录列表
        output_dir: 输出目录
        dataset_names: 数据集名称（用作前缀）
        use_full: 是否使用完整数据集（而非只用 train split）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成数据集名称
    if dataset_names is None:
        dataset_names = [d.parent.name if d.name == 'train' else d.name for d in input_dirs]

    # 检测数据类型（从第一个数据集）
    data_key = None
    all_meta = {}

    # 统计信息
    stats = defaultdict(lambda: {'samples': 0, 'classes': 0})
    merged_labels = []

    print("=" * 60)
    print("合并数据集")
    print("=" * 60)
    print(f"输入目录: {len(input_dirs)} 个")
    for i, (d, name) in enumerate(zip(input_dirs, dataset_names)):
        print(f"  [{i+1}] {name}: {d}")
    print(f"输出目录: {output_dir}")
    print(f"模式: {'完整数据集' if use_full else '仅 train split'}")
    print("-" * 60)

    # 处理每个数据集
    for dataset_idx, (input_dir, dataset_name) in enumerate(zip(input_dirs, dataset_names)):
        print(f"\n处理数据集: {dataset_name}")

        # 检查 labels.json
        labels_json = input_dir / "labels.json"
        if not labels_json.exists():
            print(f"  [警告] labels.json 不存在，跳过")
            continue

        with open(labels_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label2id = meta.get("label2id", {})

        # 保存第一个数据集的额外元数据
        if dataset_idx == 0:
            for key in meta:
                if key not in ["label2id", "id2label", "num_classes", "split", "source", "ratios", "seed"]:
                    all_meta[key] = meta[key]

        # 处理每个类别
        npz_files = list(input_dir.glob("*.npz"))
        if not npz_files:
            print(f"  [警告] 无 NPZ 文件，跳过")
            continue

        # 检测数据类型
        if data_key is None:
            data_key = detect_data_key(npz_files[0])
            print(f"  数据类型: {data_key}")

        for label_name in sorted(label2id.keys()):
            npz_path = input_dir / f"{label_name}.npz"
            if not npz_path.exists():
                continue

            # 加载数据
            data = load_npz_data(npz_path, data_key)
            n_samples = len(data)

            if n_samples == 0:
                continue

            # 生成合并后的标签名（添加数据集前缀）
            merged_label = f"{dataset_name}_{label_name}"
            merged_labels.append(merged_label)

            # 保存到输出目录
            output_path = output_dir / f"{merged_label}.npz"
            np.savez_compressed(
                output_path,
                **{data_key: data},
                label=merged_label,
                source_dataset=dataset_name,
                source_label=label_name
            )

            stats[dataset_name]['samples'] += n_samples
            stats[dataset_name]['classes'] += 1
            print(f"  [合并] {label_name} -> {merged_label}: {n_samples} 样本")

    # 生成合并后的 labels.json
    merged_label2id = {name: i for i, name in enumerate(merged_labels)}
    merged_id2label = {i: name for i, name in enumerate(merged_labels)}

    merged_meta = {
        "label2id": merged_label2id,
        "id2label": {str(k): v for k, v in merged_id2label.items()},
        "num_classes": len(merged_labels),
        "source_datasets": dataset_names,
        "merge_mode": "full" if use_full else "train_only",
        **all_meta
    }

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)

    total_samples = sum(s['samples'] for s in stats.values())
    total_classes = sum(s['classes'] for s in stats.values())

    print(f"\n各数据集统计:")
    print(f"{'数据集':<20} {'类别数':>10} {'样本数':>10} {'占比':>10}")
    print("-" * 52)
    for name in dataset_names:
        s = stats[name]
        pct = s['samples'] / total_samples * 100 if total_samples > 0 else 0
        print(f"{name:<20} {s['classes']:>10} {s['samples']:>10} {pct:>9.1f}%")
    print("-" * 52)
    print(f"{'总计':<20} {total_classes:>10} {total_samples:>10} {'100.0':>9}%")

    print(f"\n输出目录: {output_dir}/")
    print(f"  ├── labels.json")
    print(f"  └── *.npz ({len(merged_labels)} 个文件)")

    return stats


def main():
    # 使用配置参数，相对路径基于脚本目录
    input_dirs = [SCRIPT_DIR / p for p in INPUT_DIRS]
    output_dir = SCRIPT_DIR / OUTPUT_DIR

    # 验证输入目录
    for d in input_dirs:
        if not d.exists():
            raise FileNotFoundError(f"输入目录不存在: {d}")

    merge_datasets(
        input_dirs=input_dirs,
        output_dir=output_dir,
        dataset_names=DATASET_NAMES,
        use_full=USE_FULL
    )

    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print(f"""
预训练（使用合并后的数据集）:
    python train.py --mode pretrain --data_path {output_dir}

注意事项:
    1. 合并的数据集仅用于预训练
    2. 微调和评估应使用原始数据集的 train/val/test split
    3. 确保评估数据集的 test split 未参与预训练
""")


if __name__ == "__main__":
    main()
