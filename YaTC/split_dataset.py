#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本

将 NPZ 格式的数据集按 8:1:1 划分为 train/val/test，
确保预训练只使用训练集，避免数据泄露。

输入格式 (unified_*_processor.py 输出):
    data/
    ├── <label1>.npz
    ├── <label2>.npz
    └── labels.json

输出格式:
    data/
    ├── train/
    │   ├── <label1>.npz
    │   ├── <label2>.npz
    │   └── labels.json
    ├── val/
    │   ├── <label1>.npz
    │   └── labels.json
    ├── test/
    │   ├── <label1>.npz
    │   └── labels.json
    └── labels.json (原始，保留)

支持的数据类型:
    - YaTC:               images (N, 40, 40)
    - DeepFingerprinting: flows (object array)
    - FS-Net:             sequences (object array)
    - AppScanner:         features (N, 54)

Usage:
    # 划分单个数据集
    python split_dataset.py --input YaTC/data/iscxtor --output YaTC/data/iscxtor_split

    # 原地划分（在原目录下创建 train/val/test 子目录）
    python split_dataset.py --input YaTC/data/iscxtor --inplace

    # 自定义比例
    python split_dataset.py --input ./data --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tensorflow.python.keras.combinations import times

# ==================== 配置参数 ====================
# 输入目录（包含 NPZ 文件和 labels.json）
# INPUT_DIR = "data/iscxtor"
# INPUT_DIR = "data/iscxvpn"
# INPUT_DIR = "data/cross_platform"
# INPUT_DIR = "data/ustc"
# INPUT_DIR = "data/novpn_unified_output"
INPUT_DIR = "data/vpn_unified_output"
# INPUT_DIR = "data/cic_iot_2022"

# 输出目录（None 表示使用 <input>_split）
OUTPUT_DIR = None

# 是否原地划分（在输入目录下创建 train/val/test 子目录）
INPLACE = False

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子
SEED = 42

# 每个 split 最少样本数（少于此数的类别会被跳过）
MIN_SAMPLES = 1
# ================================================


def detect_data_key(npz_path: Path) -> str:
    """检测 NPZ 文件中的数据键名"""
    with np.load(npz_path, allow_pickle=True) as data:
        keys = data.files
        # 按优先级检测
        for key in ['images', 'features', 'flows', 'sequences', 'data', 'X']:
            if key in keys:
                return key
    raise ValueError(f"无法识别 NPZ 数据格式: {npz_path}, keys={keys}")


def load_npz_data(npz_path: Path, data_key: str) -> np.ndarray:
    """加载 NPZ 数据"""
    with np.load(npz_path, allow_pickle=True) as data:
        return data[data_key]


def save_npz_data(
    output_path: Path,
    data: np.ndarray,
    data_key: str,
    label: str,
    label_id: int
):
    """保存 NPZ 数据"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{data_key: data},
        label=label,
        label_id=label_id
    )


def stratified_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成分层划分的索引"""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_samples: int = 3
):
    """
    划分数据集

    Args:
        input_dir: 输入目录（包含 NPZ 文件和 labels.json）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        min_samples: 每个 split 最少样本数（少于此数的类别会被跳过）
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为 1.0, 当前: {train_ratio + val_ratio + test_ratio}"

    # 加载 labels.json
    labels_json_path = input_dir / "labels.json"
    if not labels_json_path.exists():
        raise FileNotFoundError(f"labels.json not found in {input_dir}")

    with open(labels_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta.get("label2id", {})
    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}

    # 检测数据类型
    npz_files = list(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")

    data_key = detect_data_key(npz_files[0])
    print(f"检测到数据类型: {data_key}")

    # 创建输出目录
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int),
        'skipped': []
    }

    print(f"\n划分数据集: {input_dir}")
    print(f"比例: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"随机种子: {seed}")
    print("-" * 60)

    # 处理每个类别
    for label_name, label_id in sorted(label2id.items(), key=lambda x: x[1]):
        npz_path = input_dir / f"{label_name}.npz"
        if not npz_path.exists():
            print(f"  [跳过] {label_name}: NPZ 文件不存在")
            stats['skipped'].append((label_name, "文件不存在"))
            continue

        # 加载数据
        data = load_npz_data(npz_path, data_key)
        n_samples = len(data)

        # 检查最小样本数
        min_required = int(min_samples / min(train_ratio, val_ratio, test_ratio))
        if n_samples < min_required:
            print(f"  [跳过] {label_name}: 样本数 {n_samples} < {min_required}")
            stats['skipped'].append((label_name, f"样本数不足 ({n_samples})"))
            continue

        # 划分索引
        train_idx, val_idx, test_idx = stratified_split(
            n_samples, train_ratio, val_ratio, test_ratio, seed + label_id
        )

        # 保存各 split
        if len(train_idx) > 0:
            train_data = data[train_idx] if data.dtype != object else np.array([data[i] for i in train_idx], dtype=object)
            save_npz_data(train_dir / f"{label_name}.npz", train_data, data_key, label_name, label_id)
            stats['train'][label_name] = len(train_idx)

        if len(val_idx) > 0:
            val_data = data[val_idx] if data.dtype != object else np.array([data[i] for i in val_idx], dtype=object)
            save_npz_data(val_dir / f"{label_name}.npz", val_data, data_key, label_name, label_id)
            stats['val'][label_name] = len(val_idx)

        if len(test_idx) > 0:
            test_data = data[test_idx] if data.dtype != object else np.array([data[i] for i in test_idx], dtype=object)
            save_npz_data(test_dir / f"{label_name}.npz", test_data, data_key, label_name, label_id)
            stats['test'][label_name] = len(test_idx)

        print(f"  [划分] {label_name}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 生成各 split 的 labels.json
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        split_labels = sorted([f.stem for f in split_dir.glob("*.npz")])
        split_label2id = {name: i for i, name in enumerate(split_labels)}
        split_id2label = {i: name for i, name in enumerate(split_labels)}

        split_meta = {
            "label2id": split_label2id,
            "id2label": {str(k): v for k, v in split_id2label.items()},
            "num_classes": len(split_labels),
            "split": split_name,
            "source": str(input_dir),
            "ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio
            },
            "seed": seed
        }

        # 复制额外的元数据（如 mfr_config）
        for key in meta:
            if key not in ["label2id", "id2label", "num_classes"]:
                split_meta[key] = meta[key]

        with open(split_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n" + "=" * 60)
    print("划分完成!")
    print("=" * 60)

    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    total = total_train + total_val + total_test

    print(f"\n样本统计:")
    print(f"  Train: {total_train:>8} ({total_train/total*100:.1f}%)")
    print(f"  Val:   {total_val:>8} ({total_val/total*100:.1f}%)")
    print(f"  Test:  {total_test:>8} ({total_test/total*100:.1f}%)")
    print(f"  Total: {total:>8}")

    print(f"\n类别数: {len(stats['train'])}")

    if stats['skipped']:
        print(f"\n跳过的类别 ({len(stats['skipped'])} 个):")
        for name, reason in stats['skipped']:
            print(f"  - {name}: {reason}")

    print(f"\n输出目录:")
    print(f"  {output_dir}/")
    print(f"  ├── train/  ({len(stats['train'])} 类, {total_train} 样本)")
    print(f"  ├── val/    ({len(stats['val'])} 类, {total_val} 样本)")
    print(f"  └── test/   ({len(stats['test'])} 类, {total_test} 样本)")

    return stats


def main():
    # 使用配置参数
    input_dir = Path(INPUT_DIR)

    if INPLACE:
        output_dir = input_dir
    elif OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_split"

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    split_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        min_samples=MIN_SAMPLES
    )

    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print(f"""
预训练 (只使用 train split，避免数据泄露):
    # YaTC 预训练
    python train.py --mode pretrain --data_path {output_dir}/train

微调 (使用 train，验证用 val):
    python train.py --mode finetune \\
        --data_path {output_dir}/train \\
        --val_path {output_dir}/val

最终评估 (使用 test):
    python train.py --mode eval --data_path {output_dir}/test
""")


if __name__ == "__main__":
    import merge_datasets as merge
    main()
    time.sleep(60)
    merge.main()
