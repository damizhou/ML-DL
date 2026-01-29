#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC Top-K 类别分割脚本

从已划分的 train/val/test 数据集中提取样本数量最多的 K 个类别，
生成新的子数据集用于不同类别规模的消融实验。

输入格式 (split_dataset.py 输出):
    data/xxx_split/
    ├── train/
    │   ├── labels.json
    │   ├── class1.npz
    │   └── class2.npz
    ├── val/
    │   └── ...
    └── test/
        └── ...

输出格式 (与输入目录同层级):
    data/
    ├── xxx_split/          (原始，保留)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── xxx_top10_split/    (Top-10 类别，同层级)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── xxx_top50_split/    (Top-50 类别，同层级)
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── ...

Usage:
    python split_topk_classes.py
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional


# =============================================================================
# 配置参数
# =============================================================================

# 输入目录（包含 train/val/test 子目录的已划分数据集）
INPUT_DIR = "./data/novpn_top100_split"
# INPUT_DIR = "./data/novpn_unified_output_split"

# 要处理的多个 Top-K 值
TOP_K_LIST = [10, 50]

# 每个类别的最小样本数 (低于此数的类别不参与排序)
MIN_SAMPLES = 10

# 是否复制文件（False 则使用符号链接，节省磁盘空间）
# Windows 建议设为 True，Linux/Mac 可设为 False
COPY_FILES = True


# =============================================================================
# 核心函数
# =============================================================================

def load_labels_json(path: Path) -> Dict:
    """加载 labels.json"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_sample_counts(split_dir: Path) -> Dict[str, int]:
    """
    统计指定 split 目录下各类别的样本数量。

    Args:
        split_dir: split 目录路径 (如 train/)

    Returns:
        {class_name: sample_count}
    """
    counts = {}
    labels_json = split_dir / "labels.json"

    if not labels_json.exists():
        raise FileNotFoundError(f"labels.json not found in {split_dir}")

    meta = load_labels_json(labels_json)
    label2id = meta.get("label2id", {})

    for class_name in label2id.keys():
        npz_path = split_dir / f"{class_name}.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                # 支持多种数据格式
                for key in ['images', 'features', 'flows', 'sequences', 'data', 'X']:
                    if key in data.files:
                        counts[class_name] = len(data[key])
                        break
        else:
            counts[class_name] = 0

    return counts


def select_topk_classes(
    class_counts: Dict[str, int],
    top_k: int,
    min_samples: int = 0
) -> List[str]:
    """
    选择样本数量最多的 Top-K 类别。

    Args:
        class_counts: {class_name: sample_count}
        top_k: 要选择的类别数
        min_samples: 最小样本数阈值

    Returns:
        选中的类别名列表（按样本数降序）
    """
    # 过滤最小样本数
    filtered = {k: v for k, v in class_counts.items() if v >= min_samples}

    # 按样本数降序排序
    sorted_classes = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    # 选择 Top-K
    actual_k = min(top_k, len(sorted_classes))
    selected = [name for name, count in sorted_classes[:actual_k]]

    return selected


def get_output_dir_name(input_dir_name: str, top_k: int) -> str:
    """
    根据输入目录名生成输出目录名。

    例如:
        novpn_unified_output_split -> novpn_top10_split
        vpn_unified_output_split -> vpn_top10_split
        xxx_split -> xxx_top10_split
    """
    # 尝试提取前缀
    suffixes_to_remove = ["_unified_output_split", "_output_split", "_split"]
    prefix = input_dir_name

    for suffix in suffixes_to_remove:
        if input_dir_name.endswith(suffix):
            prefix = input_dir_name[:-len(suffix)]
            break

    return f"{prefix}_top{top_k}_split"


def create_topk_split(
    src_dir: Path,
    dst_dir: Path,
    selected_classes: List[str],
    copy_files: bool = True
):
    """
    创建 Top-K 子集目录。

    Args:
        src_dir: 源 split 目录 (如 train/)
        dst_dir: 目标目录 (如 train_top10/)
        selected_classes: 选中的类别列表
        copy_files: True 复制文件，False 创建符号链接
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 复制/链接选中类别的 NPZ 文件
    for class_name in selected_classes:
        src_npz = src_dir / f"{class_name}.npz"
        dst_npz = dst_dir / f"{class_name}.npz"

        if src_npz.exists():
            if copy_files:
                shutil.copy2(src_npz, dst_npz)
            else:
                # 创建相对符号链接
                if dst_npz.exists():
                    dst_npz.unlink()
                rel_path = os.path.relpath(src_npz, dst_dir)
                dst_npz.symlink_to(rel_path)

    # 生成新的 labels.json
    new_label2id = {name: i for i, name in enumerate(sorted(selected_classes))}
    new_id2label = {str(i): name for name, i in new_label2id.items()}

    # 从源 labels.json 继承额外元数据
    src_meta = load_labels_json(src_dir / "labels.json")

    new_meta = {
        "label2id": new_label2id,
        "id2label": new_id2label,
        "num_classes": len(selected_classes),
        "source": str(src_dir),
        "top_k": len(selected_classes),
    }

    # 复制额外的元数据（如 mfr_config, split 等）
    for key in src_meta:
        if key not in ["label2id", "id2label", "num_classes"]:
            new_meta[key] = src_meta[key]

    with open(dst_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)


def process_topk(
    input_dir: Path,
    top_k: int,
    min_samples: int,
    copy_files: bool = True,
    verbose: bool = True
) -> Optional[Dict]:
    """
    处理单个 Top-K 配置，同步生成 train/val/test 子集。

    Args:
        input_dir: 输入目录（包含 train/val/test）
        top_k: Top-K 值
        min_samples: 最小样本数阈值
        copy_files: 是否复制文件
        verbose: 是否打印详细信息

    Returns:
        统计信息字典，如果跳过则返回 None
    """
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"
    test_dir = input_dir / "test"

    # 检查目录
    if not val_dir.exists():
        print(f"[错误] val 目录不存在: {val_dir}")
        return None

    # 基于 val 集统计样本数（val 数据量小，统计更快，且与 train 分布一致）
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"处理 Top-{top_k}")
        print("=" * 60)

    class_counts = get_class_sample_counts(val_dir)
    total_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    if verbose:
        print(f"\n原始 val 集:")
        print(f"  总类别数: {total_classes}")
        print(f"  总样本数: {total_samples:,}")

    # 选择 Top-K 类别
    selected_classes = select_topk_classes(class_counts, top_k, min_samples)
    actual_k = len(selected_classes)

    if actual_k == 0:
        print(f"  [跳过] 没有满足条件的类别")
        return None

    if actual_k < top_k and verbose:
        print(f"  [注意] 仅有 {actual_k} 个类别满足条件，少于请求的 {top_k}")

    # 打印选中的类别
    if verbose:
        print(f"\nTop-{actual_k} 类别 (按 val 样本数排序):")
        print("-" * 55)
        selected_total = 0
        for i, class_name in enumerate(selected_classes):
            count = class_counts[class_name]
            # 截断过长的类名
            display_name = class_name[:32] + "..." if len(class_name) > 35 else class_name
            print(f"  {i+1:3d}. {display_name:<35} {count:>8,} 样本")
            selected_total += count
        print("-" * 55)
        print(f"  合计: {selected_total:,} 样本 ({selected_total/total_samples*100:.1f}%)")

    # 计算输出目录（与输入目录同层级）
    output_dir_name = get_output_dir_name(input_dir.name, actual_k)
    output_dir = input_dir.parent / output_dir_name

    # 同步生成三个子集
    stats = {"top_k": actual_k, "output_dir": str(output_dir), "splits": {}}

    for split_name, src_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not src_dir.exists():
            if verbose:
                print(f"  [跳过] {split_name} 目录不存在")
            continue

        dst_dir = output_dir / split_name
        create_topk_split(src_dir, dst_dir, selected_classes, copy_files)

        # 统计输出
        split_counts = get_class_sample_counts(dst_dir)
        split_total = sum(split_counts.values())
        stats["splits"][split_name] = {
            "dir": str(dst_dir),
            "classes": len(split_counts),
            "samples": split_total
        }

        if verbose:
            print(f"\n已生成: {output_dir.name}/{split_name}/")
            print(f"  类别数: {len(split_counts)}")
            print(f"  样本数: {split_total:,}")

    return stats


def main():
    print("=" * 60)
    print("YaTC Top-K 类别分割")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"Top-K 列表: {TOP_K_LIST}")
    print(f"最小样本数: {MIN_SAMPLES}")
    print(f"复制模式: {'复制文件' if COPY_FILES else '符号链接'}")

    input_dir = Path(INPUT_DIR)

    if not input_dir.exists():
        print(f"\n[错误] 输入目录不存在: {input_dir}")
        return

    # 检查 val 目录（用 val 统计更快，且与 train 分布一致）
    val_dir = input_dir / "val"
    if not val_dir.exists():
        print(f"\n[错误] val 目录不存在: {val_dir}")
        print("请先运行 split_dataset.py 划分数据集")
        return

    # 显示原始数据集信息
    class_counts = get_class_sample_counts(val_dir)
    print(f"\n原始数据集 (val):")
    print(f"  类别数: {len(class_counts)}")
    print(f"  样本数: {sum(class_counts.values()):,}")

    # 批量处理所有 Top-K
    results = []
    for top_k in TOP_K_LIST:
        stats = process_topk(
            input_dir, top_k, MIN_SAMPLES, COPY_FILES, verbose=True
        )
        if stats:
            results.append(stats)

    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

    if results:
        print(f"\n生成的数据集:")
        print("-" * 60)
        print(f"{'Top-K':<10} {'Train':<15} {'Val':<15} {'Test':<15}")
        print("-" * 60)

        for stats in results:
            k = stats["top_k"]
            train_info = stats["splits"].get("train", {})
            val_info = stats["splits"].get("val", {})
            test_info = stats["splits"].get("test", {})

            train_str = f"{train_info.get('samples', 0):,}" if train_info else "-"
            val_str = f"{val_info.get('samples', 0):,}" if val_info else "-"
            test_str = f"{test_info.get('samples', 0):,}" if test_info else "-"

            print(f"Top-{k:<6} {train_str:<15} {val_str:<15} {test_str:<15}")

        print("-" * 60)

    # 打印使用说明
    # 计算示例输出目录名
    example_output = get_output_dir_name(input_dir.name, 10)
    example_path = input_dir.parent / example_output

    print(f"""
使用说明:
=========

1. 预训练 (使用完整 train 集):
   python refactor/train.py --mode pretrain --data_path {input_dir}/train

2. 微调 Top-K 子集:
   python refactor/train.py --mode finetune \\
       --data_path {example_path}/train \\
       --val_path {example_path}/val \\
       --nb_classes 10

3. 测试:
   python refactor/train.py --mode eval \\
       --data_path {example_path}/test \\
       --nb_classes 10
""")


if __name__ == "__main__":
    main()
