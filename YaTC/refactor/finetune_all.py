#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YaTC 批量微调脚本

支持 split_dataset.py 划分后的目录结构:
    data/<dataset>_split/
    ├── train/
    ├── val/
    └── test/

Usage:
    python finetune_all.py
"""

from pathlib import Path
from datetime import datetime

# 数据集列表（指向 _split 目录）
DATASETS = [
    "/root/autodl-tmp/YaTc/data/novpn_top1000_split", # NoVPN unified
]

BASE_PATH = Path(__file__).parent.parent


def main():
    import finetune

    overall_start = datetime.now()

    print("=" * 60)
    print("YaTC Batch Fine-tuning")
    print("=" * 60)
    print(f"Start time: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {DATASETS}")
    print("=" * 60)

    for i, rel_path in enumerate(DATASETS):
        data_path = BASE_PATH / rel_path
        dataset_start = datetime.now()

        print(f"\n{'#' * 60}")
        print(f"# [{i+1}/{len(DATASETS)}] {data_path.name}")
        print(f"# Path: {data_path}")
        print(f"# Start: {dataset_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}\n")

        # 检查是否为 split 目录结构
        train_dir = data_path / "train"
        val_dir = data_path / "val"
        test_dir = data_path / "test"

        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            # 已划分的目录结构
            print(f"检测到已划分目录结构")
            finetune.DATA_PATH = data_path
            finetune.TRAIN_PATH = train_dir
            finetune.VAL_PATH = val_dir
            finetune.TEST_PATH = test_dir
            finetune.USE_SPLIT_DIR = True
        elif data_path.exists():
            # 未划分的原始目录（运行时划分）
            print(f"检测到原始目录结构（将在运行时划分）")
            finetune.DATA_PATH = data_path
            finetune.USE_SPLIT_DIR = False
        else:
            print(f"[WARNING] 路径不存在: {data_path}, 跳过")
            continue

        try:
            finetune.main()
        except Exception as e:
            print(f"[ERROR] {data_path.name} 失败: {e}")
            import traceback
            traceback.print_exc()

        # 打印单个数据集用时
        dataset_end = datetime.now()
        dataset_elapsed = dataset_end - dataset_start
        ds_hours, ds_remainder = divmod(int(dataset_elapsed.total_seconds()), 3600)
        ds_mins, ds_secs = divmod(ds_remainder, 60)
        print(f"\n[{data_path.name}] 完成，用时: {ds_hours:02d}:{ds_mins:02d}:{ds_secs:02d}")

    overall_end = datetime.now()
    elapsed = overall_end - overall_start
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 60}")
    print(f"全部完成，总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
