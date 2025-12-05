#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dirseq_pipeline.py
先跑 deal_iscxvpn_labels.py 生成 label_map.csv
再跑 build_dirseq_dataset.py 生成 dirseq/data.npz 等产物
—— 所有参数集中在本脚本顶部，保持两阶段一致性 ——
"""

from __future__ import annotations
import os, sys
from pathlib import Path

# ========== 一处配置，双阶段共享 ==========
DATASET_KEY  = "iscx"
DATASET_ROOT = "/netdisk/dataset/public_dataset/ISCX-VPN-NonVPN-2016"

# dirseq 解析配置（与 build_dirseq_dataset.py 完全一致）
PAYLOAD_ONLY       = False         # False: 统计所有 TCP/UDP 包；True: 仅统计有负载的包
MIN_LEN            = 3             # 短于该包数的流丢弃
TRUNCATE_LEN       = 5000          # None=不截断；整数=超出截断到该值（不 padding）
EXCLUDE_L4         = {("udp", 5353)}  # (proto, port) 黑名单；整条流丢弃
EXCLUDE_MCAST_IPS  = {"224.0.0.251", "ff02::fb", "ff05::fb"}
USE_SHA256         = False         # 缓存签名是否包含 sha256（更稳但慢）
# =======================================

def _import_module(mod_name: str, script_dir: Path):
    """
    保证能以 import 方式加载同目录下的两个脚本文件。
    """
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return __import__(mod_name)

def main():
    here = Path(__file__).resolve().parent
    # 假定三个脚本放在同一目录；若不在同一目录，改成对应路径
    script_dir = here

    # 1) 载入标签脚本并注入配置
    try:
        deal = _import_module("deal_iscxvpn_labels", script_dir)
    except Exception as e:
        raise SystemExit(f"无法导入 deal_iscxvpn_labels.py：{e}\n请确认脚本与本文件在同一目录。")

    deal.DATASET_KEY  = DATASET_KEY
    deal.DATASET_ROOT = DATASET_ROOT

    # 2) 运行标签阶段
    print("──────────────────────── deal_iscxvpn_labels → label_map.csv ────────────────────────")
    deal.main()

    # 3) 载入构建脚本并注入与第一阶段一致的配置
    try:
        build = _import_module("build_dirseq_dataset", script_dir)
    except Exception as e:
        raise SystemExit(f"无法导入 build_dirseq_dataset.py：{e}\n请确认脚本与本文件在同一目录。")

    build.DATASET_KEY        = DATASET_KEY
    build.PAYLOAD_ONLY       = PAYLOAD_ONLY
    build.MIN_LEN            = MIN_LEN
    build.TRUNCATE_LEN       = TRUNCATE_LEN
    build.EXCLUDE_L4         = set(EXCLUDE_L4)
    build.EXCLUDE_MCAST_IPS  = set(EXCLUDE_MCAST_IPS)
    build.USE_SHA256         = USE_SHA256

    # 4) 运行构建阶段
    print("──────────────────────── build_dirseq_dataset → data.npz ───────────────────────────")
    build.main()

    out_dir = Path("artifacts") / DATASET_KEY / "dirseq"
    print(f"\n[OK] 全流程完成，产物目录：{out_dir.resolve()}")

if __name__ == "__main__":
    # 友好检查
    if not Path(DATASET_ROOT).exists():
        print(f"[warn] DATASET_ROOT 不存在：{DATASET_ROOT}（将继续执行；如有问题请检查路径）")
    main()
