#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_dir_sequences.py
查看/导出 pcap 处理后的“包方向序列”（±1，未 padding）。
- 从 outputs_binary/binary.npz 读取 X,y（X 为变长 ±1 列表）
- 控制台打印摘要 + 每类若干条样本的 head/tail 预览
- 将完整序列逐条写入 outputs_binary/dumps/*.txt（空格分隔）
"""

from __future__ import annotations
import os, json, csv
from typing import List, Dict, Tuple, Optional
import numpy as np

# ===== 路径与导出策略（零参数可运行） =====
OUTDIR        = "outputs_binary"
NPZ_PATH      = os.path.join(OUTDIR, "binary.npz")
LABELS_JSON   = os.path.join(OUTDIR, "labels.json")  # 没有也可运行

# 导出模式一：按“类名”各导出若干条（常用）
DUMP_BY_CLASS_NAMES = True
CLASS_NAMES         = ["sftp", "abc.com"]  # 想看的类名；留空 [] 表示导出所有类别
PER_CLASS           = 10                    # 每个类别导出几条

# 导出模式二：按“样本索引”精确导出（与上面互不冲突，可同时启用）
DUMP_BY_INDEXES     = []                   # 例如 [0, 15, 200]

# 控制台预览长度
PRINT_HEAD          = 64
PRINT_TAIL          = 16
# =====================================

def load_npz(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=True)
    X, y = obj["X"], obj["y"].astype(np.int64)
    # 统一成 Python list[np.ndarray[int8]]
    if isinstance(X, np.ndarray) and X.dtype != object:
        X = [np.asarray(x, dtype=np.int8) for x in X]
    else:
        X = [np.asarray(x, dtype=np.int8) for x in list(X)]
    return X, y

def load_labels(labels_json: str) -> Dict[int, str]:
    if not os.path.exists(labels_json):
        return {}
    with open(labels_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {int(k): v for k, v in m.get("id2label", {}).items()}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def preview(seq: np.ndarray, head:int=64, tail:int=16) -> str:
    if len(seq) <= head + tail:
        return " ".join(map(str, seq.tolist()))
    return f"{' '.join(map(str, seq[:head].tolist()))} ... {' '.join(map(str, seq[-tail:].tolist()))}"

def main():
    dumps_dir = ensure_dir(os.path.join(OUTDIR, "dumps"))

    X, y = load_npz(NPZ_PATH)
    id2label = load_labels(LABELS_JSON)
    n = len(X)
    num_classes = int(np.max(y)) + 1

    # — 摘要 —
    lens = np.array([len(s) for s in X], dtype=np.int64)
    print("== Summary ==")
    print(f"N={n}  classes={num_classes}  y.shape={y.shape}  y.dtype={y.dtype}")
    q = np.percentile(lens, [0,10,50,90,95,100]).astype(int)
    # 注意：这里 X 是未 padding 的原始序列，理论上只包含 -1 和 +1
    values = sorted(set(int(v) for seq in X for v in (seq[:PRINT_HEAD] if len(seq) else [])))
    print(f"len stats: min={q[0]}  p10={q[1]}  median={q[2]}  p90={q[3]}  p95={q[4]}  max={q[5]}")
    print(f"value sample (期待仅有 -1 与 +1): {values[:5]}{'...' if len(values)>5 else ''}")

    # — 类别统计 —
    print("\n== Class counts ==")
    counts = np.bincount(y, minlength=num_classes)
    for cid, cnt in enumerate(counts):
        name = id2label.get(cid, str(cid))
        print(f"[{cid:2d}] {name:20s}  count={cnt:4d}")

    # — 要导出的目标索引集合 —
    export_idx = set(DUMP_BY_INDEXES)
    if DUMP_BY_CLASS_NAMES:
        # 若 CLASS_NAMES 为空，则导出全部类别
        target_names = CLASS_NAMES or [id2label.get(i, str(i)) for i in range(num_classes)]
        name2id = {v:k for k,v in id2label.items()}
        for name in target_names:
            if name in name2id:
                cid = name2id[name]
            else:
                # 如果 labels.json 没有，尝试把 name 当作数字字符串
                try:
                    cid = int(name)
                except Exception:
                    print(f"[warn] 未找到类名：{name}，已跳过")
                    continue
            idxs = np.where(y == cid)[0]
            if len(idxs) == 0:
                print(f"[warn] 类别无样本：{name}({cid})")
                continue
            take = min(PER_CLASS, len(idxs))
            export_idx.update(idxs[:take].tolist())

    export_idx = sorted(i for i in export_idx if 0 <= i < n)
    if not export_idx:
        print("\n[info] 没有需要导出的样本索引（请配置 CLASS_NAMES 或 DUMP_BY_INDEXES）")
        return

    print(f"\n== Export {len(export_idx)} samples ==")
    for i in export_idx:
        seq = X[i]; lab = int(y[i]); lab_name = id2label.get(lab, str(lab))
        # 控制台预览
        print(f"- idx={i:6d}  label={lab}({lab_name})  len={len(seq)}")
        print(f"  seq: {preview(seq, PRINT_HEAD, PRINT_TAIL)}")
        # 写完整序列（空格分隔，一行）
        out_txt = os.path.join(dumps_dir, f"idx{i}_label{lab}_{lab_name}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, seq.tolist())))
        # 也写成 .npy（方便 Python 直接载入）
        np.save(os.path.join(dumps_dir, f"idx{i}_label{lab}_{lab_name}.npy"), seq.astype(np.int8))

    print(f"\n[write] 完整序列 -> {dumps_dir}/idx*_label*_*.txt 以及同名 .npy")
    print("[tip] 文本文件内是“未 padding”的原始方向序列，仅由 -1/+1 组成；"
          "训练时才会右侧补 0 到固定长度。")

if __name__ == "__main__":
    main()
