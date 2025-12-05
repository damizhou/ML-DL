#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_labels_json.py
- 递归扫描 root 下的 .npz（仅访问 'labels' 键），收集全量标签集合
- 将标签稳定映射到整数 id，写出 JSON：{"label2id": {...}, "id2label": {...}}
- 若输出文件已存在：在保留现有映射的前提下，仅为“新标签”追加新的 id
用法：
  python make_labels_json.py --root /path/to/npz_root --out /path/to/labels.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Set
import numpy as np

def to_str(x) -> str:
    """统一转字符串，兼容 bytes/标量/np类型。"""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return str(x)

def iter_labels_in_npz(npz_path: Path) -> Iterable[str]:
    """仅读取 .npz 中的 'labels'，不访问 'flows'。失败静默跳过。"""
    try:
        with np.load(npz_path, allow_pickle=True) as obj:
            if "labels" not in obj.files:
                return
            labs = np.asarray(obj["labels"]).reshape(-1)
            if labs.size == 0:
                return
            # 标量/向量一视同仁逐个输出
            for v in labs:
                yield to_str(v)
    except Exception:
        return  # 损坏或不兼容时跳过

def scan_all_labels(root: Path) -> Set[str]:
    """递归扫描 root 下所有 .npz 的 labels 集合。"""
    s: Set[str] = set()
    for p in root.rglob("*.npz"):
        for lab in iter_labels_in_npz(p):
            s.add(lab)
    return s

def load_existing(out_path: Path) -> Dict[str, int]:
    """读取已有 labels.json，返回 label2id；不抛错（文件缺失或不合法则返回空映射）。"""
    try:
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        l2i = data.get("label2id", {})
        # 只接受 str->int 的健壮映射
        return {str(k): int(v) for k, v in l2i.items()}
    except Exception:
        return {}

def build_mapping(labels: Set[str], existing: Dict[str, int]) -> Dict[str, int]:
    """在 existing 基础上为新标签追加 id；id 递增，保证确定性。"""
    if existing:
        next_id = max(existing.values()) + 1
        l2i = dict(existing)
    else:
        next_id = 0
        l2i = {}
    new_labels = sorted([lab for lab in labels if lab not in l2i])  # 稳定顺序
    for lab in new_labels:
        l2i[lab] = next_id
        next_id += 1
    return l2i

def invert_mapping(label2id: Dict[str, int]) -> Dict[str, str]:
    """生成 id2label（key 用字符串以便 JSON 稳定）。"""
    return {str(v): k for k, v in label2id.items()}

def main():
    root = Path(r"/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/vpn_npz_longflows_all")
    out  = root / "labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = scan_all_labels(root)
    existing = load_existing(out)
    label2id = build_mapping(labels, existing)
    id2label = invert_mapping(label2id)

    data = {"label2id": label2id, "id2label": id2label}
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[done] labels={len(labels)}  existing={len(existing)}  total={len(label2id)}")
    print(f"[path] {out}")

if __name__ == "__main__":
    main()
