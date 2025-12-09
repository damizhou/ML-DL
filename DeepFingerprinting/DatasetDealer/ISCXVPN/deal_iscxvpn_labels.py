#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deal_iscxvpn_labels1.py

- 递归扫描 DATASET_ROOT 下的 *.pcap / *.pcapng
- 直接输出 12 类最终标签（0..11），不生成细粒度中间态：
    0 chat, 1 email, 2 ft, 3 p2p, 4 stream, 5 voip,
    6 vpn-chat, 7 vpn-email, 8 vpn-ft, 9 vpn-p2p, 10 vpn-stream, 11 vpn-voip
- 针对你贴出的 140 个文件名做了归一化与正则匹配的补强：
  * 剥去末尾数字/字母后缀：1、2、1a、2b、A、B 等
  * 剥去末尾方向后缀：_up / _down（大小写）
  * VPN 判定优先使用父目录末尾标记：*_vpn / *_novpn
  * 兼容无下划线写法：facebookchat、gmailchat、icqchat、skypechat 等
  * 兼容 youtubeHTML5_1 → 归为 stream
"""

from __future__ import annotations
import os, re, csv, sys
from typing import List, Tuple, Optional, Dict
from collections import Counter

# ===== 固定数据集根目录 =====
DATASET_ROOT = "/home/pcz/DL/ML_DL/public_dataset/ISCX-VPN-NonVPN-2016"

# ===== 产物与扫描参数 =====
DATASET_KEY = "iscx"
VALID_SUFFIXES = (".pcap", ".pcapng")
IGNORE_HIDDEN = True

# ===== 12 类定义 =====
SERVICE_NAMES: List[str] = [
    "chat", "email", "ft", "p2p", "stream", "voip",
    "vpn-chat", "vpn-email", "vpn-ft", "vpn-p2p", "vpn-stream", "vpn-voip",
]
_BASE_ID: Dict[str, int] = {"chat": 0, "email": 1, "ft": 2, "p2p": 3, "stream": 4, "voip": 5}

# ===== 基础六类直接匹配（无需细粒度中间态）=====
PATTERNS: List[Tuple[re.Pattern, str]] = [
    # P2P
    (re.compile(r"(?:^|[_-])bittorrent(?:$|[_-])", re.I), "p2p"),
    (re.compile(r"(?:^|[_-])torrent(?:$|[_-])",    re.I), "p2p"),

    # Email
    (re.compile(r"(?:^|[_-])email(?:$|[_-])",           re.I), "email"),
    (re.compile(r"(?:^|[_-])email[_-]?client(?:$|[_-])",re.I), "email"),
    (re.compile(r"(?:^|[_-])gmail(?:$|[_-])",           re.I), "email"),

    # Chat（含无下划线写法）
    (re.compile(r"(?:^|[_-])aim[_-]?chat(?:$|[_-])|(?:^|[_-])aimchat(?:$|[_-])", re.I), "chat"),
    (re.compile(r"(?:^|[_-])facebook[_-]?chat(?:$|[_-])|(?:^|[_-])facebookchat(?:$|[_-])", re.I), "chat"),
    (re.compile(r"(?:^|[_-])gmail[_-]?chat(?:$|[_-])|(?:^|[_-])gmailchat(?:$|[_-])", re.I), "chat"),
    (re.compile(r"(?:^|[_-])hangouts?[_-]?chat(?:$|[_-])|(?:^|[_-])hangout[_-]?chat(?:$|[_-])", re.I), "chat"),
    (re.compile(r"(?:^|[_-])icq[_-]?chat(?:$|[_-])|(?:^|[_-])icqchat(?:$|[_-])", re.I), "chat"),
    (re.compile(r"(?:^|[_-])skype[_-]?chat(?:$|[_-])|(?:^|[_-])skypechat(?:$|[_-])", re.I), "chat"),

    # File Transfer（兼容 up/down 以及 skype_file/files）
    (re.compile(r"(?:^|[_-])sftp(?:[_-]?(?:up|down))?(?:$|[_-])", re.I), "ft"),
    (re.compile(r"(?:^|[_-])scp(?:[_-]?(?:up|down))?(?:$|[_-])",  re.I), "ft"),
    (re.compile(r"(?:^|[_-])ftps(?:[_-]?(?:up|down))?(?:$|[_-])", re.I), "ft"),
    (re.compile(r"(?:^|[_-])skype[_-]?files?(?:$|[_-])",          re.I), "ft"),

    # VoIP（实时通话）
    (re.compile(r"(?:^|[_-])skype[_-]?audio(?:$|[_-])",   re.I), "voip"),
    (re.compile(r"(?:^|[_-])skype[_-]?video(?:$|[_-])",   re.I), "voip"),
    (re.compile(r"(?:^|[_-])hangouts?[_-]?audio(?:$|[_-])", re.I), "voip"),
    (re.compile(r"(?:^|[_-])hangouts?[_-]?video(?:$|[_-])", re.I), "voip"),
    (re.compile(r"(?:^|[_-])voipbuster(?:$|[_-])",        re.I), "voip"),

    # Streaming（播放）
    (re.compile(r"(?:^|[_-])facebook[_-]?audio(?:$|[_-])", re.I), "stream"),
    (re.compile(r"(?:^|[_-])facebook[_-]?video(?:$|[_-])", re.I), "stream"),
    (re.compile(r"(?:^|[_-])netflix(?:$|[_-])",            re.I), "stream"),
    (re.compile(r"(?:^|[_-])vimeo(?:$|[_-])",              re.I), "stream"),
    (re.compile(r"(?:^|[_-])spotify(?:$|[_-])",            re.I), "stream"),
    # 兼容 youtubeHTML5_* 以及普通 youtube*
    (re.compile(r"(?:^|[_-])youtube(?:html5)?(?:$|[_-])",  re.I), "stream"),
]

# ===== 小工具 =====
def _is_hidden(name: str) -> bool:
    return name.startswith(".")

def _split_path_lower(path: str) -> List[str]:
    comps: List[str] = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            comps.append(tail.lower())
        else:
            if path:
                comps.append(path.lower())
            break
        if not path:
            break
    comps.reverse()
    return comps

def _strip_tail_tokens(s: str) -> str:
    """
    归一化文件名（不含扩展名）：
      1) 小写
      2) 去掉末尾方向后缀：_up/_down（大小写）
      3) 去掉末尾数字 + 可选 a/b（大小写）
      4) 去掉末尾的 _a/_b（大小写），用于 vpn_*_A / vpn_*_B
    """
    s = s.lower()
    s = re.sub(r"(?:[_-](?:up|down))$", "", s, flags=re.I)
    s = re.sub(r"\d+[ab]?$", "", s, flags=re.I)
    s = re.sub(r"[_-][ab]$", "", s, flags=re.I)
    return s

def infer_vpn_flag(path: str) -> bool:
    """
    True -> vpn, False -> novpn
    判定优先级：
      1) 父目录名末尾 *_vpn / *_novpn（最强）
      2) 文件名以 vpn_ / vpn- 起始
      3) 目录组件中独立出现 vpn / nonvpn（避免 nonvpn 抢占）
    """
    parent = os.path.basename(os.path.dirname(path)).lower()
    if re.search(r"(?:^|[_-])vpn$", parent):
        return True
    if re.search(r"(?:^|[_-])novpn$", parent) or re.search(r"(?:^|[^a-z])nonvpn([^a-z]|$)", parent):
        return False

    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem.startswith(("vpn_", "vpn-")):
        return True

    for c in _split_path_lower(path):
        if re.fullmatch(r"vpn", c):
            return True
        if re.fullmatch(r"novpn|nonvpn", c):
            return False
    return False  # 默认非 VPN

def detect_base_service(stem: str) -> Optional[str]:
    """
    直接基于“文件名（不含扩展名）”判断基础六类：chat/email/ft/p2p/stream/voip
    """
    s = _strip_tail_tokens(stem)
    for rx, base in PATTERNS:
        if rx.search(s):
            return base
    return None

def infer_service_id_from_path(path: str) -> Optional[Tuple[int, str]]:
    base = detect_base_service(os.path.splitext(os.path.basename(path))[0])
    if not base:
        return None
    sid = _BASE_ID[base] + (6 if infer_vpn_flag(path) else 0)
    return sid, SERVICE_NAMES[sid]

def walk_dataset(root: str) -> List[str]:
    out: List[str] = []
    for d, subdirs, files in os.walk(root):
        if IGNORE_HIDDEN:
            subdirs[:] = [s for s in subdirs if not _is_hidden(s)]
        for fn in files:
            if IGNORE_HIDDEN and _is_hidden(fn):
                continue
            if not fn.lower().endswith(VALID_SUFFIXES):
                continue
            out.append(os.path.join(d, fn))
    return out

def ensure_dir_for_file(p: str) -> None:
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ===== 主流程 =====
def main() -> None:
    root = DATASET_ROOT
    if not os.path.isdir(root):
        print(f"[error] DATASET_ROOT 不存在: {root}", file=sys.stderr)
        sys.exit(1)

    candidates = walk_dataset(root)
    mapped: List[Tuple[str, int]] = []
    skipped = 0

    for path in candidates:
        r = infer_service_id_from_path(path)
        if r is None:
            skipped += 1
            continue
        sid, _ = r
        mapped.append((path, sid))

    out_dir = os.path.join("../ISCXVPN/artifacts", DATASET_KEY)
    out_csv = os.path.join(out_dir, "label_map.csv")
    vocab_csv = os.path.join(out_dir, "service_vocab.csv")
    ensure_dir_for_file(out_csv)
    ensure_dir_for_file(vocab_csv)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["path", "label"])
        for p, sid in sorted(mapped, key=lambda x: (x[1], x[0])): wr.writerow([p, sid])

    with open(vocab_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["service_id", "service"])
        for i, name in enumerate(SERVICE_NAMES): wr.writerow([i, name])

    cnt = Counter([sid for _, sid in mapped])
    print(f"[root ] {root}")
    print(f"[done ] write -> {out_csv}")
    print(f"[info ] vocab  -> {vocab_csv}")
    print(f"[stats] files_total={len(candidates)}  labeled={len(mapped)}  skipped_unmatched={skipped}")
    print("[by service_id]")
    for i, name in enumerate(SERVICE_NAMES): print(f"  {i:2d} {name:>10s} : {cnt.get(i, 0)}")

if __name__ == "__main__":
    main()
