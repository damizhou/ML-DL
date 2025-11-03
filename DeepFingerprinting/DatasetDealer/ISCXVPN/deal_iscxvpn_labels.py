#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deal_iscxvpn_labels.py
输入: ISCX-VPN-NonVPN-2016 的根目录（默认已写在代码里，也可命令行覆盖）
功能: 递归扫描根目录下的 novpn/vpn 子树，按文件名规则自动生成 (path,label) 映射表
输出: artifacts/<DATASET_KEY>/label_map.csv
"""

from __future__ import annotations
import os, re, csv, sys
from typing import List, Tuple, Optional, Dict
from collections import Counter

# ========= 参数（默认可直接运行） =========
DATASET_KEY   = "iscx"  # 产物目录名: artifacts/iscx/
DATASET_ROOT  = "/netdisk/dataset/public_dataset/ISCX-VPN-NonVPN-2016"  # 根目录
PCAP_EXTS     = (".pcap", ".pcapng", ".pcap.gz", ".pcapng.gz")


# 文件名 -> 标签（{prefix} 会替换为 vpn/novpn）
AUTO_LABEL_RULES: List[Tuple[str, str]] = [
    (r"(?:^|_)aim[_-]?chat",                  "{prefix}_aim_chat"),
    (r"(?:^|_)email",                         "{prefix}_email"),
    (r"(?:^|_)facebook[_-]?audio",            "{prefix}_facebook_audio"),
    (r"(?:^|_)facebook[_-]?chat|facebookchat","{prefix}_facebook_chat"),
    (r"(?:^|_)facebook[_-]?video",            "{prefix}_facebook_video"),
    (r"(?:^|_)ftps[_-]?down",                 "{prefix}_ftps_down"),
    (r"(?:^|_)ftps[_-]?up",                   "{prefix}_ftps_up"),
    (r"(?:^|_)gmail[_-]?chat",                "{prefix}_gmailchat"),
    (r"(?:^|_)hangouts?[_-]?audio",           "{prefix}_hangouts_audio"),
    (r"(?:^|_)hangouts?[_-]?chat|hangout[_-]?chat", "{prefix}_hangouts_chat"),
    (r"(?:^|_)hangouts?[_-]?video",           "{prefix}_hangouts_video"),
    (r"(?:^|_)icq[_-]?chat|icqchat",          "{prefix}_icq_chat"),
    (r"(?:^|_)netflix",                       "{prefix}_netflix"),
    (r"(?:^|_)scp[_-]?down|scpDown",          "{prefix}_scp_down"),
    (r"(?:^|_)scp[_-]?up|scpUp",              "{prefix}_scp_up"),
    (r"(?:^|_)sftp[_-]?down",                 "{prefix}_sftp_down"),
    (r"(?:^|_)sftp[_-]?up",                   "{prefix}_sftp_up"),
    (r"(?:^|_)sftp(\d|$)",                    "{prefix}_sftp"),
    (r"(?:^|_)skype[_-]?audio",               "{prefix}_skype_audio"),
    (r"(?:^|_)skype[_-]?chat",                "{prefix}_skype_chat"),
    (r"(?:^|_)skype[_-]?file",                "{prefix}_skype_file"),
    (r"(?:^|_)skype[_-]?video",               "{prefix}_skype_video"),
    (r"(?:^|_)spotify",                       "{prefix}_spotify"),
    (r"(?:^|_)vimeo",                         "{prefix}_vimeo"),
    (r"(?:^|_)voipbuster",                    "{prefix}_voipbuster"),
    (r"(?:^|_)youtube",                       "{prefix}_youtube"),
    # VPN 特例归一（如 vpn_ftps_A/B → vpn_ftps）
    (r"^vpn[_-]?ftps[_-]?[ab]\b",             "vpn_ftps"),
    (r"^vpn[_-]?sftp[_-]?[ab]\b",             "vpn_sftp"),
    (r"^vpn[_-]?vimeo[_-]?[ab]\b",            "vpn_vimeo"),
    (r"^vpn[_-]?skype[_-]?files",             "vpn_skype_file"),
    (r"^vpn[_-]?bittorrent",                  "vpn_bittorrent"),
]
# =====================================

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def infer_prefix(path: str) -> str:
    """
    仅按目录组件/文件名词级匹配来判断 vpn / novpn：
    - 优先匹配 novpn，其次匹配 vpn
    - 只看词边界（_ - . /），避免 NonVPN / _novpn 这种误命中
    """
    p = os.path.normpath(path).lower()
    parts = p.split(os.sep)

    # 只检查末端的几级（父目录/文件名），越近越可信
    candidates = []
    if len(parts) >= 3:
        candidates.extend(parts[-3:])
    else:
        candidates.extend(parts)

    # 文件名去扩展名后也加入一次
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    candidates.append(stem)

    # 先找 novpn，再找 vpn（词级匹配）
    novpn_pat = re.compile(r'(^|[._-])novpn($|[._-])')
    vpn_pat   = re.compile(r'(^|[._-])vpn($|[._-])')

    for comp in candidates:
        if novpn_pat.search(comp):
            return "novpn"
    for comp in candidates:
        if vpn_pat.search(comp):
            return "vpn"

    # 回退：文件名明确以 vpn_ 开头
    if stem.startswith("vpn_"):
        return "vpn"

    # 再回退默认 novpn
    return "novpn"

def infer_label(path: str) -> Optional[str]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    s = stem.lower()
    pref = infer_prefix(path)
    for pat, tmpl in AUTO_LABEL_RULES:
        if re.search(pat, s, re.IGNORECASE):
            return tmpl.format(prefix=pref)
    # 回退：若文件名以 vpn_ 开头，去掉常见后缀 a/b/1a/1b 直接用
    if s.startswith("vpn_"):
        for cut in ("_1a","_1b","_2a","_2b","_a","_b"):
            if s.endswith(cut):
                s = s[: -len(cut)]
                break
        return s
    # 其余无法识别的返回 None（跳过）
    return None

def walk_root(root: str) -> List[str]:
    """从根目录递归收集所有 pcap 文件。"""
    files: List[str] = []
    root = os.path.abspath(os.path.expanduser(root))
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith(PCAP_EXTS):
                files.append(os.path.join(dp, fn))
    return sorted(set(files))

def main():
    # 允许命令行覆盖根目录：python deal_iscxvpn_labels.py /path/to/ISCX-VPN-NonVPN-2016
    root = DATASET_ROOT
    if len(sys.argv) >= 2:
        root = sys.argv[1]

    out_dir = ensure_dir(os.path.join("artifacts", DATASET_KEY))
    out_csv = os.path.join(out_dir, "label_map.csv")

    # 1) 扫描根目录
    candidates = walk_root(root)

    # 2) 自动贴标签
    rows: List[Tuple[str, str]] = []
    misses = 0
    for p in candidates:
        lab = infer_label(p)
        if lab is None:
            misses += 1
            continue
        rows.append((p, lab))

    # 3) 去重（后写覆盖先写）
    mapping: Dict[str, str] = {}
    for p, lab in rows:
        mapping[p] = lab

    # 4) 写 CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "label"])
        for p, lab in sorted(mapping.items()):
            wr.writerow([p, lab])

    # 5) 打印统计
    cnt = Counter(mapping.values())
    print(f"[root ] {root}")
    print(f"[done ] write -> {out_csv}")
    print(f"[stats] files_total={len(candidates)}  labeled={len(mapping)}  skipped_unmatched={misses}")
    print("[by label]")
    for k in sorted(cnt):
        print(f"  {k:24s} : {cnt[k]}")

if __name__ == "__main__":
    main()
