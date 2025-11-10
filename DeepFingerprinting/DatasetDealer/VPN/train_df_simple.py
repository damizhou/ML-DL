#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_df_simple.py — 极简可跑 + 逐行中文注释 + macro F1
====================================================
目标：给初学者的最小化训练脚本，尽量使用 PyTorch 默认配置，逻辑清晰。
- 不使用：加权采样、类权重、AMP、学习率调度、自定义 collate、复杂数据划分
- 仅做：随机切分 train/val/test（80/10/10），训练若干 epoch，最后在 test 上输出 loss / acc / macro F1
- Dataset 内在 __getitem__ 做右侧零填充/截断到 MAX_LEN，使 DataLoader 能用默认 collate
- 模型使用你现有的 DFNoDefNet；若分类头输出维度与数据集类别数不一致，会自动替换为正确的线性层

注意：
1) 需要在目录 DatasetDealer/ISCXVPN/artifacts/<DATASET_KEY>/dirseq/ 下准备 data.npz 和 labels.json
   - data.npz 应至少包含键：X(变长的 ±1/0 序列)、y(类别 id，从 0 开始)
   - labels.json 应包含键：{"id2label": {"0": "chat", "1": "email", ...}}
2) 本脚本聚焦“最小可跑”；后续想加速或提稳（AMP/分层切分/加权等），可以在此基础上逐项开启。
"""

from __future__ import annotations
import os
from tqdm import tqdm  # 新增
import math            # 新增
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import sys
from collections import OrderedDict  # 新增

ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet

# =====================
# 基本常量（保持很少很直观）
# =====================
DATASET_KEY = "iscx"                     # 数据集键名：决定读取的目录
ART_DIR     = os.path.join("artifacts",  # 根目录（按你的工程习惯）
    DATASET_KEY,
    "dirseq",
)
EPOCHS      = 30                          # 训练轮数（可按需改大/改小）
BATCH_SIZE  = 1024                        # 批大小（视显存调整）
MAX_LEN     = 5000                        # 每条样本的统一长度（右侧零填充或截断）
MULTI_NPZ_ROOT = '/home/pcz/DL/ML&DL/DeepFingerprinting/DatasetDealer/VPN/npz'
DEBUG_NPZ = 0
LOG_EVERY = int(os.getenv("LOG_EVERY", "200"))            # 每多少 step 刷新一次进度条后缀
MAX_STEPS_PER_EPOCH = int(os.getenv("MAX_STEPS_PER_EPOCH", "0"))  # 调试用：>0 时每轮仅跑前 N 个 step
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(max(2, os.cpu_count() // 2))))  # dataloader 进程数


torch.backends.cudnn.benchmark = True  # 新增：卷积算子自适应加速

class DirSeqDS(Dataset):
    """把变长 ±1 序列在 __getitem__ 时做右侧零填充/截断到 MAX_LEN，返回 float32 张量。

    好处：
    - DataLoader 可以使用默认 collate（自动把同 shape 的张量堆叠成 batch）
    - 这种“按需 pad”的策略更节省内存（不需要预先把所有样本都 pad 好）
    """
    def __init__(self, X: List[np.ndarray], y: np.ndarray, max_len: int):
        self.X = X                      # Python 列表，每个元素是 1D numpy 数组（变长）
        self.y = y                      # numpy 向量（类别 id，从 0 开始）
        self.max_len = max_len          # 统一的输出长度

    def __len__(self) -> int:
        return len(self.X)              # 样本总数

    def __getitem__(self, i: int):
        s = self.X[i]                   # 取出第 i 个变长序列（numpy 数组，元素通常为 -1/0/1）
        if len(s) >= self.max_len:      # 若超长，则直接截断
            a = s[: self.max_len]
        else:                           # 若不足，则右侧零填充
            a = np.zeros(self.max_len, dtype=np.int8)
            a[: len(s)] = s
        # 返回：特征张量（float32），标签（int）
        return torch.from_numpy(a.astype(np.float32)), int(self.y[i])

def _to_seq_list(x) -> List[np.ndarray]:
    """把各种形态的 X/flows 转成“变长 1D int8 向量”的列表。"""
    if isinstance(x, np.ndarray) and x.dtype == object:
        seqs = [np.asarray(s, dtype=np.int8).ravel() for s in x]
    elif isinstance(x, np.ndarray) and x.ndim == 2:
        seqs = [row.astype(np.int8).ravel() for row in x]
    else:
        seqs = [np.asarray(s, dtype=np.int8).ravel() for s in list(x)]
    return [s for s in seqs if s.size > 0]

class LazyNPZDataset(Dataset):
    def __init__(self,
                 file_paths: List[str],
                 file_ids: np.ndarray,
                 flow_idx: np.ndarray,
                 y_ids: np.ndarray,
                 max_len: int,
                 cache_files: int = 2):
        self.file_paths = file_paths
        self.file_ids   = file_ids.astype(np.int32, copy=False)
        self.flow_idx   = flow_idx.astype(np.int32, copy=False)
        self.y          = y_ids.astype(np.int64, copy=False)
        self.max_len    = max_len
        self.cache_max  = max(1, int(os.getenv("NPZ_CACHE_FILES", str(cache_files))))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()  # file_id -> flows(obj array)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def _get_flows(self, fid: int):
        if fid in self._cache:
            self._cache.move_to_end(fid)
            return self._cache[fid]
        # 只把该文件的 flows 读入内存
        with np.load(self.file_paths[fid], allow_pickle=True) as obj:
            flows = obj["flows"]
        self._cache[fid] = flows
        if len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)  # LRU 淘汰
        return flows

    def __getitem__(self, i: int):
        fid = int(self.file_ids[i])
        j   = int(self.flow_idx[i])
        flows = self._get_flows(fid)
        s = np.asarray(flows[j], dtype=np.int8)
        if s.size >= self.max_len:
            a = s[: self.max_len]
        else:
            a = np.zeros(self.max_len, dtype=np.int8)
            a[: s.size] = s
        return torch.from_numpy(a.astype(np.float32)), int(self.y[i])


def build_compact_index(root_dir: str):
    """返回:
       file_paths: List[str]               # 文件列表，按 file_id 对应
       file_ids:  np.ndarray[int32, N]     # 每个样本所属的 file_id
       flow_idx:  np.ndarray[int32, N]     # 在该文件中的 flow 下标
       y_ids:     np.ndarray[int32, N]     # 整数类标（对字符串标签做一次全局映射）
       id2label:  Dict[int,str]            # 类别id到原始标签
    只读取形状和 labels；不把 flows 留在内存。
    """
    root = Path(root_dir).expanduser().resolve()
    files = sorted(root.rglob("*.npz"))

    # -------- pass 1: 统计每个文件的条数，建立 file 列表 --------
    lengths = []  # (path_str, n_flow)
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as obj:
                if "flows" not in obj.files or "labels" not in obj.files:
                    continue
                n = len(obj["flows"])
                if n <= 0:
                    continue
                lengths.append((str(p), n))
        except Exception:
            continue
    if not lengths:
        raise RuntimeError("no usable npz found")

    total = int(sum(n for _, n in lengths))
    # 预分配紧凑索引
    file_ids = np.empty(total, dtype=np.int32)
    flow_idx = np.empty(total, dtype=np.int32)
    y_ids    = np.empty(total, dtype=np.int32)

    # -------- pass 2: 填充索引 + 统一标签到整数id --------
    label2id, id2label, next_id = {}, {}, 0
    off = 0
    file_paths = []
    for fid, (path, n) in enumerate(lengths):
        file_paths.append(path)
        try:
            with np.load(path, allow_pickle=True) as obj:
                labs = np.asarray(obj["labels"]).reshape(-1)
        except Exception:
            continue

        # labels 对齐
        if labs.size == 1:
            s = str(labs[0])
            if s not in label2id:
                label2id[s] = next_id
                id2label[next_id] = s
                next_id += 1
            y_ids[off:off+n] = label2id[s]
        elif labs.size == n:
            # 逐个映射（9428 类，字典很小，不是瓶颈）
            for i in range(n):
                s = str(labs[i])
                if s not in label2id:
                    label2id[s] = next_id
                    id2label[next_id] = s
                    next_id += 1
                y_ids[off+i] = label2id[s]
        else:
            # 长度异常，按首元素广播
            s = str(labs[0]) if labs.size > 0 else ""
            if s not in label2id:
                label2id[s] = next_id
                id2label[next_id] = s
                next_id += 1
            y_ids[off:off+n] = label2id[s]

        file_ids[off:off+n] = fid
        flow_idx[off:off+n] = np.arange(n, dtype=np.int32)
        off += n

    return file_paths, file_ids, flow_idx, y_ids.astype(np.int64), id2label

def _extract_xy_from_npz(obj: np.lib.npyio.NpzFile):
    """仅支持从 'flows' / 'labels' 提取样本。
    - flows: object 数组，每个元素为变长 1D 序列（int8）
    - labels: 与 flows 等长，或 size==1（按标量广播）
    返回:
      (X_list, y_array)；若不符合要求则返回 ([], None) 表示跳过
    """
    keys = set(obj.files)
    flows = obj["flows"]
    labels = np.asarray(obj["labels"]).reshape(-1)

    # flows 转为变长序列列表
    X_list = _to_seq_list(flows)
    n_flow = len(X_list)

    # labels 对齐：等长或标量广播，其它一律跳过
    if labels.size == n_flow:
        y_list = labels.tolist()
    elif labels.size == 1:
        y_list = [labels[0]] * n_flow
    else:
        return [], None  # 长度不匹配，跳过

    # 过滤空序列并保持标签对齐
    keep = [i for i, s in enumerate(X_list) if s.size > 0]
    if len(keep) != n_flow:
        X_list = [X_list[i] for i in keep]
        y_list = [y_list[i] for i in keep]

    # 注意：保留原始标签类型（可能是 str）；训练前再映射为整数
    return X_list, np.asarray(y_list, dtype=object)

def load_npz_tree(root_dir: str) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """递归读取 root_dir 下所有 .npz，聚合为 (X_list, y_raw_array, id2label=None)。
       注意：此处不做任何标签→id 的编码，保持原始(可能为字符串)标签。
    """
    debug = DEBUG_NPZ

    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NPZ root not found: {root}")

    X_all: List[np.ndarray] = []
    y_raw: List[object] = []
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found under: {root}")

    ok, bad, empty = 0, 0, 0
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as obj:
                Xi, yi = _extract_xy_from_npz(obj)  # yi: dtype=object，可能是 str/int
            if len(Xi) == 0:
                empty += 1
                if debug and empty <= 40:
                    print(f"[empty] {p}")
                continue
            X_all.extend(Xi)
            y_raw.extend(yi.tolist())
            ok += 1
        except Exception as e:
            bad += 1
            if debug and bad <= 40:
                print(f"[bad]   {p}: {type(e).__name__}: {e}")
            continue

    if not X_all:
        raise RuntimeError(
            f"Loaded 0 samples from {len(files)} files (ok={ok}, empty={empty}, bad={bad}). "
            f"开启 DEBUG_NPZ=1 可查看失败样例。"
        )

    print(f"[multi-npz] files={len(files)} ok={ok} empty={empty} bad={bad}  samples={len(X_all)}")
    return X_all, np.asarray(y_raw, dtype=object), None

def simple_split(n: int, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """返回 (tr_idx, va_idx, te_idx)，索引用 int32 以降低内存。"""
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    n_train = max(0, n - n_test - n_val)

    idx = np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)

    te = idx[:n_test]
    va = idx[n_test:n_test + n_val]
    tr = idx[n_test + n_val:n_test + n_val + n_train]
    return tr, va, te

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """最简单的分类准确率：预测==真实 的比例。"""
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> float:
    """计算 macro F1（不依赖第三方库）。

    定义：
      - 对每个类别 c 单独计算 F1_c = 2*TP / (2*TP + FP + FN)
      - 对所有类别取算术平均：macro_F1 = (1/ncls) * ∑_c F1_c
    约定：
      - 若对某类 c 而言 (2*TP + FP + FN) == 0（即该类在真值与预测中都没出现），将其 F1_c 记为 0（sklearn 的做法是可以通过参数控制，这里取直观处理）。
    """
    if y_true.size == 0:
        return 0.0
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    f1_list = []
    for c in range(ncls):
        # 逐类统计 TP/FP/FN
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1_c)
    return float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0

def main() -> None:
    print('starting...')
    # ---------- 设备选择：优先 GPU ----------
    if torch.cuda.is_available():
        print(f"Using CUDA device: GPU {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------- 读取数据：最小化校验 ----------
    file_paths, file_ids, flow_idx, y_ids, id2label = build_compact_index(MULTI_NPZ_ROOT)
    ncls = len(id2label)

    print(f"samples={len(y_ids)}  classes={ncls} (dir-aggregated, lazy index)")

    # ---------- Dataset / DataLoader（默认参数） ----------
    base = LazyNPZDataset(
        file_paths=file_paths,
        file_ids=file_ids,
        flow_idx=flow_idx,
        y_ids=y_ids,
        max_len=MAX_LEN,
        cache_files=int(os.getenv("NPZ_CACHE_FILES", "2"))  # 可用环境变量调大/调小
    )
    N = len(base)
    tr_idx, va_idx, te_idx = simple_split(N, val_ratio=0.1, test_ratio=0.1)

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=int(os.getenv("NUM_WORKERS", "2")),  # 先用 1~2，稳定后再升
        pin_memory=False,  # 先关，避免大 pinned 内存
        persistent_workers=False, prefetch_factor=1
    )
    train_loader = DataLoader(Subset(base, tr_idx), shuffle=True, **dl_kwargs)
    val_loader = DataLoader(Subset(base, va_idx), shuffle=False, **dl_kwargs)
    test_loader = DataLoader(Subset(base, te_idx), shuffle=False, **dl_kwargs)

    # 可视化每轮大致步数
    steps_per_epoch = math.ceil(len(tr_idx) / BATCH_SIZE)
    print(f"~steps/epoch = {steps_per_epoch}")

    # ---------- 模型：必要的分类头自适配 ----------
    model = DFNoDefNet()
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 .classifier 线性层 —— 无法确定输出类别数")
    if model.classifier.out_features != ncls:
        model.classifier = nn.Linear(model.classifier.in_features, ncls)
    model = model.to(device)

    # ---------- 损失/优化器：全部默认 ----------
    criterion = nn.CrossEntropyLoss().to(device)    # 默认：均匀类权重
    optimizer = torch.optim.Adam(model.parameters()) # 默认：lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())  # 新增

    # ---------- 训练循环 ----------
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"train {ep:02d}")
        for step, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = xb.size(0)
            run_loss += loss.item() * n
            run_n += n
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

            if step % LOG_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if MAX_STEPS_PER_EPOCH and step >= MAX_STEPS_PER_EPOCH:
                pbar.set_postfix_str(pbar.postfix + " | early-stop-epoch")
                break

        tr_loss = run_loss / max(1, run_n)
        tr_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        tr_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        tr_acc = accuracy(tr_y, tr_p)

        # ------ 验证（可选也加进度条） ------
        model.eval()
        with torch.no_grad():
            run_loss, run_n = 0.0, 0
            y_true_chunks, y_pred_chunks = [], []
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True);
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                n = xb.size(0)
                run_loss += loss.item() * n
                run_n += n
                y_true_chunks.append(yb.detach().cpu().numpy())
                y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            va_loss = run_loss / max(1, run_n)
            va_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
            va_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
            va_acc = accuracy(va_y, va_p)
            va_f1m = macro_f1(va_y, va_p, model.classifier.out_features)

        print(
            f"[{ep:02d}/{EPOCHS}] train {tr_loss:.4f}/{tr_acc:.4f}  |  val {va_loss:.4f}/{va_acc:.4f}  (val macroF1={va_f1m:.4f})")

    # ---------- Test ----------
    model.eval()
    with torch.no_grad():
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            n = xb.size(0)
            run_loss += loss.item() * n
            run_n += n
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        te_loss = run_loss / max(1, run_n)
        te_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        te_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        te_acc = accuracy(te_y, te_p)
        te_f1m = macro_f1(te_y, te_p, ncls)

    print(f"[TEST] loss={te_loss:.4f}  acc={te_acc:.4f}  macroF1={te_f1m:.4f}")


if __name__ == "__main__":
    main()
