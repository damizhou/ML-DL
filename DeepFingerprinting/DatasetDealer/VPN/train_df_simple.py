#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_df_simple.py — 多NPZ（flows/labels）+ 外部 labels.json 映射 + 懒加载（最小可运行）
环境变量：
  MULTI_NPZ_ROOT=/path/to/npz_root
  LABELS_JSON=/path/to/labels.json      # 可选，默认在 MULTI_NPZ_ROOT/labels.json
  EPOCHS=5  BATCH_SIZE=512  MAX_LEN=5000
  NUM_WORKERS=2  NPZ_CACHE_FILES=2
"""

from __future__ import annotations
import sys, json, torch
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from functools import lru_cache

torch.backends.cudnn.benchmark = True
# Enable TF32 for faster computation on Ampere+ GPUs (compatible with older PyTorch)
if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True

# —— 超参（保持很少）——
EPOCHS          = 30
BATCH_SIZE      = 512
MAX_LEN         = 5000
MULTI_NPZ_ROOT  = "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows_little"
NUM_WORKERS     = 8
NPZ_CACHE_FILES = 20000
LABELS_JSON     = "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows_little/npz_longflows_little.json"


# NPZ_ROOT: str = "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows_little"
# LABELS_JSON: str = "/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz_longflows_little/npz_longflows_little.json"

# —— 导入你的 DF 模型（保持工程结构）——
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFingerprinting
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model_NoDef_pytorch import DFNoDefNet  # noqa: E402


# ============ 工具 ============
def to_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return str(x)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0: return 0.0
    return float((y_true == y_pred).mean())

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, ncls: int) -> float:
    if y_true.size == 0: return 0.0
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    f1s = []
    for c in range(ncls):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        d = 2*tp + fp + fn
        f1s.append((2.0*tp/d) if d>0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def simple_split(n: int, val_ratio=0.1, test_ratio=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int32)
    np.random.default_rng(42).shuffle(idx)
    n_te = int(round(n*test_ratio)); n_va = int(round(n*val_ratio))
    te = idx[:n_te]; va = idx[n_te:n_te+n_va]; tr = idx[n_te+n_va:]
    return tr, va, te

def collate_fn_fast(batch):
    # batch: List[Tuple[x, y]]；x 允许是 list/np.ndarray/torch.Tensor 的 1D 序列
    xs, ys = zip(*batch)
    # 固定输出长度为 MAX_LEN，确保与模型 fc1 层输入维度匹配
    t_list = []
    for x in xs:
        # 创建固定长度的零数组
        padded = np.zeros(MAX_LEN, dtype=np.int8)
        if isinstance(x, np.ndarray):
            length = min(len(x), MAX_LEN)
            padded[:length] = x[:length]
        elif torch.is_tensor(x):
            x_np = x.numpy()
            length = min(len(x_np), MAX_LEN)
            padded[:length] = x_np[:length]
        else:  # list 等
            length = min(len(x), MAX_LEN)
            padded[:length] = x[:length]
        t_list.append(torch.from_numpy(padded))
    # 堆叠成 batch（所有序列长度相同，无需 pad_sequence）
    xb = torch.stack(t_list).to(torch.float32)
    yb = torch.as_tensor(ys, dtype=torch.int64)
    return xb, yb

# ============ 索引构建（仅读 labels，按 labels.json 映射为 id） ============
def load_label_mapping(labels_json_path: Path) -> Tuple[Dict[str,int], Dict[int,str]]:
    with labels_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    l2i = {str(k): int(v) for k, v in obj.get("label2id", {}).items()}
    i2l = {int(k): str(v) for k, v in obj.get("id2label", {}).items()}
    if not l2i or not i2l:
        raise ValueError(f"labels.json 缺少 label2id/id2label：{labels_json_path}")
    return l2i, i2l

def build_index(root_dir: Path, label2id: Dict[str,int]):
    """返回 (file_paths, file_ids, flow_idx, y_ids, ncls)
    全程不把 flows 驻内存，仅生成紧凑索引与整数标签数组。
    """
    files = sorted(root_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found under: {root_dir}")

    file_paths: List[str] = []
    file_ids: List[int] = []
    flow_idx: List[int] = []
    y_ids:    List[int] = []

    for path in tqdm(files, desc="scan npz", unit="file", mininterval=1.0):
        try:
            with np.load(path, allow_pickle=True) as obj:
                if "flows" not in obj.files or "labels" not in obj.files:
                    continue
                n = len(obj["flows"])
                if n <= 0:
                    continue
                labs = np.asarray(obj["labels"]).reshape(-1)
        except Exception:
            continue

        fid = len(file_paths)
        file_paths.append(str(path))

        if labs.size == 1:
            lid = label2id[to_str(labs[0])]
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            y_ids.extend([lid]*n)
        elif labs.size == n:
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            for j in range(n):
                lid = label2id[to_str(labs[j])]
                y_ids.append(lid)
        else:
            # 长度异常：按首元素广播
            lid = label2id[to_str(labs[0])] if labs.size>0 else label2id[to_str("")]
            file_ids.extend([fid]*n)
            flow_idx.extend(range(n))
            y_ids.extend([lid]*n)

    if not y_ids:
        raise RuntimeError("No usable samples after scanning.")

    return (
        file_paths,
        np.asarray(file_ids, dtype=np.int32),
        np.asarray(flow_idx, dtype=np.int32),
        np.asarray(y_ids,   dtype=np.int64),
        max(y_ids)+1
    )

# 放在文件前部（imports 之后、main 之前）
def evaluate(loader, model, criterion, device, use_amp: bool):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true_cpu, y_pred_cpu = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == yb).sum().item()
            total += yb.size(0)
            # 汇总到 CPU，便于可选的 F1 计算
            y_true_cpu.append(yb.detach().cpu())
            y_pred_cpu.append(pred.detach().cpu())
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)

    # 可选：若安装了 scikit-learn，则计算 F1（没有也不报错）
    f1m = f1mi = f1w = None
    try:
        from sklearn.metrics import f1_score
        yt = torch.cat(y_true_cpu).numpy()
        yp = torch.cat(y_pred_cpu).numpy()
        f1m  = f1_score(yt, yp, average="macro")
        f1mi = f1_score(yt, yp, average="micro")
        f1w  = f1_score(yt, yp, average="weighted")
    except Exception:
        pass
    return avg_loss, acc, f1m, f1mi, f1w
# ============ 懒加载 Dataset（使用 LRU 本地缓存） ============
# 全局 LRU 缓存函数（每个 worker 进程有独立缓存，避免跨进程通信开销）
@lru_cache(maxsize=NPZ_CACHE_FILES)
def _load_npz_flows(path: str):
    """加载 NPZ 文件的 flows 数据，使用 LRU 缓存避免重复读取"""
    with np.load(path, allow_pickle=True) as obj:
        return obj["flows"]


class LazyNPZDataset(Dataset):
    def __init__(self, file_paths: List[str], file_ids: np.ndarray, flow_idx: np.ndarray, y_ids: np.ndarray,
                 max_len: int):
        self.file_paths = file_paths
        self.file_ids = file_ids.astype(np.int32, copy=False)
        self.flow_idx = flow_idx.astype(np.int32, copy=False)
        self.y = y_ids.astype(np.int64, copy=False)
        self.max_len = max_len

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        fid = int(self.file_ids[i])
        j = int(self.flow_idx[i])
        flows = _load_npz_flows(self.file_paths[fid])
        s = np.asarray(flows[j], dtype=np.int8)
        return s, int(self.y[i])

# ============ 主程序 ============
def main():
    root_dir = Path(MULTI_NPZ_ROOT).expanduser().resolve()
    labels_json = Path(LABELS_JSON).expanduser().resolve() if LABELS_JSON else (root_dir / "labels.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device:", torch.cuda.get_device_name(0))

    label2id, id2label = load_label_mapping(labels_json)
    file_paths, file_ids, flow_idx, y_ids, ncls = build_index(root_dir, label2id)
    print(f"samples={len(y_ids)}  classes={ncls}")

    # 使用 LRU 本地缓存（每个 worker 进程独立缓存）
    base = LazyNPZDataset(file_paths, file_ids, flow_idx, y_ids, MAX_LEN)
    tr_idx, va_idx, te_idx = simple_split(len(base), 0.1, 0.1)

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn_fast,
    )
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(Subset(base, tr_idx), shuffle=True, **dl_kwargs)
    val_loader = DataLoader(Subset(base, va_idx), shuffle=False, **dl_kwargs)
    test_loader = DataLoader(Subset(base, te_idx), shuffle=False, **dl_kwargs)

    model = DFNoDefNet()
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("DFNoDefNet 缺少 .classifier 线性层")
    if model.classifier.out_features != ncls:
        model.classifier = nn.Linear(model.classifier.in_features, ncls)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # 混合精度（GPU 时启用）
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    for ep in range(1, EPOCHS + 1):
        # ------- Train -------
        model.train()
        run_loss, run_n = 0.0, 0
        y_true_chunks, y_pred_chunks = [], []

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"train {ep:02d}", mininterval=3.0)
        for step, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
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

            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 训练指标
        tr_loss = run_loss / max(1, run_n)
        tr_y = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int64)
        tr_p = np.concatenate(y_pred_chunks) if y_pred_chunks else np.empty((0,), dtype=np.int64)
        tr_acc = accuracy(tr_y, tr_p)

        # ------- Val -------
        use_amp = (device.type == "cuda")
        va_loss, va_acc, va_f1m, _, _ = evaluate(val_loader, model, criterion, device, use_amp)

        # ------- 日志 -------
        mf1 = "N/A" if va_f1m is None else f"{va_f1m:.4f}"
        print(f"[{ep:02d}/{EPOCHS}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} (macroF1={mf1})")
    # ======= 测试集评估 =======
    # 训练完评测测试集
    use_amp = (device.type == "cuda")
    test_loss, test_acc, f1m, f1mi, f1w = evaluate(test_loader, model, criterion, device, use_amp)
    if f1m is None:
        print(f"[TEST] loss={test_loss:.4f}  acc={test_acc:.4f}")
    else:
        print(
            f"[TEST] loss={test_loss:.4f}  acc={test_acc:.4f}  f1(macro/micro/weighted)={f1m:.4f}/{f1mi:.4f}/{f1w:.4f}")


if __name__ == "__main__":
    main()
