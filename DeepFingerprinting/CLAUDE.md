# CLAUDE.md - DeepFingerprinting

本文件为 Claude Code 在 DeepFingerprinting 项目中工作时提供指导。

## 项目概述

DeepFingerprinting (DF) 是一种用于网站指纹攻击的深度学习模型，使用 1D CNN 处理流量方向序列。

**论文**: *Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning* (CCS 2018)

## 模型架构

```
输入: 方向序列 (长度 5000, 值为 ±1)
  ↓
Block 1: Conv1d(1→32, k=8) → BN → ELU → Conv → BN → ELU → MaxPool(8,4) → Dropout(0.1)
  ↓
Block 2: Conv1d(32→64, k=8) → BN → ReLU → Conv → BN → ReLU → MaxPool(8,4) → Dropout(0.1)
  ↓
Block 3: Conv1d(64→128, k=8) → BN → ReLU → ... → MaxPool(8,4) → Dropout(0.1)
  ↓
Block 4: Conv1d(128→256, k=8) → BN → ReLU → ... → MaxPool(8,4) → Dropout(0.1)
  ↓
Flatten → FC(512) → BN → ReLU → Dropout(0.7)
  ↓
FC(512) → BN → ReLU → Dropout(0.5)
  ↓
Classifier → num_classes
```

## 文件结构

```
DeepFingerprinting/
├── Model_NoDef_pytorch.py      # DFNoDefNet 模型定义
│
└── DatasetDealer/              # 数据处理脚本
    ├── VPN/                    # VPN 流量处理
    │   ├── df_build_npz_with_vpn.py   # PCAP → NPZ 转换 (多进程)
    │   ├── train_df_simple.py         # 训练脚本 (懒加载)
    │   └── make_labels_json.py        # 生成 labels.json
    │
    ├── NOVPN/                  # 非 VPN 流量处理
    │   ├── df_build_npz_with_novpn.py
    │   └── kimi_train.py
    │
    └── ISCXVPN/                # ISCXVPN 数据集
        ├── build_dirseq_dataset.py    # PCAP → 方向序列
        └── train_df_simple.py         # 简化训练脚本
```

## 快速开始

### 1. 数据准备 (PCAP → NPZ)

```bash
cd DatasetDealer/VPN
# 编辑 df_build_npz_with_vpn.py 中的路径配置
python df_build_npz_with_vpn.py
# 输出: vpn_npz_longflows_all/<label>/*.npz
```

### 2. 生成标签映射

```bash
python make_labels_json.py
# 输出: labels.json
```

### 3. 训练

```bash
python train_df_simple.py
```

### 关键配置 (train_df_simple.py)
```python
EPOCHS = 30
BATCH_SIZE = 512
MAX_LEN = 5000                # 固定输入长度
MULTI_NPZ_ROOT = "/path/to/npz_dir"
LABELS_JSON = "/path/to/labels.json"
NUM_WORKERS = 8
NPZ_CACHE_FILES = 20000       # LRU 缓存大小
```

## 数据格式

### NPZ 文件结构
```python
{
    'flows': np.ndarray(dtype=object),   # 变长方向序列列表 (int8, ±1)
    'labels': np.ndarray(dtype=object)   # 对应标签字符串
}
```

### 方向序列编码
- `+1`: 出站数据包
- `-1`: 入站数据包
- `0`: padding

### labels.json 格式
```json
{
    "label2id": {"chat": 0, "email": 1, ...},
    "id2label": {"0": "chat", "1": "email", ...}
}
```

## 数据处理流程

```
PCAP 文件
    ↓ (df_build_npz_with_vpn.py)
按流聚合 → 提取方向序列 → 过滤短流 (>200包)
    ↓
保存为 NPZ (每个 PCAP 一个 NPZ)
    ↓ (train_df_simple.py)
扫描所有 NPZ → 构建索引 → 懒加载训练
```

## 训练优化

1. **混合精度**: 自动启用 AMP (float16)
2. **TF32 加速**: 已启用 (Ampere+ GPU)
3. **LRU 缓存**: 每个 worker 独立缓存，避免 IPC 开销
4. **固定长度**: collate_fn 确保输出固定为 MAX_LEN

## 评估指标

```
[TEST] loss=0.xxxx  acc=0.xxxx  f1(macro/micro/weighted)=0.xxxx/0.xxxx/0.xxxx
```

## 注意事项

1. **输入长度**: 必须固定为 5000，否则 FC 层维度不匹配
2. **流过滤**: 默认只保留 >200 包的流
3. **文件过滤**: 跳过 <20KB 的 PCAP 文件
4. **torch.compile**: PyTorch 2.0+ 自动启用编译优化
