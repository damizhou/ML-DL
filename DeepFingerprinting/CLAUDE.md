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
├── run_train.py                # 统一训练脚本 (推荐使用)
├── checkpoints/                # 模型保存目录
│
└── DatasetDealer/              # 数据处理脚本
    ├── VPN/                    # VPN 流量处理
    ├── NOVPN/                  # 非 VPN 流量处理
    ├── ISCXVPN/                # ISCXVPN 数据集
    ├── ISCXTor/                # ISCXTor 数据集
    ├── USTC/                   # USTC-TFC 数据集
    ├── CIC_IOT_2022/           # CIC-IoT-2022 数据集
    └── Cross_Platform/         # 跨平台数据集
```

## 快速开始

### 1. 数据准备

数据处理脚本位于 `DatasetDealer/<dataset>/` 目录下：

```bash
# 以 ISCXVPN 为例
cd DatasetDealer/ISCXVPN
python build_dirseq_dataset.py
# 输出: artifacts/iscx/dirseq/data.npz + labels.json
```

### 2. 训练 (统一脚本)

```bash
# 编辑 run_train.py 中的 DATA_PATH
python run_train.py
```

### 关键配置 (run_train.py)
```python
# 数据路径 (自动检测格式)
DATA_PATH = Path("./data")

# 模型参数
MAX_LEN = 5000              # 固定输入长度

# 训练参数
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

# 数据划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
```

## 支持的数据格式

`run_train.py` 自动检测并支持三种数据格式：

### 格式 1: 单 NPZ 文件
```
data/
├── data.npz          # 包含 X (序列) 和 y (标签)
└── labels.json       # id2label 映射
```

### 格式 2: 统一目录格式
```
data/
├── chat.npz          # 每个类别一个 NPZ
├── email.npz         # 包含 flows 数组
├── ...
└── labels.json       # label2id + id2label
```

### 格式 3: 多 NPZ 目录 (懒加载)
```
data/
├── subdir1/
│   ├── file1.npz     # 包含 flows + labels
│   └── file2.npz
├── subdir2/
│   └── ...
└── labels.json
```

## 数据格式详解

### NPZ 文件结构
```python
# 格式 1 (data.npz)
{
    'X': np.ndarray(dtype=object),    # 变长方向序列列表
    'y': np.ndarray(dtype=int64)      # 类别标签
}

# 格式 2/3 (flows + labels)
{
    'flows': np.ndarray(dtype=object),  # 变长方向序列列表 (int8, ±1)
    'labels': np.ndarray                # 标签 (字符串或整数)
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

## 训练输出

```
======================================================================
DeepFingerprinting Training
======================================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4090

Data path: ./data
Data format: unified_dir

Loading data...
Total samples: 50000
Original classes: 12
Kept classes: 12
Split: train=40000, val=5000, test=5000

Model: DFNoDefNet
Parameters: 1,234,567

Epoch   1 | Train Loss: 2.3456 Acc: 0.1234 | Val Acc: 0.2345 F1: 0.2100 *
Epoch   2 | Train Loss: 1.8765 Acc: 0.3456 | Val Acc: 0.4567 F1: 0.4300 *
...

======================================================================
Final Evaluation on Test Set
======================================================================

Overall Results:
  Accuracy:  0.8500
  Precision: 0.8423
  Recall:    0.8367
  F1 Score:  0.8389
  TPR_AVE:   0.8367
  FPR_AVE:   0.0152

----------------------------------------------------------------------
Classification Report:
----------------------------------------------------------------------
              precision    recall  f1-score   support
        chat       0.85      0.87      0.86      1234
       email       0.82      0.80      0.81       567
         ...
```

## 训练优化

1. **混合精度**: 自动启用 AMP (float16) 加速 GPU 训练
2. **cuDNN 优化**: 已启用 `cudnn.benchmark = True`
3. **懒加载**: 大规模数据自动使用 LRU 缓存懒加载
4. **分层划分**: 确保每个类别按比例划分到 train/val/test

## 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Precision | 宏平均精确率 |
| Recall | 宏平均召回率 |
| F1 Score | 宏平均 F1 |
| TPR_AVE | 平均真阳性率 |
| FPR_AVE | 平均假阳性率 |

## 注意事项

1. **输入长度**: 必须固定为 5000，否则 FC 层维度不匹配
2. **最小样本数**: 类别样本少于 10 个会被自动剔除
3. **Windows 兼容**: 默认 `NUM_WORKERS = 0` 避免多进程问题
4. **GPU 内存**: 如果 OOM，减小 `BATCH_SIZE`
