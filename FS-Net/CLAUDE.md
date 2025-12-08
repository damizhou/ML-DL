# CLAUDE.md - FS-Net

本文件为 Claude Code 在 FS-Net 项目中工作时提供指导。

## 项目概述

FS-Net（Flow Sequence Network）是一种用于加密流量分类的深度学习模型，使用双向 GRU 处理数据包长度序列。

**论文**: *FS-Net: A Flow Sequence Network For Encrypted Traffic Classification* (INFOCOM 2019)

## 模型架构

```
输入: 数据包长度序列 (带方向: ±1~±1500)
  ↓
Embedding Layer (vocab_size=3002, embed_dim=128)
  ↓
Encoder (2层 Bi-GRU, hidden=128)
  ↓
Decoder (2层 Bi-GRU, hidden=128) → 重建损失
  ↓
Dense Layer: z = [ze, zd, ze⊙zd, |ze-zd|] → 256维
  ↓
Classifier → num_classes
```

**损失函数**: `L = L_class + α × L_recon` (α=1.0)

## 文件结构

```
FS-Net/
├── models.py          # FSNet, FSNetND (无解码器变体)
├── config.py          # 超参数配置
├── data.py            # 数据加载 (pickle/PCAP)
├── engine.py          # train_one_epoch, evaluate
├── run_train.py       # 训练入口 (硬编码参数)
├── iscx_processor.py  # ISCXVPN → pickle 转换
└── checkpoints/       # 模型保存目录
```

## 快速开始

### 1. 数据准备
```bash
# 将 PCAP 数据转换为 pickle 格式
python iscx_vpn_processor.py
# 输出: data/iscxvpn/iscxvpn_fsnet.pkl
```

### 2. 训练
```bash
python run_train.py
```

### 关键配置 (run_train.py)
```python
DATA_PATH = "/path/to/iscxvpn_fsnet.pkl"
NUM_CLASSES = 12          # ISCXVPN: 12类
BATCH_SIZE = 2048         # RTX 4090 推荐
EPOCHS = 200
LEARNING_RATE = 0.0005
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
ALPHA = 1.0               # 重建损失权重
```

## 数据格式

### Pickle 文件结构
```python
{
    'sequences': List[np.ndarray],  # 变长序列 (int8, ±值表示方向)
    'labels': np.ndarray,           # 类别标签 (int64)
    'label_map': Dict[int, str]     # {0: 'class_name', ...}
}
```

### 数据包长度编码
- 正值 (+1 ~ +1500): 出站数据包
- 负值 (-1 ~ -1500): 入站数据包
- Embedding 索引: `length + 1501` (范围 1~3001, 0 为 padding)

## 评估指标

输出示例:
```
================================================================================
Final Evaluation on Test Set
================================================================================
Overall Results:
  Accuracy:  0.7252
  Precision: 0.7327
  Recall:    0.7085
  F1 Score:  0.7170
  TPR_AVE:   0.7252
  FPR_AVE:   0.0603

--------------------------------------------------------------------------------
Per-Class Results:
--------------------------------------------------------------------------------
Class                   Count  Precision     Recall         F1        TPR        FPR
--------------------------------------------------------------------------------
AIM_Chat                  xxx     0.xxxx     0.xxxx     0.xxxx     0.xxxx     0.xxxx
...
```

## 注意事项

1. **GPU 优化**: 已启用 cuDNN (`torch.backends.cudnn.enabled = True`)
2. **Windows 兼容**: `NUM_WORKERS = 0` 避免多进程问题
3. **Early Stopping**: 已禁用，完整运行 200 epochs
4. **FSNet-ND**: 设置 `USE_NO_DECODER = True` 使用无解码器变体
