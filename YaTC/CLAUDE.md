# CLAUDE.md - YaTC

本文件为 Claude Code 在 YaTC 项目中工作时提供指导。

## 项目概述

YaTC（Yet Another Traffic Classifier）是一个基于掩码自编码器的流量 Transformer 模型，使用多级流表示（MFR）进行加密流量分类。

**论文**: *Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation* (AAAI 2023)

## 模型架构

### 预训练阶段：MAE_YaTC (掩码自编码器)

YaTC 采用 **掩码自编码器 (Masked Autoencoder, MAE)** 进行自监督预训练，核心思想是随机遮挡输入的大部分内容，然后让模型重建被遮挡的部分。

```
输入: MFR 图像 (B, 1, 40, 40)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Patch Embedding                                      │
│    - patch_size = 2×2                                   │
│    - 每个数据包(8行) → 80 个 patch (4×20)               │
│    - 5 个数据包 → 总计 400 个 patch                     │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Random Masking (90%)                                 │
│    - 随机打乱 patch 顺序                                │
│    - 保留前 10% (40 个 patch)，遮挡 90% (360 个)        │
│    - 记录 ids_restore 用于后续恢复顺序                  │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Encoder (轻量高效)                                   │
│    - 4 层 Transformer Block                             │
│    - embed_dim = 192, num_heads = 16                    │
│    - 仅处理未被遮挡的 40 个 patch (计算量降低 90%)      │
│    - 添加 CLS token 和位置编码                          │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Decoder (重建)                                       │
│    - 线性映射: 192 → 128 维                             │
│    - 插入可学习的 mask_token 替代被遮挡位置             │
│    - 恢复完整 400 patch 序列                            │
│    - 2 层 Transformer Block (decoder_dim=128)           │
│    - 预测头: 128 → 4 (patch_size² × in_chans)           │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Reconstruction Loss                                  │
│    - 将预测与原始 patch 像素值比较                      │
│    - MSE Loss: (pred - target)²                         │
│    - 仅在被遮挡位置计算损失 (mask=1 的位置)             │
└─────────────────────────────────────────────────────────┘
```

**为什么使用 90% 掩码率？**
- 网络流量数据存在高度冗余性（协议头部、填充字节、重复模式）
- 高掩码率迫使模型学习更鲁棒的语义特征，而非简单的像素插值
- 计算效率高：Encoder 只需处理 10% 的 token，大幅降低计算成本

### 微调阶段：TraFormer_YaTC (分类器)

```
输入: MFR 图像 (B, 1, 40, 40)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Patch Embedding (与预训练相同)                       │
│    - 400 个 patch + CLS token + 位置编码                │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 加载预训练 Encoder 权重                              │
│    - 仅加载 Encoder 部分，丢弃 Decoder                  │
│    - 丢弃 mask_token, decoder_embed, decoder_blocks     │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Encoder (全部 patch 参与)                            │
│    - 4 层 Transformer Block                             │
│    - 处理完整 400 个 patch (无掩码)                     │
│    - 逐层学习率衰减 (layer_decay=0.65)                  │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Classification Head                                  │
│    - 提取 CLS token 作为全局表示                        │
│    - Linear: embed_dim → num_classes                    │
│    - CrossEntropy Loss                                  │
└─────────────────────────────────────────────────────────┘
```

## MFR 数据格式

将 PCAP 流转换为 40×40 灰度图像：
- 每个流取 **5 个数据包**
- 每个数据包: 160 字节头部 + 480 字节载荷 = **320 字节**
- 总计: 5 × 320 = 1600 字节 → **40×40 像素矩阵**

```
数据包 1: [头部 160B][载荷 480B] → 行 1-8
数据包 2: [头部 160B][载荷 480B] → 行 9-16
...
数据包 5: [头部 160B][载荷 480B] → 行 33-40
```

## 文件结构

```
YaTC/
├── Model_YaTC_pytorch.py         # 独立模型定义 (YaTC 分类器，含 Packet/Flow-level Attention)
│
├── refactor/                     # 重构版本 (推荐使用)
│   ├── models.py                 # MAE_YaTC (预训练) + TraFormer_YaTC (微调)
│   ├── config.py                 # 配置定义
│   ├── data.py                   # 数据加载
│   ├── engine.py                 # 训练/评估循环
│   ├── train.py                  # 训练入口
│   ├── pcap_to_mfr.py            # PCAP → MFR 转换
│   └── tests.py                  # 单元测试
│
├── iscx_vpn_processor.py         # ISCXVPN2016 数据处理
├── iscx_tor_processor.py         # ISCXTor2016 数据处理
├── ustc_processor.py             # USTC-TFC2016 数据处理
├── cic_iot_2022_processor.py     # CICIoT2022 数据处理
├── cross_platform_processor.py   # 跨平台数据处理
│
├── vpn_build_npz.py              # VPN 数据集 NPZ 构建
└── train_vpn.py                  # VPN 训练脚本
```

## 快速开始

### 1. 数据准备 (PCAP → MFR)
```bash
cd refactor
# 编辑 pcap_to_mfr.py 中的路径
python pcap_to_mfr.py
# 输出: data/<dataset>_MFR/train/class/*.png
#       data/<dataset>_MFR/test/class/*.png
```

### 2. 预训练 (MAE)
```bash
cd refactor
python train.py --mode pretrain \
    --batch_size 128 \
    --blr 1e-3 \
    --steps 150000 \
    --mask_ratio 0.9
# 输出: output_dir/YaTC_pretrained_model.pth
```

### 3. 微调 (分类)
```bash
python train.py --mode finetune \
    --blr 2e-3 \
    --epochs 200 \
    --data_path ./data/ISCXVPN2016_MFR \
    --nb_classes 7 \
    --finetune ./output_dir/YaTC_pretrained_model.pth \
    --layer_decay 0.65
```

## 关键参数

### 预训练参数
| 参数 | 默认值    | 说明 |
|------|--------|------|
| `--batch_size` | 512    | 批大小 |
| `--blr` | 1e-3   | 基础学习率 |
| `--steps` | 150000 | 训练步数 |
| `--mask_ratio` | 0.9    | 掩码比例 (90%) |
| `--warmup_epochs` | 40     | 预热 epochs |

### 微调参数
| 参数 | 默认值  | 说明 |
|------|------|------|
| `--batch_size` | 64   | 批大小 |
| `--blr` | 2e-3 | 基础学习率 |
| `--epochs` | 200  | 训练轮数 |
| `--data_path` | -    | MFR 数据集路径 |
| `--nb_classes` | -    | 类别数量 |
| `--finetune` | -    | 预训练模型路径 |
| `--layer_decay` | 0.65 | 逐层学习率衰减 |

## 支持的数据集

| 数据集 | 类别数 | 路径 |
|--------|--------|------|
| ISCXVPN2016 | 7 | `./data/ISCXVPN2016_MFR` |
| ISCXTor2016 | 8 | `./data/ISCXTor2016_MFR` |
| USTC-TFC2016 | 20 | `./data/USTC-TFC2016_MFR` |
| CICIoT2022 | 10 | `./data/CICIoT2022_MFR` |

## 数据集目录结构

```
data/ISCXVPN2016_MFR/
├── train/
│   ├── chat/
│   │   ├── flow_001.png
│   │   └── ...
│   ├── email/
│   └── ...
└── test/
    ├── chat/
    ├── email/
    └── ...
```

## 依赖版本

```
torch >= 1.9.0
numpy
PIL/Pillow
scikit-learn
```

**注意**: refactor 版本已移除对 `timm` 的依赖，所有模型组件均为原生 PyTorch 实现。

## 模型关键参数

```python
# MAE_YaTC (预训练)
img_size = 40
patch_size = 2
in_chans = 1           # 灰度图
embed_dim = 192
depth = 4              # Encoder 层数
num_heads = 16
decoder_embed_dim = 128
decoder_depth = 2

# TraFormer_YaTC (微调)
# 继承 MAE encoder 配置
# 添加分类头
```

## 注意事项

1. **预训练数据**: 预训练可以使用大规模无标签数据，不需要类别标签
2. **逐层衰减**: 微调时底层使用更小的学习率 (`layer_decay=0.65`)，让预训练特征更稳定
3. **掩码比例**: 90% 掩码比例是论文推荐值，对流量数据效果最好
4. **权重加载**: 微调时仅加载 Encoder 权重，Decoder 权重会被丢弃
5. **学习率调度**: 预训练和微调都使用 warmup + cosine decay 策略
