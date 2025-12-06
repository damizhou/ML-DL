# CLAUDE.md - YaTC

本文件为 Claude Code 在 YaTC 项目中工作时提供指导。

## 项目概述

YaTC（Yet Another Traffic Classifier）是一个基于掩码自编码器的流量 Transformer 模型，使用多级流表示（MFR）进行加密流量分类。

**论文**: *Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation* (AAAI 2023)

## 模型架构

### 预训练阶段：MAE_YaTC (掩码自编码器)
```
输入: MFR 图像 (40×40 灰度)
  ↓
PatchEmbed (patch_size=2) → 400 个 patch
  ↓
随机掩码 (90%)
  ↓
Encoder (4 层 Transformer, dim=192, heads=16)
  ↓
Decoder (2 层 Transformer, dim=128)
  ↓
重建原始图像 (MSE Loss)
```

### 微调阶段：TraFormer_YaTC (分类器)
```
输入: MFR 图像 (40×40 灰度)
  ↓
PatchEmbed (patch_size=2)
  ↓
加载预训练 Encoder 权重
  ↓
Encoder + CLS token
  ↓
Classifier → num_classes
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
├── Model_YaTC_pytorch.py       # 独立模型定义
│
├── github/                     # 原始实现
│   ├── models_YaTC.py          # MAE_YaTC, TraFormer_YaTC
│   ├── pre-train.py            # 预训练脚本
│   ├── fine-tune.py            # 微调脚本
│   ├── data_process.py         # PCAP → MFR 转换
│   ├── engine.py               # 训练循环
│   └── util/                   # 工具函数
│       ├── lr_decay.py         # 逐层学习率衰减
│       ├── lr_sched.py         # 学习率调度
│       ├── pos_embed.py        # 位置嵌入
│       └── misc.py             # 杂项工具
│
├── ISCXVPN/                    # ISCXVPN 专用
│   ├── build_yatc_dataset.py   # 数据集构建
│   ├── train_yatc_simple.py    # 简化训练脚本
│   └── github/                 # ISCXVPN 专用原始代码
│
└── refactor/                   # 重构版本
    ├── models.py
    ├── data.py
    ├── engine.py
    └── train.py
```

## 快速开始

### 1. 数据准备 (PCAP → MFR)
```bash
cd github
# 编辑 data_process.py 中的路径
python data_process.py
# 输出: data/<dataset>_MFR/train/class/*.png
#       data/<dataset>_MFR/test/class/*.png
```

### 2. 预训练
```bash
python pre-train.py \
    --batch_size 128 \
    --blr 1e-3 \
    --steps 150000 \
    --mask_ratio 0.9
# 输出: output_dir/YaTC_pretrained_model.pth
```

### 3. 微调
```bash
python fine-tune.py \
    --blr 2e-3 \
    --epochs 200 \
    --data_path ./data/ISCXVPN2016_MFR \
    --nb_classes 7 \
    --finetune ./output_dir/YaTC_pretrained_model.pth
```

## 关键参数

### 预训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 128 | 批大小 |
| `--blr` | 1e-3 | 基础学习率 |
| `--steps` | 150000 | 训练步数 |
| `--mask_ratio` | 0.9 | 掩码比例 (90%) |
| `--warmup_epochs` | 40 | 预热 epochs |

### 微调参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--blr` | 2e-3 | 基础学习率 |
| `--epochs` | 200 | 训练轮数 |
| `--data_path` | - | MFR 数据集路径 |
| `--nb_classes` | - | 类别数量 |
| `--finetune` | - | 预训练模型路径 |
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
timm == 0.3.2  # 必须，版本敏感！
torch >= 1.9.0
numpy
PIL/Pillow
scikit-learn
```

**注意**: `timm` 版本必须为 `0.3.2`，代码中有版本检查：
```python
assert timm.__version__ == "0.3.2"  # models_YaTC.py
```

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

1. **timm 版本**: 必须使用 `timm==0.3.2`，新版本 API 不兼容
2. **预训练数据**: 预训练可以使用大规模无标签数据
3. **逐层衰减**: 微调时底层使用更小的学习率 (`layer_decay=0.65`)
4. **掩码比例**: 90% 掩码比例是论文推荐值，对流量数据效果最好
