# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

YaTC（Yet Another Traffic Classifier）是一个基于掩码自编码器的流量 Transformer 模型，使用多级流表示（MFR）进行加密流量分类。发表于 AAAI 2023。

## 参考论文

- 论文：*Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation*
- 文件：`Yet Another_Traffic Classifier A Masked Autoencoder Based Traffic Transformer.pdf`

## 主要依赖

```
python=3.8
torch=1.9.0
timm=0.3.2  # 版本敏感，代码中有版本检查
scikit-learn=0.24.2
numpy=1.19.5
scapy  # 用于 pcap 处理
PIL/Pillow
```

## 常用命令

### 预训练
```bash
cd github
python pre-train.py --batch_size 128 --blr 1e-3 --steps 150000 --mask_ratio 0.9
```

### 微调
```bash
cd github
python fine-tune.py --blr 2e-3 --epochs 200 --data_path ./data/ISCXVPN2016_MFR --nb_classes 7
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--data_path` | MFR 数据集路径 |
| `--nb_classes` | 类别数量 |
| `--finetune` | 预训练模型路径（默认：`./output_dir/YaTC_pretrained_model.pth`）|
| `--mask_ratio` | 掩码比例（默认 0.9）|

### 支持的数据集

| 数据集 | 类别数 |
|--------|--------|
| ISCXVPN2016_MFR | 7 |
| ISCXTor2016_MFR | 8 |
| USTC-TFC2016_MFR | 20 |
| CICIoT2022_MFR | 10 |

## 架构说明

### 两阶段训练

1. **预训练阶段**：掩码自编码器（`MAE_YaTC`）
   - 90% 掩码比例
   - 编码器：4 个 Transformer 块，192 维嵌入，16 个注意力头
   - 解码器：2 个 Transformer 块，128 维嵌入

2. **微调阶段**：流量 Transformer（`TraFormer_YaTC`）
   - 继承自 timm 的 VisionTransformer
   - 自定义 PatchEmbed（patch_size=2）
   - 逐层学习率衰减

### MFR 数据格式

将 pcap 流转换为 40x40 灰度图像：
- 每个流取 5 个数据包
- 每个数据包：160 字符头部 + 480 字符载荷 = 320 字节
- 总计：5 × 320 = 1600 字节 → 40×40 矩阵

## 目录结构

```
YaTC/
├── github/
│   ├── data/                    # 数据集目录
│   │   ├── CICIoT2022_MFR/
│   │   ├── ISCXTor2016_MFR/
│   │   ├── ISCXVPN2016_MFR/
│   │   └── USTC-TFC2016_MFR/
│   ├── output_dir/              # 模型输出目录
│   ├── util/                    # 工具函数
│   │   ├── lr_decay.py          # 学习率衰减
│   │   ├── lr_sched.py          # 学习率调度
│   │   ├── misc.py              # 杂项工具
│   │   └── pos_embed.py         # 位置嵌入
│   ├── data_process.py          # PCAP 转 MFR 图像
│   ├── engine.py                # 训练循环
│   ├── fine-tune.py             # 微调脚本
│   ├── models_YaTC.py           # 模型定义
│   └── pre-train.py             # 预训练脚本
└── refactor/                    # 重构代码（空）
```

## 核心文件说明

- `models_YaTC.py`：定义 `MAE_YaTC`（预训练）和 `TraFormer_YaTC`（微调）模型
- `engine.py`：包含 `pretrain_one_epoch`、`train_one_epoch`、`evaluate` 函数
- `data_process.py`：使用 `MFR_generator()` 将 pcap 文件转换为 PNG 图像
