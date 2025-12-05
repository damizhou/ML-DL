# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

本仓库包含用于加密网络流量分类的深度学习模型实现。主要项目是 **YaTC**（Yet Another Traffic Classifier），一个基于掩码自编码器的流量Transformer，发表于 AAAI 2023。此外还包含用于网站指纹攻击的 **DeepFingerprinting** 模型。

## 主要依赖

- Python 3.8
- PyTorch 1.9.0
- timm 0.3.2（YaTC 必需，版本敏感）
- scikit-learn 0.24.2
- scapy（用于 pcap 处理）
- PIL/Pillow
- numpy 1.19.5

## 常用命令

### YaTC 预训练
```bash
cd YaTC/github
python pre-train.py --batch_size 128 --blr 1e-3 --steps 150000 --mask_ratio 0.9
```

### YaTC 微调
```bash
cd YaTC/github
python fine-tune.py --blr 2e-3 --epochs 200 --data_path ./data/ISCXVPN2016_MFR --nb_classes 7
```

关键微调参数：
- `--data_path`：MFR 数据集路径（例如 `./data/ISCXVPN2016_MFR`）
- `--nb_classes`：类别数量（ISCXVPN2016: 7, ISCXTor2016: 8, USTC-TFC2016: 20, CICIoT2022: 10）
- `--finetune`：预训练模型检查点路径（默认：`./output_dir/YaTC_pretrained_model.pth`）

### 数据处理（PCAP 转 MFR）
```bash
cd YaTC/github
# 使用 data_process.py 中的 MFR_generator() 将 pcap 文件转换为 MFR 矩阵
```

## 架构说明

### YaTC (YaTC/github/)

采用多级流表示（MFR）的两阶段训练方法：

1. **预训练阶段**：掩码自编码器（`models_YaTC.py` 中的 `MAE_YaTC`）
   - 默认 90% 掩码比例
   - 编码器：4 个 Transformer 块，192 维嵌入，16 个注意力头
   - 解码器：2 个 Transformer 块，128 维嵌入
   - 输入：40x40 灰度图像（MFR 矩阵）

2. **微调阶段**：流量 Transformer（`models_YaTC.py` 中的 `TraFormer_YaTC`）
   - 继承自 timm 的 VisionTransformer
   - 自定义 `PatchEmbed` 用于 MFR 矩阵（patch_size=2）
   - 逐层学习率衰减

**数据格式**：MFR 将 pcap 流转换为 40x40 灰度图像：
- 每个流取 5 个数据包
- 每个数据包：160 个十六进制字符头部 + 480 个十六进制字符载荷 = 320 字节
- 总计：5 * 320 = 1600 字节 = 40x40 矩阵

**目录结构**：
- `data_process.py`：将 pcap 文件转换为 MFR PNG 图像
- `engine.py`：训练循环（`pretrain_one_epoch`、`train_one_epoch`、`evaluate`）
- `util/`：学习率调度器、位置嵌入、杂项工具

### DeepFingerprinting (DeepFingerprinting/)

用于流量指纹识别的一维 CNN（`Model_NoDef_pytorch.py` 中的 `DFNoDefNet`）：
- 4 个卷积块，包含 BatchNorm、MaxPool、Dropout
- 输入：5000 长度的方向序列
- 全连接层：512 -> 512 -> num_classes

**DatasetDealer/**：用于将 VPN/非VPN 流量数据集处理为训练格式的脚本。

## 数据集结构

YaTC 期望的格式：
```
data/<数据集名称>/
├── train/
│   ├── class1/
│   │   └── *.png
│   └── class2/
│       └── *.png
└── test/
    ├── class1/
    └── class2/
```

预训练模型位置：`./output_dir/pretrained-model.pth`
