# CLAUDE.md

本文件为 Claude Code 在此代码仓库中工作时提供指导。

## 项目概述

本仓库包含用于**加密网络流量分类**的多种深度学习模型实现，涵盖不同的技术路线：

| 项目 | 方法 | 输入特征 | 模型架构 |
|------|------|----------|----------|
| **FS-Net** | 流序列网络 | 数据包长度序列（±方向） | Bi-GRU + 自编码器 |
| **DeepFingerprinting** | 网站指纹攻击 | 方向序列（±1） | 1D CNN |
| **AppScanner** | 应用指纹识别 | 统计特征（54维） | MLP / Random Forest |
| **YaTC** | 流量Transformer | MFR图像（40×40） | ViT + MAE |

## 主要依赖

```
python >= 3.8
torch >= 1.9.0
numpy
scikit-learn
dpkt          # PCAP 解析
scapy         # PCAP 处理（可选）
timm == 0.3.2 # YaTC 专用，版本敏感
```

## 项目结构

```
ML&DL/
├── FS-Net/                 # 流序列网络（Bi-GRU）
│   ├── models.py           # FSNet, FSNetND 模型
│   ├── data.py             # 数据加载
│   ├── engine.py           # 训练/评估循环
│   ├── run_train.py        # 训练入口
│   └── iscx_processor.py   # ISCXVPN 数据处理
│
├── DeepFingerprinting/     # 网站指纹攻击（1D CNN）
│   ├── Model_NoDef_pytorch.py  # DFNoDefNet 模型
│   └── DatasetDealer/          # 数据处理脚本
│       ├── VPN/                # VPN 流量处理
│       ├── NOVPN/              # 非VPN 流量处理
│       └── ISCXVPN/            # ISCXVPN 数据集处理
│
├── AppScanner/             # 应用指纹识别（MLP/RF）
│   ├── models.py           # AppScannerNN, AppScannerRF
│   ├── data.py             # 特征提取
│   ├── config.py           # 配置（54维统计特征）
│   └── train.py            # 训练脚本
│
└── YaTC/                   # 流量Transformer（ViT + MAE）
    ├── github/             # 原始实现
    │   ├── models_YaTC.py  # MAE_YaTC, TraFormer_YaTC
    │   ├── pre-train.py    # 预训练脚本
    │   ├── fine-tune.py    # 微调脚本
    │   └── data_process.py # PCAP → MFR 转换
    └── refactor/           # 重构版本
```

## 快速开始

### FS-Net 训练
```bash
cd FS-Net
python run_train.py
# 配置在 run_train.py 顶部硬编码
# BATCH_SIZE=2048, EPOCHS=200, NUM_CLASSES=12
```

### DeepFingerprinting 训练
```bash
cd DeepFingerprinting/DatasetDealer/VPN
python train_df_simple.py
# 需要先准备 NPZ 数据文件和 labels.json
```

### AppScanner 训练
```bash
cd AppScanner
python train.py
# 使用 54 维统计特征，支持 NN/RF/SVM 分类器
```

### YaTC 预训练 + 微调
```bash
cd YaTC/github
# 预训练
python pre-train.py --batch_size 128 --blr 1e-3 --steps 150000 --mask_ratio 0.9
# 微调
python fine-tune.py --blr 2e-3 --epochs 200 --data_path ./data/ISCXVPN2016_MFR --nb_classes 7
```

## 数据集支持

| 数据集 | FS-Net | DF | AppScanner | YaTC |
|--------|--------|-----|------------|------|
| ISCXVPN2016 | ✓ (12类) | ✓ | ✓ | ✓ (7类) |
| ISCXTor2016 | - | ✓ | - | ✓ (8类) |
| USTC-TFC2016 | - | - | - | ✓ (20类) |
| CICIoT2022 | - | - | - | ✓ (10类) |

## 评估指标

所有项目统一使用以下指标：
- **Accuracy**: 分类准确率
- **Precision / Recall / F1**: 宏平均（macro）
- **TPR / FPR**: 每类真阳性率/假阳性率

## 注意事项

1. **GPU 优化**：确保启用 cuDNN（`torch.backends.cudnn.enabled = True`）
2. **Windows 兼容**：多进程 DataLoader 建议设置 `num_workers=0`
3. **版本敏感**：YaTC 必须使用 `timm==0.3.2`
