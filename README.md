# ML-DL: Encrypted Traffic Classification Models

本仓库包含多个用于加密网络流量分类的深度学习模型实现。

## 项目列表

| 模型 | 论文 | 会议/期刊 | 方法 |
|------|------|-----------|------|
| **YaTC** | Yet Another Traffic Classifier | AAAI 2023 | Masked Autoencoder + Transformer |
| **FS-Net** | Flow Sequence Network | INFOCOM 2019 | Bi-GRU + Reconstruction |
| **DeepFingerprinting** | Deep Fingerprinting | CCS 2018 | 1D CNN |
| **AppScanner** | Automatic Fingerprinting of Smartphone Apps | Euro S&P 2016 | Random Forest / Neural Network |

## 环境要求

```
Python >= 3.8
PyTorch >= 1.9.0
NumPy
scikit-learn
scapy (用于 PCAP 处理)
Pillow
```

## 快速开始

### YaTC

基于掩码自编码器的流量 Transformer，使用多级流表示（MFR）。

```bash
cd YaTC/refactor

# 预训练
python train.py pretrain --data_path ./data --mask_ratio 0.9 --steps 150000

# 微调
python train.py finetune --data_path ./data/ISCXVPN2016_MFR --num_classes 7 --epochs 200
```

详见 [YaTC/refactor/README.md](YaTC/refactor/README.md)

### FS-Net

基于双向 GRU 的流序列网络，包含重构机制。

```bash
cd FS-Net

# 训练
python run_train.py
```

关键参数（与论文一致）：
- 嵌入维度：128
- 隐藏维度：128
- GRU 层数：2
- 学习率：0.0005

详见 [FS-Net/README.md](FS-Net/README.md)

### DeepFingerprinting

用于网站指纹攻击的一维卷积神经网络。

```bash
cd DeepFingerprinting

# 使用 DatasetDealer 处理数据
python DatasetDealer/VPN/train_df_simple.py
```

### AppScanner

从加密流量中提取统计特征进行应用识别。

```bash
cd AppScanner

# 训练
python train.py --mode train --data_dir ./data/apps --num_classes 110
```

详见 [AppScanner/README.md](AppScanner/README.md)

## 数据集格式

### YaTC (MFR 格式)

将 PCAP 转换为 40x40 灰度图像：
```
data/
├── train/
│   ├── class1/*.png
│   └── class2/*.png
└── test/
    ├── class1/*.png
    └── class2/*.png
```

### FS-Net / DeepFingerprinting

按类别组织的 PCAP 或序列文件：
```
data/
├── train/
│   ├── class1/*.pcap (或 *.json)
│   └── class2/*.pcap
└── test/
    ├── class1/*.pcap
    └── class2/*.pcap
```

## 支持的数据集

| 数据集 | 类别数 | 用途 |
|--------|--------|------|
| ISCXVPN2016 | 7/12 | VPN 流量分类 |
| ISCXTor2016 | 8 | Tor 流量分类 |
| USTC-TFC2016 | 20 | 恶意流量检测 |
| CICIoT2022 | 10 | IoT 设备识别 |

## 目录结构

```
ML-DL/
├── YaTC/                    # Masked Autoencoder Traffic Transformer
│   ├── github/              # 原始实现 (Python 3.8 + timm 0.3.2)
│   └── refactor/            # 重构版本 (Python 3.12 + PyTorch 2.9)
├── FS-Net/                  # Flow Sequence Network
│   ├── models.py            # 模型定义
│   ├── data.py              # 数据处理
│   ├── engine.py            # 训练引擎
│   └── run_train.py         # 训练脚本
├── DeepFingerprinting/      # Website Fingerprinting CNN
│   ├── Model_NoDef_pytorch.py
│   └── DatasetDealer/       # 数据处理脚本
├── AppScanner/              # App Fingerprinting
│   ├── models.py
│   ├── features.py
│   └── train.py
└── README.md
```

## 引用

### YaTC
```bibtex
@inproceedings{zhao2023yatc,
  title={Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation},
  author={Zhao, Ruijie and Zhan, Mingwei and Deng, Xianwen and Wang, Yanhao and Wang, Yijun and Gui, Guan and Xue, Zhi},
  booktitle={AAAI},
  year={2023}
}
```

### FS-Net
```bibtex
@inproceedings{liu2019fsnet,
  title={FS-Net: A Flow Sequence Network For Encrypted Traffic Classification},
  author={Liu, Chang and He, Longtao and Xiong, Gang and Cao, Zigang and Li, Zhen},
  booktitle={IEEE INFOCOM},
  year={2019}
}
```

### DeepFingerprinting
```bibtex
@inproceedings{sirinam2018deep,
  title={Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning},
  author={Sirinam, Payap and Imani, Mohsen and Juarez, Marc and Wright, Matthew},
  booktitle={ACM CCS},
  year={2018}
}
```

### AppScanner
```bibtex
@inproceedings{taylor2016appscanner,
  title={AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic},
  author={Taylor, Vincent F and Spolaor, Riccardo and Conti, Mauro and Martinovic, Ivan},
  booktitle={IEEE Euro S&P},
  year={2016}
}
```

## 许可证

本项目仅供学术研究使用。
