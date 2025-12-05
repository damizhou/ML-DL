# YaTC: Yet Another Traffic Classifier

基于掩码自编码器的流量 Transformer 模型重构实现。

## 概述

YaTC 是一个用于加密网络流量分类的深度学习模型，采用多级流表示（MFR）和两阶段训练策略。本项目是对原始实现的重构，兼容 Python 3.12 和 PyTorch 2.9。

## 论文信息

- **标题**: Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation
- **会议**: AAAI 2023
- **作者**: Ruijie Zhao et al.

## 环境要求

- Python 3.12
- PyTorch 2.9.0
- NumPy
- Pillow
- scikit-learn (可选，用于评估)

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
refactor/
├── config.py              # 配置和超参数
├── models.py              # 模型定义 (MAE_YaTC, TraFormer_YaTC)
├── data.py                # 数据处理和 MFR 生成
├── engine.py              # 训练和评估引擎
├── train.py               # 主训练脚本
├── tests.py               # 单元测试
├── requirements.txt       # 依赖列表
├── analysis_report.md     # 论文分析报告
├── model_spec.md          # 模型规格说明
├── implementation_details.md  # 实现细节
└── README.md              # 本文件
```

## 使用方法

### 1. 数据准备

将 PCAP 文件转换为 MFR 格式：

```python
from data import MFRGenerator

generator = MFRGenerator()
packets = ["<hex_string_1>", "<hex_string_2>", ...]  # 5 个数据包
mfr = generator.flow_to_mfr(packets)
generator.save_mfr_as_png(mfr, "output.png")
```

数据集目录结构：

```
data/
├── train/
│   ├── class1/
│   │   └── *.png
│   └── class2/
│       └── *.png
└── test/
    ├── class1/
    └── class2/
```

### 2. 预训练

```bash
python train.py pretrain \
    --data_path ./data \
    --batch_size 128 \
    --lr 1e-3 \
    --steps 150000 \
    --mask_ratio 0.9 \
    --output_dir ./output_dir
```

### 3. 微调

```bash
python train.py finetune \
    --data_path ./data/ISCXVPN2016_MFR \
    --pretrained ./output_dir/pretrained.pth \
    --num_classes 7 \
    --batch_size 128 \
    --lr 2e-3 \
    --epochs 200 \
    --output_dir ./output_dir
```

### 4. 评估

```bash
python train.py eval \
    --data_path ./data/ISCXVPN2016_MFR \
    --checkpoint ./output_dir/best.pth \
    --num_classes 7
```

## 模型架构

### MAE 预训练

| 组件 | 参数 |
|------|------|
| 输入大小 | 40×40 |
| Patch 大小 | 2×2 |
| 总 Patch 数 | 400 |
| 编码器维度 | 192 |
| 编码器层数 | 4 |
| 注意力头数 | 16 |
| 解码器维度 | 128 |
| 解码器层数 | 2 |
| 掩码比例 | 90% |

### TraFormer 微调

| 组件 | 参数 |
|------|------|
| 编码器维度 | 192 |
| 编码器层数 | 4 |
| 注意力头数 | 16 |
| MLP 比率 | 4 |
| Drop Path | 0.1 |

## 支持的数据集

| 数据集 | 类别数 |
|--------|--------|
| ISCXVPN2016 | 7 |
| ISCXTor2016 | 8 |
| USTC-TFC2016 | 20 |
| CICIoT2022 | 10 |

## 运行测试

```bash
python -m pytest tests.py -v
```

或：

```bash
python tests.py
```

## API 参考

### 创建模型

```python
from models import mae_yatc, traformer_yatc

# 创建 MAE 预训练模型
mae_model = mae_yatc()

# 创建 TraFormer 微调模型
traformer_model = traformer_yatc(num_classes=7)
```

### 加载预训练权重

```python
from engine import load_pretrained_weights

model = traformer_yatc(num_classes=7)
model = load_pretrained_weights(model, "pretrained.pth")
```

### 自定义配置

```python
from config import MAEConfig, TraFormerConfig

mae_config = MAEConfig(
    img_size=40,
    patch_size=2,
    mask_ratio=0.9
)
```

## 与原实现的差异

1. **Python 版本**: 3.8 → 3.12
2. **PyTorch 版本**: 1.9.0 → 2.9.0
3. **timm 依赖**: 移除对 timm 0.3.2 的依赖，完全使用 PyTorch 原生实现
4. **代码结构**: 模块化重构，提高可读性和可维护性
5. **类型注解**: 添加完整的类型注解

## 许可证

本项目仅供学术研究使用。

## 引用

```bibtex
@inproceedings{zhao2023yatc,
  title={Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation},
  author={Zhao, Ruijie and Zhan, Mingwei and Deng, Xianwen and Wang, Yanhao and Wang, Yijun and Gui, Guan and Xue, Zhi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
