# FS-Net: Flow Sequence Network

基于循环神经网络的加密流量分类模型实现。

## 论文信息

- **标题**: FS-Net: A Flow Sequence Network For Encrypted Traffic Classification
- **作者**: Chang Liu, Longtao He, Gang Xiong, Zigang Cao, Zhen Li
- **单位**: Institute of Information Engineering, Chinese Academy of Sciences

## 模型架构

FS-Net 是一个端到端的加密流量分类模型，包含 7 层：

1. **Embedding Layer** - 将包长度转换为密集向量
2. **Encoder Layer** - 多层双向 GRU 学习流表示
3. **Decoder Layer** - 多层双向 GRU 重构输入
4. **Reconstruction Layer** - Softmax 预测原始序列
5. **Dense Layer** - 组合编码器和解码器特征
6. **Classification Layer** - Softmax 分类器

## 关键参数（与论文一致）

| 参数 | 值 |
|------|-----|
| 嵌入维度 | 128 |
| GRU 隐藏维度 | 128 |
| GRU 层数 | 2 |
| Dropout | 0.3 |
| 学习率 | 0.0005 |
| 优化器 | Adam |
| 损失权重 α | 1.0 |

## 安装

```bash
pip install torch numpy scapy tqdm
```

## 使用方法

### 1. 准备数据

将 PCAP 文件按以下结构组织：

```
data/
├── train/
│   ├── class1/*.pcap
│   └── class2/*.pcap
└── test/
    ├── class1/*.pcap
    └── class2/*.pcap
```

### 2. 转换 PCAP 为序列（可选）

```bash
python train.py convert --input ./pcap_data --output ./seq_data
```

### 3. 训练模型

```bash
python train.py train \
    --data_path ./data \
    --num_classes 18 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.0005
```

### 4. 评估模型

```bash
python train.py eval \
    --data_path ./data \
    --checkpoint ./checkpoints/best.pth \
    --num_classes 18
```

## 模型变体

### FS-Net (完整版)

使用编码器和解码器，包含重构机制：

```bash
python train.py train --data_path ./data --num_classes 18
```

### FS-Net-ND (无解码器)

简化版本，仅使用编码器特征：

```bash
python train.py train --data_path ./data --num_classes 18 --no_decoder
```

## API 使用

```python
import torch
from models import create_fsnet

# 创建模型
model = create_fsnet(num_classes=18)

# 推理
x = torch.randint(1, 1500, (1, 50))  # 包长度序列
lengths = torch.tensor([50])
class_logits, recon_logits = model(x, lengths)

# 预测
pred = class_logits.argmax(dim=1)
```

## 评估指标

与论文一致，使用以下指标：

- **TPR** (True Positive Rate): 真正率
- **FPR** (False Positive Rate): 假正率
- **FTF**: 论文定义的综合指标

```
FTF = Σ(wi * TPRi / (1 + FPRi))
```

## 文件结构

```
FS-Net/
├── config.py       # 配置和超参数
├── models.py       # 模型定义
├── data.py         # 数据处理
├── engine.py       # 训练引擎
├── train.py        # 主脚本
├── tests.py        # 单元测试
└── README.md       # 说明文档
```

## 运行测试

```bash
python -m pytest tests.py -v
```

## 论文结果

在 18 类应用数据集上：

| 指标 | 值 |
|------|-----|
| TPR_AVE | 99.14% |
| FPR_AVE | 0.05% |
| FTF | 0.9906 |

## 引用

```bibtex
@inproceedings{liu2019fsnet,
  title={FS-Net: A Flow Sequence Network For Encrypted Traffic Classification},
  author={Liu, Chang and He, Longtao and Xiong, Gang and Cao, Zigang and Li, Zhen},
  booktitle={IEEE INFOCOM},
  year={2019}
}
```
