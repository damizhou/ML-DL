# CLAUDE.md - AppScanner

本文件为 Claude Code 在 AppScanner 项目中工作时提供指导。

## 项目概述

AppScanner 是一种基于机器学习的智能手机应用指纹识别方法，从加密网络流量的统计特征中识别应用程序。

**论文**: *AppScanner: Automatic Fingerprinting of Smartphone Apps From Encrypted Network Traffic* (Euro S&P 2015)

## 模型架构

### AppScannerNN (神经网络版本)
```
输入: 54 维统计特征
  ↓
FC(256) → BN → ReLU → Dropout(0.3)
  ↓
FC(128) → BN → ReLU → Dropout(0.3)
  ↓
FC(64) → BN → ReLU → Dropout(0.3)
  ↓
Classifier → num_classes
```

### AppScannerRF (随机森林，论文原方法)
- 100 棵决策树
- 论文最佳结果：99.6% 准确率 (阈值 0.9)

### AppScannerDeep (深度变体)
- 4 层残差块
- Pre-activation 结构

## 统计特征 (54 维)

每个方向（入站/出站/双向）提取 18 个特征：

| 类型 | 特征 | 数量 |
|------|------|------|
| 计数 | 数据包数量 | 1 |
| 统计 | Min, Max, Mean, Std, Var | 5 |
| 分布 | Skewness, Kurtosis | 2 |
| 离散 | MAD (中位数绝对偏差) | 1 |
| 分位数 | 10%, 20%, ..., 90% | 9 |

总计: 18 × 3 = 54 维 (可精简至 40 维)

## 文件结构

```
AppScanner/
├── models.py              # AppScannerNN, AppScannerRF, AppScannerSVM
├── config.py              # 配置类 (AppScannerConfig)
├── data.py                # 特征提取与数据加载
├── engine.py              # 训练/评估循环
├── train.py               # 训练入口
└── iscxvpn_processor.py   # ISCXVPN 数据处理
```

## 快速开始

### 1. 训练神经网络
```bash
python train.py --model nn --epochs 100
```

### 2. 训练随机森林
```bash
python train.py --model rf --n_estimators 100
```

### 关键配置 (config.py)
```python
# 特征提取参数
burst_threshold = 1.0      # 突发阈值 (秒)
min_flow_length = 7        # 最小数据包数
max_flow_length = 260      # 最大数据包数

# 模型参数
input_dim = 54             # 特征维度 (或 40)
hidden_dims = [256, 128, 64]
dropout = 0.3
num_classes = 110          # 默认应用数

# 训练参数
batch_size = 128
learning_rate = 0.001
epochs = 100
prediction_threshold = 0.9 # 高置信度阈值
```

## 分类方法 (论文 Section IV)

| 方法 | 描述 | 准确率 |
|------|------|--------|
| Approach 1 | 每应用二分类器 | - |
| Approach 2 | 每应用单类分类器 | - |
| Approach 3 | 多类分类器（分离训练/测试） | - |
| **Approach 4** | **单一大型随机森林** | **99.6%** |
| Approach 5 | 二分类 + 分离训练/测试 | - |
| Approach 6 | 多类 + 全部数据 | - |

## 置信度预测

论文核心创新：仅对高置信度预测做出判断

```python
preds, confidences, is_confident = model.predict_with_confidence(x, threshold=0.9)
# 仅使用 is_confident=True 的预测
```

## 模型对比

| 模型 | 优点 | 缺点 |
|------|------|------|
| NN | 端到端训练，特征交互 | 需要更多数据 |
| RF | 稳健，可解释，快速 | 特征工程依赖 |
| SVM | 小样本表现好 | 大规模慢 |
| Ensemble | 综合优势 | 复杂度高 |

## 使用示例

```python
from models import build_model

# 神经网络
model = build_model('nn', input_dim=54, num_classes=110)

# 随机森林
rf = build_model('rf', n_estimators=100)
rf.fit(X_train, y_train)

# 集成模型
ensemble = AppScannerEnsemble(nn_model, rf_model, nn_weight=0.5)
```

## 注意事项

1. **特征标准化**: 建议对输入特征做标准化 (StandardScaler)
2. **类别不平衡**: 可使用类权重或过采样
3. **特征选择**: 可从 54 维精简至 40 维重要特征
4. **阈值调优**: prediction_threshold 影响精度/召回权衡
