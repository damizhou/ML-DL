# 代码重构计划

本文档记录 ML&DL 项目的代码重复问题分析和重构方案。

## 一、当前问题

### 1.1 代码重复统计

| 类别 | 重复行数 | 文件数 | 可复用度 |
|------|---------|--------|---------|
| PCAP 解析层 | ~2,000行 | 15个 processor | 100% |
| 特征提取层 | ~400-500行 | 6个 | 90-95% |
| 数据操作层 | ~500-600行 | 21个 | 100% |
| 训练框架 | ~500-600行 | 9个 | 85% |
| **总计** | **~3,500-4,000行** | - | 占总代码 23-27% |

### 1.2 重复文件分布

#### Processor 文件（最严重，15个文件）

每个模型都有 5 个几乎相同的数据集处理器：

```
FS-Net/
├── iscx_vpn_processor.py
├── iscx_tor_processor.py
├── ustc_processor.py
├── cross_platform_processor.py
└── cic_iot_2022_processor.py

AppScanner/
├── iscx_vpn_processor.py
├── iscx_tor_processor.py
├── ustc_processor.py
├── cross_platform_processor.py
└── cic_iot_2022_processor.py

YaTC/
├── iscx_vpn_processor.py
├── iscx_tor_processor.py
├── ustc_processor.py
├── cross_platform_processor.py
└── cic_iot_2022_processor.py
```

**重复的核心函数**：

| 函数 | 行数 | 重复次数 | 总重复行数 |
|------|------|---------|-----------|
| `parse_l3()` / `extract_ip_from_buf()` | 60 | 15 | 900 |
| `extract_flows_from_pcap()` | 150 | 15 | 2,250 |
| `load_label_map()` / `load_vocab()` | 15 | 21 | 315 |
| 标签映射保存逻辑 | 20 | 15 | 300 |

#### 训练脚本（9+个文件）

```
FS-Net/
├── run_train.py
└── novpn_processor.py

AppScanner/
├── train.py
├── train_with_dataset.py
└── novpn_processor.py

YaTC/
├── train_vpn.py
└── refactor/train.py

DeepFingerprinting/
├── run_train.py
├── DatasetDealer/VPN/train_df_simple.py
├── DatasetDealer/ISCXVPN/train_df_simple.py
└── ... (多个版本)
```

#### 已有的统一方案

项目根目录已有两个统一处理器（证明四模型共享是可行的）：

```
ML&DL/
├── unified_vpn_processor.py      # VPN 数据统一处理
└── unified_novpn_processor.py    # NOVPN 数据统一处理
```

---

## 二、重构目标

1. **消除重复代码**：减少 ~3,500 行重复代码
2. **统一数据处理**：15 个 processor → 1 个统一脚本
3. **简化训练流程**：每个模型保留 1 个标准训练入口
4. **提高可维护性**：Bug 修复和功能更新只需改一处

---

## 三、目标目录结构

```
ML&DL/
├── core/                              # 共享核心库
│   ├── __init__.py
│   ├── pcap_parser.py                 # PCAP 解析 (~300行)
│   ├── flow_extractor.py              # 流提取与聚合 (~200行)
│   ├── feature_extractors.py          # 四种特征提取器 (~250行)
│   ├── metadata.py                    # 标签加载/保存 (~80行)
│   ├── dataset.py                     # 基础 Dataset 类 (~150行)
│   └── trainer.py                     # 训练基类 (~200行)
│
├── scripts/                           # 统一脚本
│   ├── process_dataset.py             # 统一数据处理（替代15个processor）
│   └── generate_labels_json.py        # 生成 labels.json
│
├── FS-Net/
│   ├── models.py                      # 模型定义
│   ├── config.py                      # 配置
│   ├── data.py                        # 数据加载（调用 core/）
│   ├── engine.py                      # 训练引擎（调用 core/）
│   └── run_train.py                   # 训练入口
│
├── AppScanner/
│   ├── models.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   └── run_train.py
│
├── DeepFingerprinting/
│   ├── Model_NoDef_pytorch.py
│   └── run_train.py
│
├── YaTC/
│   ├── refactor/
│   │   ├── models.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── engine.py
│   │   └── train.py
│   └── Model_YaTC_pytorch.py
│
├── unified_vpn_processor.py           # 保留：VPN 统一处理
├── unified_novpn_processor.py         # 保留：NOVPN 统一处理
│
└── data/                              # 统一数据目录（可选）
    ├── iscxvpn/
    ├── iscxtor/
    ├── ustc/
    ├── cic_iot_2022/
    └── cross_platform/
```

---

## 四、核心模块设计

### 4.1 `core/pcap_parser.py`

提取所有 processor 共享的 PCAP 解析逻辑：

```python
"""PCAP 解析模块 - 统一的数据包解析接口"""

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
import dpkt

@dataclass
class PacketInfo:
    """解析后的数据包信息"""
    timestamp: float
    length: int
    direction: int        # +1 出站, -1 入站
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str         # "TCP" / "UDP"
    ip_header: bytes
    l4_header: bytes
    payload: bytes


def parse_l3(buf: bytes) -> Optional[dpkt.ip.IP]:
    """解析 L3 层（支持 Ethernet/SLL/Raw IP/NULL）"""
    ...

def iter_packets(pcap_path: str) -> Iterator[Tuple[float, bytes]]:
    """迭代 PCAP 文件中的数据包"""
    ...

def parse_packet(ts: float, buf: bytes, client_ip: str = None) -> Optional[PacketInfo]:
    """解析单个数据包"""
    ...
```

### 4.2 `core/flow_extractor.py`

统一的流提取与聚合逻辑：

```python
"""流提取模块 - 将数据包聚合为流"""

from dataclasses import dataclass, field
from typing import Dict, List
from .pcap_parser import PacketInfo

@dataclass
class FlowConfig:
    """流提取配置"""
    timeout: float = 60.0           # 流超时（秒）
    min_packets: int = 1            # 最小数据包数
    max_packets: int = 10000        # 最大数据包数
    exclude_ports: set = field(default_factory=lambda: {5353})


@dataclass
class Flow:
    """网络流"""
    flow_key: tuple
    packets: List[PacketInfo]

    @property
    def packet_count(self) -> int: ...

    def get_lengths(self) -> List[int]: ...

    def get_directions(self) -> List[int]: ...


def extract_flows(pcap_path: str, config: FlowConfig = None) -> Dict[tuple, Flow]:
    """从 PCAP 提取流"""
    ...

def split_flow_by_timeout(flow: Flow, timeout: float) -> List[Flow]:
    """按超时分割流"""
    ...
```

### 4.3 `core/feature_extractors.py`

四种模型的特征提取器：

```python
"""特征提取模块 - 为四种模型提取特征"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .flow_extractor import Flow

@dataclass
class FeatureConfig:
    """特征提取配置"""
    # FS-Net
    fsnet_max_seq_len: int = 100
    fsnet_max_pkt_len: int = 1500

    # DeepFingerprinting
    df_max_seq_len: int = 5000

    # AppScanner
    appscanner_percentiles: List[int] = field(
        default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90]
    )

    # YaTC
    yatc_num_packets: int = 5
    yatc_header_len: int = 40
    yatc_payload_len: int = 280
    yatc_image_size: int = 40


def extract_fsnet_features(flow: Flow, config: FeatureConfig) -> Optional[np.ndarray]:
    """提取 FS-Net 特征：数据包长度序列（带方向）"""
    ...

def extract_df_features(flow: Flow, config: FeatureConfig) -> Optional[np.ndarray]:
    """提取 DeepFingerprinting 特征：方向序列"""
    ...

def extract_appscanner_features(flow: Flow, config: FeatureConfig) -> Optional[np.ndarray]:
    """提取 AppScanner 特征：54维统计特征"""
    ...

def extract_yatc_features(flow: Flow, config: FeatureConfig) -> Optional[np.ndarray]:
    """提取 YaTC 特征：MFR 图像"""
    ...


# 统计特征计算（AppScanner 使用）
def compute_statistics(lengths: np.ndarray, percentiles: List[int]) -> np.ndarray:
    """计算 18 维统计特征"""
    ...
```

### 4.4 `core/metadata.py`

标签加载与保存：

```python
"""元数据模块 - 标签映射加载与保存"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

def load_label_map(csv_path: str) -> List[Tuple[str, int]]:
    """从 CSV 加载标签映射"""
    ...

def load_vocab(csv_path: str) -> Dict[int, str]:
    """从 CSV 加载词汇表"""
    ...

def save_labels_json(
    output_path: Path,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    extra_meta: Dict = None
):
    """保存 labels.json"""
    ...

def load_labels_json(json_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载 labels.json"""
    ...
```

### 4.5 `scripts/process_dataset.py`

统一的数据处理脚本（替代 15 个 processor）：

```python
"""统一数据处理脚本

Usage:
    # 处理单个数据集，为所有模型生成数据
    python process_dataset.py --dataset iscxvpn --root /path/to/data

    # 只为特定模型生成数据
    python process_dataset.py --dataset iscxvpn --models fsnet appscanner

    # 处理所有数据集
    python process_dataset.py --all --root /path/to/datasets

支持的数据集：
    - iscxvpn
    - iscxtor
    - ustc
    - cic_iot_2022
    - cross_platform
    - novpn (自定义目录结构)
"""

import argparse
from pathlib import Path
from core.pcap_parser import iter_packets, parse_packet
from core.flow_extractor import extract_flows, FlowConfig
from core.feature_extractors import (
    extract_fsnet_features,
    extract_df_features,
    extract_appscanner_features,
    extract_yatc_features,
    FeatureConfig
)
from core.metadata import save_labels_json


DATASET_CONFIGS = {
    'iscxvpn': {
        'label_csv': 'artifacts/iscxvpn/label_map.csv',
        'vocab_csv': 'artifacts/iscxvpn/label_vocab.csv',
    },
    'iscxtor': { ... },
    'ustc': { ... },
    'cic_iot_2022': { ... },
    'cross_platform': { ... },
}


def process_dataset(
    dataset: str,
    root: Path,
    output_root: Path,
    models: List[str] = None
):
    """处理数据集，为指定模型生成数据"""
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASET_CONFIGS.keys())
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--models', nargs='+',
                        choices=['fsnet', 'df', 'appscanner', 'yatc'],
                        default=['fsnet', 'df', 'appscanner', 'yatc'])
    parser.add_argument('--all', action='store_true')
    ...
```

---

## 五、实施计划

### 阶段一：创建核心库（1-2天）

**目标**：提取共享代码，不影响现有功能

1. 创建 `core/` 目录
2. 实现 `pcap_parser.py`（从现有 processor 提取）
3. 实现 `flow_extractor.py`
4. 实现 `feature_extractors.py`
5. 实现 `metadata.py`
6. 添加单元测试

**验证**：
- 核心模块可独立运行
- 与现有 processor 输出一致

### 阶段二：创建统一处理脚本（1天）

**目标**：用单一脚本替代 15 个 processor

1. 创建 `scripts/process_dataset.py`
2. 支持所有 5 个数据集
3. 支持选择性生成（指定模型）
4. 添加进度显示和断点续传

**验证**：
- 输出与原 processor 一致
- 性能不下降

### 阶段三：简化模型目录（1-2天）

**目标**：各模型目录只保留必要文件

1. 修改各模型的 `data.py`，调用 `core/` 模块
2. 删除各模型目录下的 processor 文件
3. 统一训练脚本命名为 `run_train.py`
4. 更新各模型的 CLAUDE.md

**删除文件清单**：
```
FS-Net/
├── iscx_vpn_processor.py      # 删除
├── iscx_tor_processor.py      # 删除
├── ustc_processor.py          # 删除
├── cross_platform_processor.py # 删除
└── cic_iot_2022_processor.py  # 删除

AppScanner/
├── iscx_vpn_processor.py      # 删除
├── iscx_tor_processor.py      # 删除
├── ustc_processor.py          # 删除
├── cross_platform_processor.py # 删除
└── cic_iot_2022_processor.py  # 删除

YaTC/
├── iscx_vpn_processor.py      # 删除
├── iscx_tor_processor.py      # 删除
├── ustc_processor.py          # 删除
├── cross_platform_processor.py # 删除
└── cic_iot_2022_processor.py  # 删除
```

### 阶段四：整理 DeepFingerprinting（1天）

**目标**：简化 DatasetDealer 目录

1. 保留 `run_train.py`（已创建）
2. 整理 `DatasetDealer/` 下的重复脚本
3. 统一数据格式

### 阶段五：文档更新（0.5天）

1. 更新根目录 CLAUDE.md
2. 更新各模型 CLAUDE.md
3. 添加 core/ 模块文档

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 重构导致功能回归 | 高 | 保留原文件备份，添加单元测试 |
| 性能下降 | 中 | 基准测试对比 |
| 学习成本 | 低 | 保持接口简单，文档完善 |

---

## 七、预期收益

| 指标 | 当前 | 重构后 | 改进 |
|------|------|--------|------|
| Processor 文件数 | 15 | 1 | -93% |
| 重复代码行数 | ~3,500 | ~200 | -94% |
| Bug 修复点 | 15+ | 1 | -93% |
| 新数据集支持工作量 | 4个文件 | 1个配置 | -75% |

---

## 八、保留的统一处理器

以下文件作为参考实现保留，后续可考虑整合到 `scripts/process_dataset.py`：

```
ML&DL/
├── unified_vpn_processor.py      # VPN 数据统一处理（已验证可用）
└── unified_novpn_processor.py    # NOVPN 数据统一处理
```

这两个文件证明了四模型共享 PCAP 解析的可行性，是本次重构的基础。
