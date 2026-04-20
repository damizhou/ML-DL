# AppScanner

PyTorch implementation of AppScanner from the paper:

**"AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic"**
(Euro S&P 2015)

## Overview

AppScanner identifies smartphone applications from encrypted network traffic using statistical features extracted from packet lengths. This implementation provides both:

1. **Neural Network classifier** (PyTorch) - Deep learning approach
2. **Random Forest classifier** (scikit-learn) - Original paper approach

## Key Features

- 54 statistical features extracted from packet lengths
- Features computed for 3 directions: incoming, outgoing, bidirectional
- Support for PCAP file processing
- Confidence-based prediction thresholding (90% threshold for high accuracy)

## Paper Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Burst threshold | 1.0s | Time threshold for burst detection |
| Min flow length | 7 packets | Minimum packets per flow |
| Max flow length | 260 packets | Maximum packets per flow |
| Prediction threshold | 0.9 | Confidence threshold for classification |
| Features per direction | 18 | Statistical features extracted |
| Total features | 54 | 18 features × 3 directions |

## Statistical Features (per direction)

1. Packet count
2. Minimum packet length
3. Maximum packet length
4. Mean packet length
5. Standard deviation
6. Variance
7. Skewness
8. Kurtosis
9. Median Absolute Deviation (MAD)
10-18. Percentiles (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training on PCAP Directory

```bash
python train.py --mode train \
    --data_dir ./data/apps \
    --num_classes 110 \
    --epochs 100
```

### Training on Pre-extracted Features (CSV)

```bash
python train.py --mode train \
    --csv_path ./data/features.csv \
    --epochs 100
```

### Extract Features from PCAP

```bash
python train.py --mode extract \
    --data_dir ./data/apps \
    --output ./data/features.pkl
```

### Evaluate Trained Model

```bash
python train.py --mode eval \
    --checkpoint ./output/best_model.pth \
    --data_dir ./data/apps
```

### Compare NN vs Random Forest

```bash
python train.py --mode compare \
    --data_dir ./data/apps
```

### Concurrent Training on Multiple Feature Datasets

```bash
python train_multi_datasets.py
```

This script launches multiple `train_with_dataset.py` subprocesses concurrently and
expects pre-extracted `.pkl` feature files under the standard AppScanner layout such as:

```text
data/vpn/vpn_appscanner.pkl
data/novpn/novpn_appscanner.pkl
data/novpn_top10/novpn_top10_appscanner.pkl
data/vpn_top10/vpn_top10_appscanner.pkl
...
```

Wrapper logs and summaries are written to `output_multi/`, while per-dataset model
artifacts still go to `output/<dataset_name>/`. The summary records Train/Val/Test
Accuracy and Macro-F1 for Random Forest runs. Per-dataset wrapper logs are named
`<dataset_name>_subprocess_output.log`.

## Data Format

### PCAP Directory Structure

```
data/
├── app1/
│   ├── trace1.pcap
│   └── trace2.pcap
├── app2/
│   ├── trace1.pcap
│   └── trace2.pcap
└── ...
```

### CSV Format

```csv
feature_0,feature_1,...,feature_53,label
1.23,4.56,...,7.89,app_name
```

## Model Architectures

### AppScannerNN (Neural Network)

```
Input (54) → Linear(256) → BN → ReLU → Dropout
          → Linear(128) → BN → ReLU → Dropout
          → Linear(64)  → BN → ReLU → Dropout
          → Linear(num_classes)
```

### AppScannerDeep (Deep Residual Network)

```
Input (54) → Linear(256) → BN → ReLU
          → ResBlock × 4
          → Linear(128) → ReLU → Dropout
          → Linear(num_classes)
```

### AppScannerRF (Random Forest)

- 100 trees (n_estimators)
- Scikit-learn implementation
- Feature importance available
- Disk-saved tree evaluation defaults to `auto`: exact tree-first soft voting is used only
  when the full probability buffer fits the configured cap; large sample/class splits fall
  back to batch-first soft voting to avoid full-matrix OOM.
- Default RF runtime settings are tuned for a 512GB RAM / 112-core CPU host and 100
  trees: fit 50 trees concurrently, evaluate 100 trees concurrently, use 100
  tree-first workers, use a 512MiB per-batch probability buffer, and cap RF
  evaluation probability memory at 360GiB. The defaults choose divisors of the tree
  count to avoid a low-utilization tail batch.

## Results (Paper Reference)

| Approach | Description | Accuracy |
|----------|-------------|----------|
| Approach 4 | Single Large RF | 99.6% |
| Approach 1 | Per-app Binary | 99.1% |
| Approach 3 | Multi-class | 98.8% |

*With 90% confidence threshold*

## Configuration

Key parameters in `config.py`:

```python
# Feature extraction
burst_threshold = 1.0      # seconds
min_flow_length = 7        # packets
max_flow_length = 260      # packets

# Classification
prediction_threshold = 0.9  # confidence threshold

# Neural Network
hidden_dims = [256, 128, 64]
dropout = 0.3
learning_rate = 0.001

# Random Forest
n_estimators = 100
```

## Running Tests

```bash
python -m pytest tests.py -v
# or
python tests.py
```

## Citation

```bibtex
@inproceedings{taylor2016appscanner,
  title={Appscanner: Automatic fingerprinting of smartphone apps from encrypted network traffic},
  author={Taylor, Vincent F and Spolaor, Riccardo and Conti, Mauro and Martinovic, Ivan},
  booktitle={2016 IEEE European Symposium on Security and Privacy (EuroS\&P)},
  pages={439--454},
  year={2016},
  organization={IEEE}
}
```

## Requirements

- Python 3.12
- PyTorch 2.9
- scikit-learn (for Random Forest)
- scapy (for PCAP processing)
- scipy
- numpy
