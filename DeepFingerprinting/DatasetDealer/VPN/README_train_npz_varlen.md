# Training Script for DFNoDefNet with Variable-Length NPZ Flows

This script provides a memory-efficient, GPU-optimized training pipeline for the DFNoDefNet model using variable-length flow sequences stored in NPZ files.

## Features

### Memory Efficiency
- **Lazy Loading**: NPZ files are loaded on-demand using PyTorch's IterableDataset
- **LRU Cache**: Configurable cache for recently accessed NPZ files
- **No Preloading**: Avoids loading the entire ~1.1GB dataset into memory
- **Streaming**: Processes samples file-by-file to minimize RAM usage

### GPU Optimization
- **Mixed Precision Training**: Supports automatic mixed precision (AMP) for faster training
- **Pinned Memory**: Faster CPU→GPU transfers when enabled
- **Non-blocking Transfers**: Asynchronous data transfers to GPU
- **Gradient Accumulation**: Support for larger effective batch sizes

### Data Handling
- **Variable-Length Sequences**: Automatically pads/truncates flows to fixed length (default: 5000)
- **Label Mapping**: Loads label-to-ID mapping from JSON file
- **Stratified Splits**: File-level train/val/test split
- **Multi-worker Loading**: Parallel data loading with configurable workers

### Training Features
- **Checkpointing**: Saves both best and last model checkpoints
- **Resume Training**: Can resume from saved checkpoints
- **Progress Logging**: Real-time training and validation metrics
- **Robust Error Handling**: Gracefully handles corrupted NPZ files

## Requirements

```bash
pip install torch numpy
```

## Usage

### Basic Usage

```bash
python train_npz_varlen.py \
  --npz_root /path/to/npz/directory \
  --labels_json /path/to/labels.json \
  --output_dir ./runs \
  --epochs 20 \
  --batch_size 128
```

### Full Example

```bash
python train_npz_varlen.py \
  --npz_root /home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz \
  --labels_json /home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json \
  --output_dir ./runs/vpn_experiment \
  --max_len 5000 \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --grad_accum_steps 2 \
  --num_workers 4 \
  --cache_size 10 \
  --use_amp \
  --seed 42
```

### GPU Training with Mixed Precision

```bash
python train_npz_varlen.py \
  --npz_root /path/to/npz \
  --labels_json /path/to/labels.json \
  --use_amp \
  --batch_size 256 \
  --grad_accum_steps 2
```

### Resume Training

```bash
python train_npz_varlen.py \
  --npz_root /path/to/npz \
  --labels_json /path/to/labels.json \
  --resume ./runs/last.pt \
  --epochs 50
```

## Command-Line Arguments

### Data Arguments
- `--npz_root`: Root directory containing NPZ files (default: `/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/npz`)
- `--labels_json`: Path to labels.json file (default: `/home/pcz/DL/ML_DL/DeepFingerprinting/DatasetDealer/VPN/labels.json`)
- `--output_dir`: Directory to save checkpoints and logs (default: `./runs`)

### Model Arguments
- `--max_len`: Maximum sequence length for padding/truncation (default: `5000`)
  - **Note**: DFNoDefNet is designed for sequences of length 5000. Using other values may cause dimension mismatches.

### Training Arguments
- `--epochs`: Number of training epochs (default: `20`)
- `--batch_size`: Batch size (default: `128`)
- `--lr`: Learning rate (default: `1e-3`)
- `--grad_accum_steps`: Gradient accumulation steps (default: `1`)
- `--num_workers`: Number of data loading workers (default: `4`)
- `--cache_size`: Number of NPZ files to cache in memory (default: `10`)

### Optimization Arguments
- `--use_amp`: Enable automatic mixed precision training (flag)
- `--no_pin_memory`: Disable pinned memory for data loading (flag)

### Split Arguments
- `--train_ratio`: Ratio of files for training (default: `0.8`)
- `--val_ratio`: Ratio of files for validation (default: `0.1`)

### Misc Arguments
- `--seed`: Random seed for reproducibility (default: `0`)
- `--resume`: Path to checkpoint to resume from (optional)

## Data Format

### NPZ File Structure

Each NPZ file should contain two arrays:

```python
{
  'flows': np.array([...], dtype=object),   # Variable-length int8 arrays
  'labels': np.array([...], dtype=object)   # String labels
}
```

**Example:**
```python
flows[0] = array([-1, 1, -1, 1], dtype=int8)       # Length: 4
flows[1] = array([-1, 1, -1, -1, 1, 1], dtype=int8) # Length: 6
labels[0] = '0123movie.net'
labels[1] = '0123movie.net'
```

### Labels JSON Format

```json
{
  "label2id": {
    "0123movie.net": 0,
    "01net.com": 1,
    "104.com.tw": 2
  },
  "id2label": {
    "0": "0123movie.net",
    "1": "01net.com",
    "2": "104.com.tw"
  }
}
```

## Model Architecture

The script uses the **DFNoDefNet** model from `Model_NoDef_pytorch.py`:

- **Input**: (batch_size, 1, 5000) - Single-channel flow sequences
- **Architecture**: 
  - 4 convolutional blocks with pooling
  - 2 fully connected layers (512 units each)
  - Output: Number of classes (auto-configured)
- **Parameters**: ~3.7M (varies with number of classes)

## Performance Tips

### For Maximum GPU Performance
1. Enable mixed precision: `--use_amp`
2. Increase batch size: `--batch_size 256`
3. Use gradient accumulation: `--grad_accum_steps 4`
4. Enable pinned memory (default enabled on GPU)
5. Use multiple workers: `--num_workers 4`

### For Limited Memory
1. Reduce batch size: `--batch_size 64`
2. Reduce cache size: `--cache_size 5`
3. Use gradient accumulation to maintain effective batch size
4. Reduce number of workers: `--num_workers 2`

### For Large Datasets
1. Adjust cache size based on available RAM: `--cache_size 20`
2. Use more workers for I/O parallelism: `--num_workers 8`
3. Consider using gradient accumulation for stability

## Output

The script saves the following files in `--output_dir`:

- `best.pt`: Checkpoint with the best validation accuracy
- `last.pt`: Checkpoint from the last epoch

### Checkpoint Contents
```python
{
  'epoch': int,
  'model_state_dict': OrderedDict,
  'optimizer_state_dict': dict,
  'train_loss': float,
  'val_acc': float,
  'best_val_acc': float,
  'num_classes': int,
  'max_len': int,
  'scaler_state_dict': dict  # Only when using AMP
}
```

## Example Output

```
Using device: cuda
GPU: NVIDIA GeForce RTX 3090
CUDA Version: 11.8
Loaded 95 classes from /path/to/labels.json
Found 156 NPZ files
Split: Train=124 Val=16 Test=16

Model architecture:
  Input shape: (batch_size, 1, 5000)
  Output classes: 95
  Total parameters: 3,668,611

Starting training for 20 epochs...
Mixed precision: True
Gradient accumulation steps: 2
Effective batch size: 256

============================================================
Epoch 1/20
============================================================
Train Loss: 4.2341
Val Accuracy: 0.1234
*** New best validation accuracy: 0.1234 ***

============================================================
Epoch 2/20
============================================================
Train Loss: 3.8765
Val Accuracy: 0.2345
*** New best validation accuracy: 0.2345 ***
...
```

## Troubleshooting

### RuntimeError: mat1 and mat2 shapes cannot be multiplied
- **Cause**: Using `--max_len` other than 5000
- **Solution**: Use `--max_len 5000` (model is designed for this length)

### CUDA out of memory
- **Solution**: Reduce `--batch_size`, increase `--grad_accum_steps`, or disable `--use_amp`

### FileNotFoundError: labels.json
- **Solution**: Ensure `--labels_json` points to a valid labels.json file

### No NPZ files found
- **Solution**: Verify `--npz_root` contains .npz files in subdirectories

## License

This script is part of the DeepFingerprinting project.
