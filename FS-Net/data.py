"""
FS-Net Data Processing

Data processing utilities for FS-Net:
- PCAP to packet length sequence conversion
- Dataset classes for training and evaluation
- Data augmentation and collation

Input format: PCAP files organized by class
Output format: Packet length sequences with labels
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import DEFAULT_DATA_CONFIG, DataConfig

import logging

def log(message: str = ""):
    """Log message using configured logger."""
    logging.info(message)

try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False
    log("Warning: dpkt not installed. PCAP processing will not be available.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def extract_packet_lengths(pcap_path: str, config: DataConfig = None) -> List[int]:
    """Extract packet length sequence from PCAP file.

    Uses dpkt for fast streaming PCAP parsing (memory efficient).

    Args:
        pcap_path: Path to PCAP file
        config: Data configuration

    Returns:
        List of packet lengths (positive for outgoing, negative for incoming)
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required for PCAP processing")

    import socket

    if config is None:
        config = DEFAULT_DATA_CONFIG

    lengths = []
    first_src = None

    try:
        with open(pcap_path, 'rb') as f:
            # Try pcap format first, then pcapng
            try:
                pcap = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            for ts, buf in pcap:
                # Parse Ethernet frame
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue

                # Check for IP layer
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data
                pkt_len = len(buf)

                # Cap packet length
                if pkt_len > config.max_packet_len:
                    pkt_len = config.max_packet_len

                # Get source IP as string
                src_ip = socket.inet_ntoa(ip.src)

                # Determine direction based on first packet's source
                if first_src is None:
                    first_src = src_ip
                    lengths.append(pkt_len)  # First packet is outgoing (positive)
                else:
                    if src_ip == first_src:
                        lengths.append(pkt_len)  # Outgoing (positive)
                    else:
                        lengths.append(-pkt_len)  # Incoming (negative)

                # Limit sequence length
                if len(lengths) >= config.max_seq_len:
                    break

    except Exception as e:
        log(f"Error reading {pcap_path}: {e}")
        return []

    return lengths


def lengths_to_tensor(
    lengths: List[int],
    max_len: int = 1500,
    signed: bool = True
) -> torch.Tensor:
    """Convert packet lengths to tensor indices.

    Args:
        lengths: List of packet lengths (can be negative for direction)
        max_len: Maximum packet length
        signed: If True, use signed lengths (with direction)

    Returns:
        Tensor of indices
    """
    if signed:
        # Map: [-max_len, max_len] -> [1, 2*max_len+1], 0 for padding
        indices = [l + max_len + 1 if l != 0 else 0 for l in lengths]
    else:
        # Map: [1, max_len] -> [1, max_len], 0 for padding
        indices = [abs(l) for l in lengths]

    return torch.tensor(indices, dtype=torch.long)


class FlowSequenceDataset(Dataset):
    """Dataset for packet length sequences.

    Loads pre-processed packet length sequences from directory structure:
    data_path/
    ├── train/
    │   ├── class1/
    │   │   └── *.json  (or *.pcap)
    │   └── class2/
    └── test/
        ├── class1/
        └── class2/

    JSON format: {"lengths": [l1, l2, ...], "label": "class_name"}
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        config: DataConfig = None,
        use_direction: bool = False
    ):
        """Initialize dataset.

        Args:
            data_path: Root path to dataset
            split: 'train', 'val', or 'test'
            config: Data configuration
            use_direction: If True, use signed packet lengths
        """
        self.data_path = Path(data_path) / split
        self.config = config or DEFAULT_DATA_CONFIG
        self.use_direction = use_direction

        # Get class names
        self.classes = sorted([
            d.name for d in self.data_path.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect samples
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_path = self.data_path / cls_name
            cls_idx = self.class_to_idx[cls_name]

            # Support both JSON and PCAP
            for ext in ['*.json', '*.pcap']:
                for file_path in cls_path.glob(ext):
                    self.samples.append((file_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index.

        Returns:
            Tuple of (sequence tensor, label)
        """
        file_path, label = self.samples[idx]

        if file_path.suffix == '.json':
            lengths = self._load_json(file_path)
        else:
            lengths = extract_packet_lengths(str(file_path), self.config)

        # Convert to tensor
        if self.use_direction:
            # Signed lengths: [-1500, 1500] -> [1, 3001], 0 for padding
            max_val = self.config.max_packet_len
            indices = [l + max_val + 1 for l in lengths]
        else:
            # Unsigned: [1, 1500], 0 for padding
            indices = [abs(l) for l in lengths]

        sequence = torch.tensor(indices, dtype=torch.long)

        return sequence, label

    def _load_json(self, path: Path) -> List[int]:
        """Load lengths from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('lengths', [])


class PCAPDataset(Dataset):
    """Dataset that directly loads PCAP files."""

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        config: DataConfig = None
    ):
        self.data_path = Path(data_path) / split
        self.config = config or DEFAULT_DATA_CONFIG

        # Get class names
        self.classes = sorted([
            d.name for d in self.data_path.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect PCAP files
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_path = self.data_path / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for pcap_path in cls_path.glob('*.pcap'):
                self.samples.append((pcap_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pcap_path, label = self.samples[idx]

        # Extract packet lengths
        lengths = extract_packet_lengths(str(pcap_path), self.config)

        # Use absolute lengths (unsigned)
        indices = [min(abs(l), self.config.max_packet_len) for l in lengths]

        # Ensure minimum length
        if len(indices) < self.config.min_packets:
            indices = indices + [0] * (self.config.min_packets - len(indices))

        sequence = torch.tensor(indices, dtype=torch.long)

        return sequence, label


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.

    Pads sequences to same length within batch.

    Args:
        batch: List of (sequence, label) tuples

    Returns:
        sequences: Padded sequences (batch, max_seq_len)
        lengths: Original sequence lengths (batch,)
        labels: Labels (batch,)
    """
    sequences, labels = zip(*batch)

    # Get lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return sequences_padded, lengths, labels


def build_dataloader(
    data_path: Union[str, Path],
    split: str = 'train',
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    config: DataConfig = None,
    use_direction: bool = True
) -> DataLoader:
    """Build DataLoader for FS-Net.

    Args:
        data_path: Path to dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        config: Data configuration
        use_direction: If True, use signed packet lengths (direction info)

    Returns:
        DataLoader
    """
    dataset = FlowSequenceDataset(data_path, split, config, use_direction=use_direction)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )


def pcap_to_sequences(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: DataConfig = None
):
    """Convert PCAP files to JSON sequence files.

    Args:
        input_dir: Input directory with PCAP files
        output_dir: Output directory for JSON files
        config: Data configuration
    """
    if not DPKT_AVAILABLE:
        raise ImportError("dpkt is required for PCAP processing")

    config = config or DEFAULT_DATA_CONFIG
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all PCAP files
    pcap_files = list(input_path.rglob('*.pcap'))
    log(f"Found {len(pcap_files)} PCAP files")

    for pcap_file in tqdm(pcap_files, desc="Converting"):
        # Compute relative path
        rel_path = pcap_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix('.json')

        # Create output directory
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Extract lengths
        lengths = extract_packet_lengths(str(pcap_file), config)

        if len(lengths) >= config.min_packets:
            # Save as JSON
            with open(out_file, 'w') as f:
                json.dump({'lengths': lengths}, f)

    log(f"Converted to {output_path}")


def get_dataset_info(data_path: Union[str, Path]) -> Dict:
    """Get dataset information.

    Args:
        data_path: Path to dataset

    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_path)
    info = {
        'classes': [],
        'num_classes': 0,
        'splits': {},
        'total_samples': 0
    }

    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            continue

        split_info = {}
        for cls_dir in sorted(split_path.iterdir()):
            if cls_dir.is_dir():
                num_files = len(list(cls_dir.glob('*')))
                split_info[cls_dir.name] = num_files
                if cls_dir.name not in info['classes']:
                    info['classes'].append(cls_dir.name)

        info['splits'][split] = split_info
        info['total_samples'] += sum(split_info.values())

    info['num_classes'] = len(info['classes'])

    return info


# =============================================================================
# Pickle Dataset Support (new format from iscx_vpn_processor.py)
# =============================================================================

import pickle


def load_pickle_dataset(pickle_path: str) -> Tuple[List[List[int]], np.ndarray, Dict[int, str]]:
    """Load dataset from pickle file.

    Args:
        pickle_path: Path to pickle file

    Returns:
        sequences: List of variable-length sequences
        labels: numpy array of labels
        label_map: mapping from label_id to class name
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    labels = data['labels']
    label_map = data['label_map']

    return sequences, labels, label_map


class SequenceDataset(Dataset):
    """Dataset for pre-loaded sequences (from pickle)."""

    def __init__(
        self,
        sequences: List[List[int]],
        labels: np.ndarray,
        max_packet_len: int = 1500,
        use_direction: bool = True
    ):
        """Initialize dataset.

        Args:
            sequences: List of variable-length sequences (signed packet lengths)
            labels: numpy array of labels
            max_packet_len: Maximum packet length for index mapping
            use_direction: If True, use signed lengths
        """
        self.sequences = sequences
        self.labels = labels
        self.max_packet_len = max_packet_len
        self.use_direction = use_direction

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        lengths = self.sequences[idx]
        label = int(self.labels[idx])

        if self.use_direction:
            # Signed lengths: [-1500, 1500] -> [1, 3001], 0 for padding
            indices = [l + self.max_packet_len + 1 for l in lengths]
        else:
            # Unsigned: [1, 1500], 0 for padding
            indices = [abs(l) for l in lengths]

        sequence = torch.tensor(indices, dtype=torch.long)

        return sequence, label


def create_dataloaders(
    sequences: List[List[int]],
    labels: np.ndarray,
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    max_packet_len: int = 1500,
    use_direction: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Default split ratio is 8:1:1 (train:val:test) by flow.

    Args:
        sequences: List of variable-length sequences
        labels: numpy array of labels
        batch_size: Batch size
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        seed: Random seed
        num_workers: Number of data loading workers
        max_packet_len: Maximum packet length
        use_direction: If True, use signed lengths

    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(seed)

    # Shuffle indices
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)

    # Split with 8:1:1 ratio
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create split datasets
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = labels[train_indices]

    val_sequences = [sequences[i] for i in val_indices]
    val_labels = labels[val_indices]

    test_sequences = [sequences[i] for i in test_indices]
    test_labels = labels[test_indices]

    # Create datasets
    train_dataset = SequenceDataset(train_sequences, train_labels, max_packet_len, use_direction)
    val_dataset = SequenceDataset(val_sequences, val_labels, max_packet_len, use_direction)
    test_dataset = SequenceDataset(test_sequences, test_labels, max_packet_len, use_direction)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    log(f"Dataset split (8:1:1):")
    log(f"  Train: {len(train_dataset)} samples")
    log(f"  Val:   {len(val_dataset)} samples")
    log(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
