"""
YaTC Data Processing

This module implements data processing utilities for the YaTC model:
- MFR (Multi-level Flow Representation) generation from PCAP files
- Dataset classes for pre-training and fine-tuning
- Data augmentation and transforms

MFR Format:
- Each flow is converted to a 40x40 grayscale image
- 5 packets per flow
- Each packet: 160 hex chars header + 480 hex chars payload = 320 bytes
- Each packet represented as 8 rows × 40 columns
- Total: 5 × 8 = 40 rows × 40 columns
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from config import DEFAULT_MFR_CONFIG, MFRConfig


class MFRGenerator:
    """Generate Multi-level Flow Representation (MFR) from network packets.

    MFR converts network flows into 40x40 grayscale images:
    - Each flow contains 5 packets
    - Each packet: 80 bytes header + 240 bytes payload = 320 bytes
    - Each packet represented as 8 rows × 40 columns
    - Total: 5 packets × 8 rows = 40 rows × 40 columns

    This representation captures both spatial (within packet) and
    temporal (across packets) patterns in network traffic.
    """

    def __init__(self, config: MFRConfig = None):
        """Initialize MFR generator.

        Args:
            config: MFR configuration. If None, uses default.
        """
        self.config = config or DEFAULT_MFR_CONFIG

    def packet_to_bytes(
        self,
        packet_hex: str,
        header_len: int = 160,
        payload_len: int = 480
    ) -> np.ndarray:
        """Convert packet hex string to byte array.

        Args:
            packet_hex: Hexadecimal string representation of packet
            header_len: Number of hex characters for header (160 = 80 bytes)
            payload_len: Number of hex characters for payload (480 = 240 bytes)

        Returns:
            Byte array of shape (320,)
        """
        total_len = header_len + payload_len

        # Pad or truncate to required length
        if len(packet_hex) < total_len:
            packet_hex = packet_hex + '0' * (total_len - len(packet_hex))
        else:
            packet_hex = packet_hex[:total_len]

        # Convert hex to bytes
        bytes_arr = np.array([
            int(packet_hex[i:i+2], 16)
            for i in range(0, total_len, 2)
        ], dtype=np.uint8)

        return bytes_arr

    def flow_to_mfr(
        self,
        packets: List[str],
        num_packets: int = 5
    ) -> np.ndarray:
        """Convert a flow (list of packets) to MFR matrix.

        Args:
            packets: List of packet hex strings
            num_packets: Number of packets to use (default: 5)

        Returns:
            MFR matrix of shape (40, 40)
        """
        mfr = np.zeros((40, 40), dtype=np.uint8)

        for i in range(min(num_packets, len(packets))):
            packet_bytes = self.packet_to_bytes(packets[i])
            # Each packet: 320 bytes -> 8 rows × 40 columns
            packet_matrix = packet_bytes.reshape(8, 40)
            mfr[i * 8:(i + 1) * 8, :] = packet_matrix

        return mfr

    def save_mfr_as_png(
        self,
        mfr: np.ndarray,
        output_path: Union[str, Path]
    ):
        """Save MFR matrix as PNG image.

        Args:
            mfr: MFR matrix of shape (40, 40)
            output_path: Path to save PNG file
        """
        img = Image.fromarray(mfr, mode='L')
        img.save(output_path)

    def load_mfr_from_png(
        self,
        png_path: Union[str, Path]
    ) -> np.ndarray:
        """Load MFR matrix from PNG image.

        Args:
            png_path: Path to PNG file

        Returns:
            MFR matrix of shape (40, 40)
        """
        img = Image.open(png_path).convert('L')
        mfr = np.array(img, dtype=np.uint8)
        return mfr


class MFRDataset(Dataset):
    """Dataset for MFR images.

    Loads MFR images from directory structure:
    data_path/
    ├── train/
    │   ├── class1/
    │   │   └── *.png
    │   └── class2/
    │       └── *.png
    └── test/
        ├── class1/
        └── class2/
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """Initialize MFR dataset.

        Args:
            data_path: Root path to dataset
            split: 'train' or 'test'
            transform: Optional transform for images
            target_transform: Optional transform for labels
        """
        self.data_path = Path(data_path) / split
        self.transform = transform
        self.target_transform = target_transform

        # Get class names from directory structure
        self.classes = sorted([
            d.name for d in self.data_path.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all samples
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_path = self.data_path / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in cls_path.glob('*.png'):
                self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img_tensor, label


class MFRPretrainDataset(Dataset):
    """Dataset for MAE pre-training.

    Similar to MFRDataset but doesn't require labels.
    Used for self-supervised pre-training.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None
    ):
        """Initialize pre-training dataset.

        Args:
            data_path: Path to directory containing PNG images
            transform: Optional transform for images
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # Collect all PNG files recursively
        self.samples: List[Path] = list(self.data_path.rglob('*.png'))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Image tensor of shape (1, 40, 40)
        """
        img_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor


def build_pretrain_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None
) -> DataLoader:
    """Build dataloader for pre-training.

    Args:
        data_path: Path to dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform

    Returns:
        DataLoader for pre-training
    """
    dataset = MFRPretrainDataset(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def build_finetune_dataloader(
    data_path: Union[str, Path],
    split: str = 'train',
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None
) -> DataLoader:
    """Build dataloader for fine-tuning.

    Args:
        data_path: Path to dataset
        split: 'train' or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform for images
        target_transform: Optional transform for labels

    Returns:
        DataLoader for fine-tuning
    """
    dataset = MFRDataset(
        data_path,
        split=split,
        transform=transform,
        target_transform=target_transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    return dataloader
