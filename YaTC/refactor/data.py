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
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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


class MFRNpzDataset(Dataset):
    """Dataset for MFR images stored in NPZ format.

    Loads MFR images from NPZ files:
    data_path/
    ├── labels.json      # {"label2id": {...}, "id2label": {...}}
    ├── class1.npz       # {"images": (N, 40, 40), "label": str, "label_id": int}
    └── class2.npz
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split: str = 'train',
        seed: int = 42
    ):
        """Initialize NPZ dataset.

        Args:
            data_path: Path to directory containing NPZ files and labels.json
            transform: Optional transform for images
            target_transform: Optional transform for labels
            train_ratio: Ratio for train split
            val_ratio: Ratio for validation split
            split: 'train', 'val', or 'test'
            seed: Random seed for reproducible splits
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Load label mapping
        labels_json = self.data_path / "labels.json"
        with open(labels_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.label2id = meta["label2id"]
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.classes = list(self.label2id.keys())
        self.num_classes = len(self.classes)

        # Load all images and labels
        all_images = []
        all_labels = []

        for label_name, label_id in self.label2id.items():
            npz_path = self.data_path / f"{label_name}.npz"
            if not npz_path.exists():
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                images = data["images"]  # (N, 40, 40)
                all_images.append(images)
                all_labels.extend([label_id] * len(images))

        if not all_images:
            raise ValueError(f"No images found in {data_path}")

        all_images = np.vstack(all_images)
        all_labels = np.array(all_labels)

        # Split data
        np.random.seed(seed)
        indices = np.random.permutation(len(all_labels))

        n_train = int(len(all_labels) * train_ratio)
        n_val = int(len(all_labels) * val_ratio)

        if split == 'train':
            idx = indices[:n_train]
        elif split == 'val':
            idx = indices[n_train:n_train + n_val]
        else:  # test
            idx = indices[n_train + n_val:]

        self.images = all_images[idx]
        self.labels = all_labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        img = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img_tensor, label


class MFRNpzPretrainDataset(Dataset):
    """Dataset for MAE pre-training from NPZ files.

    Similar to MFRNpzDataset but doesn't require labels.
    Used for self-supervised pre-training.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None
    ):
        """Initialize pre-training dataset from NPZ.

        Args:
            data_path: Path to directory containing NPZ files
            transform: Optional transform for images
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # Load all images from NPZ files
        all_images = []

        for npz_path in self.data_path.glob("*.npz"):
            with np.load(npz_path, allow_pickle=True) as data:
                if "images" in data:
                    images = data["images"]  # (N, 40, 40)
                    all_images.append(images)

        if not all_images:
            raise ValueError(f"No images found in {data_path}")

        self.images = np.vstack(all_images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Image tensor of shape (1, 40, 40)
        """
        img = self.images[idx].astype(np.float32) / 255.0

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


def build_npz_pretrain_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None
) -> DataLoader:
    """Build dataloader for pre-training from NPZ files.

    Args:
        data_path: Path to directory containing NPZ files
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform

    Returns:
        DataLoader for pre-training
    """
    dataset = MFRNpzPretrainDataset(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def build_npz_finetune_dataloader(
    data_path: Union[str, Path],
    split: str = 'train',
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, int]:
    """Build dataloader for fine-tuning from NPZ files.

    Args:
        data_path: Path to directory containing NPZ files and labels.json
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform for images
        target_transform: Optional transform for labels
        train_ratio: Ratio for train split
        val_ratio: Ratio for validation split
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (DataLoader, num_classes)
    """
    dataset = MFRNpzDataset(
        data_path,
        transform=transform,
        target_transform=target_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split=split,
        seed=seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    return dataloader, dataset.num_classes
