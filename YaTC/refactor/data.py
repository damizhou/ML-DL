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
        seed: int = 42,
        min_samples: int = 10
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
            min_samples: Minimum samples per class, classes with fewer samples are removed
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Load label mapping
        labels_json = self.data_path / "labels.json"
        with open(labels_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label2id_orig = meta["label2id"]
        id2label_orig = {int(k): v for k, v in meta["id2label"].items()}

        # Load all images and labels, filter by min_samples
        all_images = []
        all_labels = []
        kept_classes = []
        removed_classes = []

        for label_name, label_id in label2id_orig.items():
            npz_path = self.data_path / f"{label_name}.npz"
            if not npz_path.exists():
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                images = data["images"]  # (N, 40, 40)

                if len(images) < min_samples:
                    removed_classes.append((label_name, len(images)))
                    continue

                kept_classes.append(label_id)
                all_images.append(images)
                all_labels.extend([label_id] * len(images))

        if removed_classes and split == 'train':
            print(f"\n[Warning] 以下类别样本数不足 {min_samples}，已剔除:")
            for label_name, count in removed_classes:
                print(f"  - {label_name}: {count} 个样本")

        if not all_images:
            raise ValueError(f"No images found in {data_path}")

        all_images = np.vstack(all_images)
        all_labels = np.array(all_labels)

        # Remap labels to continuous 0, 1, 2, ...
        old_to_new = {old_label: new_label for new_label, old_label in enumerate(kept_classes)}
        all_labels = np.array([old_to_new[y] for y in all_labels])

        # Update label mappings
        self.label2id = {id2label_orig[old]: new for old, new in old_to_new.items()}
        self.id2label = {new: id2label_orig[old] for old, new in old_to_new.items()}
        self.classes = list(self.label2id.keys())
        self.num_classes = len(kept_classes)

        # Stratified split: each class split by ratio
        np.random.seed(seed)

        train_images, train_labels = [], []
        val_images, val_labels = [], []
        test_images, test_labels = [], []

        for label in range(self.num_classes):
            class_indices = np.where(all_labels == label)[0]
            np.random.shuffle(class_indices)

            n_train = int(len(class_indices) * train_ratio)
            n_val = int(len(class_indices) * val_ratio)

            train_idx = class_indices[:n_train]
            val_idx = class_indices[n_train:n_train + n_val]
            test_idx = class_indices[n_train + n_val:]

            train_images.append(all_images[train_idx])
            train_labels.extend([label] * len(train_idx))

            val_images.append(all_images[val_idx])
            val_labels.extend([label] * len(val_idx))

            test_images.append(all_images[test_idx])
            test_labels.extend([label] * len(test_idx))

        if split == 'train':
            self.images = np.vstack(train_images)
            self.labels = np.array(train_labels)
        elif split == 'val':
            self.images = np.vstack(val_images)
            self.labels = np.array(val_labels)
        else:  # test
            self.images = np.vstack(test_images)
            self.labels = np.array(test_labels)

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


class MFRNpzSplitDataset(Dataset):
    """Dataset for MFR images from pre-split NPZ directories.

    Loads MFR images from split_dataset.py output:
    data_path/
    ├── labels.json      # {"label2id": {...}, "id2label": {...}}
    ├── class1.npz       # {"images": (N, 40, 40)}
    └── class2.npz

    Unlike MFRNpzDataset, this class loads from a single split directory
    without performing runtime splitting.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """Initialize split dataset.

        Args:
            data_path: Path to split directory (e.g., data/xxx_split/train)
            transform: Optional transform for images
            target_transform: Optional transform for labels
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform

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

        self.images = np.vstack(all_images)
        self.labels = np.array(all_labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index."""
        img = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])

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

    Uses memory-mapped loading to avoid memory spikes during initialization.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None,
        log_fn: Optional[Callable[[str], None]] = None
    ):
        """Initialize pre-training dataset from NPZ.

        Args:
            data_path: Path to directory containing NPZ files
            transform: Optional transform for images
            log_fn: Optional logging function (default: print)
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self._log = log_fn if log_fn is not None else print

        # 第一遍扫描：统计总样本数
        self._log(f"Scanning NPZ files from {data_path}...")
        self.npz_files = sorted(self.data_path.glob("*.npz"))
        self._log(f"Found {len(self.npz_files)} NPZ files, scanning for sample counts...")

        self.file_sample_counts = []
        total_samples = 0

        for i, npz_path in enumerate(self.npz_files):
            with np.load(npz_path, allow_pickle=True) as data:
                if "images" in data:
                    n = len(data["images"])
                    self.file_sample_counts.append(n)
                    total_samples += n
                else:
                    self.file_sample_counts.append(0)

            if (i + 1) % 100 == 0 or (i + 1) == len(self.npz_files):
                self._log(f"  Scanned {i + 1}/{len(self.npz_files)} files ({total_samples} samples so far)")

        self._log(f"Scan complete: {total_samples} samples total")

        # 预分配大数组，避免 vstack 的内存峰值
        self._log(f"Pre-allocating array for {total_samples} samples...")
        self.images = np.empty((total_samples, 40, 40), dtype=np.uint8)

        # 第二遍加载：直接填充到预分配数组
        self._log(f"Loading data into pre-allocated array...")
        offset = 0
        for i, npz_path in enumerate(self.npz_files):
            n = self.file_sample_counts[i]
            if n == 0:
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                self.images[offset:offset + n] = data["images"]
            offset += n

            if (i + 1) % 100 == 0 or (i + 1) == len(self.npz_files):
                self._log(f"  Loaded {i + 1}/{len(self.npz_files)} files ({offset}/{total_samples} samples)")

        self._log(f"Dataset ready: {len(self.images)} samples")

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
    num_workers: int = 8,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    log_fn: Optional[Callable[[str], None]] = None
) -> DataLoader:
    """Build dataloader for pre-training from NPZ files.

    Args:
        data_path: Path to directory containing NPZ files
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep worker processes alive between epochs
        log_fn: Optional logging function (default: print)

    Returns:
        DataLoader for pre-training
    """
    dataset = MFRNpzPretrainDataset(data_path, transform=transform, log_fn=log_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
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
    seed: int = 42,
    min_samples: int = 10
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
        min_samples: Minimum samples per class, classes with fewer samples are removed

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
        seed=seed,
        min_samples=min_samples
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


def build_split_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None
) -> Tuple[DataLoader, int]:
    """Build dataloader for pre-split NPZ directory.

    Args:
        data_path: Path to split directory (e.g., data/xxx_split/train)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transform for images
        target_transform: Optional transform for labels

    Returns:
        Tuple of (DataLoader, num_classes)
    """
    dataset = MFRNpzSplitDataset(
        data_path,
        transform=transform,
        target_transform=target_transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle  # drop_last only when shuffling (training)
    )
    return dataloader, dataset.num_classes
