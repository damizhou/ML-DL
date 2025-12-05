"""
YaTC Model Configuration

This module contains all hyperparameters and configurations for the YaTC
(Yet Another Traffic Classifier) model, as specified in the AAAI 2023 paper:
"Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer"

All parameters are consistent with the paper and original implementation.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MFRConfig:
    """Multi-level Flow Representation (MFR) configuration.

    MFR converts network flows into 40x40 grayscale images:
    - Each flow contains 5 packets
    - Each packet: 160 hex chars header + 480 hex chars payload = 320 bytes
    - Each packet represented as 8 rows × 40 columns
    - Total: 5 packets × 8 rows = 40 rows × 40 columns
    """
    num_packets: int = 5
    bytes_per_packet: int = 320
    header_bytes: int = 80  # 160 hex chars / 2
    payload_bytes: int = 240  # 480 hex chars / 2
    rows_per_packet: int = 8  # 320 bytes / 40 columns
    img_size: int = 40
    in_channels: int = 1  # Grayscale


@dataclass
class PatchEmbedConfig:
    """Patch embedding configuration.

    The patch embedding treats each packet as a separate unit:
    - Per-packet image size: (8, 40) = (img_size/5, img_size)
    - Patch size: 2x2
    - Patches per packet: (8/2) × (40/2) = 4 × 20 = 80
    - Total patches: 80 × 5 = 400
    """
    img_size: Tuple[int, int] = (8, 40)  # Per-packet: (img_size/5, img_size)
    patch_size: Tuple[int, int] = (2, 2)
    num_patches_per_packet: int = 80  # (8/2) × (40/2)
    num_packets: int = 5
    total_patches: int = 400  # 80 × 5


@dataclass
class EncoderConfig:
    """Encoder configuration for both MAE and TraFormer.

    Architecture (from paper Table 1):
    - Embedding dimension: 192
    - Number of layers: 4
    - Number of attention heads: 16
    - MLP ratio: 4
    - Head dimension: 192 / 16 = 12
    """
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    norm_eps: float = 1e-6


@dataclass
class DecoderConfig:
    """Decoder configuration for MAE pre-training.

    Architecture (from paper Table 1):
    - Embedding dimension: 128
    - Number of layers: 2
    - Number of attention heads: 16
    - MLP ratio: 4
    """
    embed_dim: int = 128
    depth: int = 2
    num_heads: int = 16
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6


@dataclass
class MAEConfig:
    """Masked Autoencoder configuration for pre-training.

    Key parameters:
    - Mask ratio: 0.9 (90% of patches are masked)
    - Input: 40x40 grayscale MFR image
    - Patch size: 2x2
    - Total patches: 400
    """
    img_size: int = 40
    patch_size: int = 2
    in_channels: int = 1
    mask_ratio: float = 0.9
    encoder: EncoderConfig = None
    decoder: DecoderConfig = None

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()
        if self.decoder is None:
            self.decoder = DecoderConfig()


@dataclass
class TraFormerConfig:
    """Traffic Transformer configuration for fine-tuning.

    Inherits encoder architecture from MAE pre-training.
    Classification head is added on top.
    """
    img_size: int = 40
    patch_size: int = 2
    in_channels: int = 1
    encoder: EncoderConfig = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()


@dataclass
class PretrainConfig:
    """Pre-training hyperparameters (from paper).

    Optimizer: AdamW
    - Base learning rate: 1e-3
    - Weight decay: 0.05
    - Betas: (0.9, 0.95)

    Training:
    - Batch size: 128
    - Steps: 150,000
    - Warmup steps: 10,000
    """
    batch_size: int = 128
    base_lr: float = 1e-3
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    total_steps: int = 150000
    warmup_steps: int = 10000
    min_lr: float = 0.0
    mask_ratio: float = 0.9


@dataclass
class FinetuneConfig:
    """Fine-tuning hyperparameters (from paper).

    Optimizer: AdamW
    - Base learning rate: 2e-3
    - Weight decay: 0.05
    - Betas: (0.9, 0.999)

    Training:
    - Epochs: 200
    - Warmup epochs: 5
    - Layer-wise learning rate decay: 0.65
    """
    batch_size: int = 128
    base_lr: float = 2e-3
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    epochs: int = 200
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    layer_decay: float = 0.65
    drop_path: float = 0.1
    mixup: float = 0.0
    cutmix: float = 0.0
    smoothing: float = 0.1  # Label smoothing


@dataclass
class DatasetConfig:
    """Dataset configurations.

    Supported datasets:
    - ISCXVPN2016: 7 classes (VPN traffic classification)
    - ISCXTor2016: 8 classes (Tor traffic classification)
    - USTC-TFC2016: 20 classes (malware traffic classification)
    - CICIoT2022: 10 classes (IoT traffic classification)
    """
    ISCXVPN2016_CLASSES: int = 7
    ISCXTor2016_CLASSES: int = 8
    USTC_TFC2016_CLASSES: int = 20
    CICIoT2022_CLASSES: int = 10


# Default configurations
DEFAULT_MAE_CONFIG = MAEConfig()
DEFAULT_TRAFORMER_CONFIG = TraFormerConfig()
DEFAULT_PRETRAIN_CONFIG = PretrainConfig()
DEFAULT_FINETUNE_CONFIG = FinetuneConfig()
DEFAULT_MFR_CONFIG = MFRConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
