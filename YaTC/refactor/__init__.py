"""
YaTC: Yet Another Traffic Classifier

A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation.

This package provides a refactored implementation of the YaTC model for encrypted
network traffic classification, compatible with Python 3.12 and PyTorch 2.9.

Modules:
    config: Configuration and hyperparameters
    models: Model definitions (MAE_YaTC, TraFormer_YaTC)
    data: Data processing and MFR generation
    engine: Training and evaluation engine
    train: Main training script
"""

from config import (
    MFRConfig,
    PatchEmbedConfig,
    EncoderConfig,
    DecoderConfig,
    MAEConfig,
    TraFormerConfig,
    PretrainConfig,
    FinetuneConfig,
    DatasetConfig,
    DEFAULT_MAE_CONFIG,
    DEFAULT_TRAFORMER_CONFIG,
    DEFAULT_PRETRAIN_CONFIG,
    DEFAULT_FINETUNE_CONFIG,
)

from models import (
    MAE_YaTC,
    TraFormer_YaTC,
    mae_yatc,
    traformer_yatc,
    PatchEmbed,
    Block,
    Attention,
    Mlp,
)

from data import (
    MFRGenerator,
    MFRDataset,
    MFRPretrainDataset,
    build_pretrain_dataloader,
    build_finetune_dataloader,
)

from engine import (
    pretrain_one_epoch,
    train_one_epoch,
    evaluate,
    load_pretrained_weights,
    save_checkpoint,
)

__version__ = "1.0.0"
__author__ = "YaTC Team"

__all__ = [
    # Config
    "MFRConfig",
    "PatchEmbedConfig",
    "EncoderConfig",
    "DecoderConfig",
    "MAEConfig",
    "TraFormerConfig",
    "PretrainConfig",
    "FinetuneConfig",
    "DatasetConfig",
    "DEFAULT_MAE_CONFIG",
    "DEFAULT_TRAFORMER_CONFIG",
    "DEFAULT_PRETRAIN_CONFIG",
    "DEFAULT_FINETUNE_CONFIG",
    # Models
    "MAE_YaTC",
    "TraFormer_YaTC",
    "mae_yatc",
    "traformer_yatc",
    "PatchEmbed",
    "Block",
    "Attention",
    "Mlp",
    # Data
    "MFRGenerator",
    "MFRDataset",
    "MFRPretrainDataset",
    "build_pretrain_dataloader",
    "build_finetune_dataloader",
    # Engine
    "pretrain_one_epoch",
    "train_one_epoch",
    "evaluate",
    "load_pretrained_weights",
    "save_checkpoint",
]
