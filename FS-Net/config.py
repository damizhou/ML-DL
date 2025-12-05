"""
FS-Net Configuration

Configuration and hyperparameters for the FS-Net model as specified in the paper:
"FS-Net: A Flow Sequence Network For Encrypted Traffic Classification"

All parameters are consistent with the paper.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EmbeddingConfig:
    """Embedding layer configuration.

    The embedding layer converts packet lengths to dense vectors.
    """
    vocab_size: int = 3002  # Signed lengths: -1500 to +1500 -> 1 to 3001, 0 for padding
    embed_dim: int = 128
    padding_idx: int = 0


@dataclass
class EncoderConfig:
    """Encoder configuration.

    The encoder uses stacked bidirectional GRUs to learn flow representations.
    From paper: 2-layer bi-GRU with 128 hidden dimensions.
    """
    input_dim: int = 128  # Same as embedding dim
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True


@dataclass
class DecoderConfig:
    """Decoder configuration.

    The decoder reconstructs the input sequence from encoder features.
    Same architecture as encoder.
    """
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True


@dataclass
class DenseConfig:
    """Dense layer configuration.

    Combines encoder and decoder features:
    z = [ze, zd, ze ⊙ zd, |ze - zd|]

    Then compresses with two-layer perceptron + SELU activation.
    """
    # Input dim = 4 * (num_layers * 2 * hidden_dim) for bidirectional
    # = 4 * (2 * 2 * 128) = 4 * 512 = 2048
    hidden_dim: int = 256
    dropout: float = 0.3


@dataclass
class FSNetConfig:
    """Complete FS-Net configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)

    # Reconstruction loss weight
    alpha: float = 1.0

    # Maximum sequence length
    max_seq_len: int = 100


@dataclass
class TrainConfig:
    """Training configuration.

    From paper:
    - Adam optimizer with lr=0.0005
    - Dropout 0.3
    - 5-fold cross validation
    """
    batch_size: int = 128
    learning_rate: float = 0.0005
    epochs: int = 100
    dropout: float = 0.3
    alpha: float = 1.0  # Reconstruction loss weight

    # Early stopping
    patience: int = 10

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_seq_len: int = 100  # Maximum number of packets per flow
    max_packet_len: int = 1500  # Maximum packet length (MTU)
    min_packets: int = 2  # Minimum packets per flow


# Default configurations
DEFAULT_FSNET_CONFIG = FSNetConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_DATA_CONFIG = DataConfig()
