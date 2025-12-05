"""
FS-Net Model Definition

Implementation of the Flow Sequence Network (FS-Net) for encrypted traffic classification
as described in the paper:
"FS-Net: A Flow Sequence Network For Encrypted Traffic Classification"

Architecture:
1. Embedding Layer - converts packet lengths to dense vectors
2. Encoder Layer - multi-layer bi-GRU to learn flow representations
3. Decoder Layer - multi-layer bi-GRU to reconstruct input
4. Reconstruction Layer - softmax to predict original sequence
5. Dense Layer - combines encoder and decoder features
6. Classification Layer - softmax classifier

All parameters are consistent with the paper.
Compatible with PyTorch 2.9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

from config import (
    DEFAULT_FSNET_CONFIG,
    FSNetConfig,
    EmbeddingConfig,
    EncoderConfig,
    DecoderConfig,
    DenseConfig,
)


class EmbeddingLayer(nn.Module):
    """Embedding layer for packet lengths.

    Converts discrete packet length values to dense embedding vectors.
    From paper: embedding dimension = 128
    """

    def __init__(self, config: EmbeddingConfig = None):
        super().__init__()
        if config is None:
            config = EmbeddingConfig()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Packet length sequence of shape (batch, seq_len)

        Returns:
            Embedded sequence of shape (batch, seq_len, embed_dim)
        """
        return self.embedding(x)


class EncoderLayer(nn.Module):
    """Encoder layer using stacked bi-directional GRU.

    From paper:
    - 2-layer bi-GRU
    - Hidden dimension: 128
    - Dropout: 0.3
    """

    def __init__(self, config: EncoderConfig = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()

        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Embedded sequence of shape (batch, seq_len, embed_dim)
            lengths: Sequence lengths of shape (batch,)

        Returns:
            outputs: All hidden states (batch, seq_len, hidden_dim * num_directions)
            ze: Encoder feature vector (batch, num_layers * num_directions * hidden_dim)
        """
        # Pack sequence for efficient computation
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward through GRU
        packed_output, hidden = self.gru(packed)

        # Unpack output
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        # Reshape to (batch, num_layers * num_directions * hidden_dim)
        batch_size = x.size(0)
        ze = hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return outputs, ze


class DecoderLayer(nn.Module):
    """Decoder layer using stacked bi-directional GRU.

    Takes encoder feature ze as input at each time step.
    From paper: same architecture as encoder.
    """

    def __init__(self, encoder_feature_dim: int, config: DecoderConfig = None):
        super().__init__()
        if config is None:
            config = DecoderConfig()

        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1

        self.gru = nn.GRU(
            input_size=encoder_feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

    def forward(
        self,
        ze: torch.Tensor,
        seq_len: int,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ze: Encoder feature vector (batch, encoder_feature_dim)
            seq_len: Maximum sequence length
            lengths: Actual sequence lengths (batch,)

        Returns:
            outputs: Decoder output sequence (batch, seq_len, hidden_dim * num_directions)
            zd: Decoder feature vector (batch, num_layers * num_directions * hidden_dim)
        """
        batch_size = ze.size(0)

        # Repeat ze for each time step
        # (batch, encoder_feature_dim) -> (batch, seq_len, encoder_feature_dim)
        ze_expanded = ze.unsqueeze(1).expand(-1, seq_len, -1)

        # Pack sequence
        packed = pack_padded_sequence(
            ze_expanded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward through GRU
        packed_output, hidden = self.gru(packed)

        # Unpack output
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get decoder feature vector
        zd = hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return outputs, zd


class ReconstructionLayer(nn.Module):
    """Reconstruction layer.

    Predicts the original packet length at each time step.
    Uses softmax over the vocabulary.
    """

    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_outputs: Decoder output sequence (batch, seq_len, input_dim)

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        return self.fc(decoder_outputs)


class DenseLayer(nn.Module):
    """Dense layer for feature combination and compression.

    Combines encoder and decoder features:
    z = [ze, zd, ze ⊙ zd, |ze - zd|]

    Then applies two-layer perceptron with SELU activation.
    """

    def __init__(self, feature_dim: int, config: DenseConfig = None):
        super().__init__()
        if config is None:
            config = DenseConfig()

        # Input: [ze, zd, ze * zd, |ze - zd|] = 4 * feature_dim
        input_dim = 4 * feature_dim

        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ze: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ze: Encoder feature vector (batch, feature_dim)
            zd: Decoder feature vector (batch, feature_dim)

        Returns:
            Compressed feature vector (batch, hidden_dim)
        """
        # Combine features: [ze, zd, ze ⊙ zd, |ze - zd|]
        z = torch.cat([
            ze,
            zd,
            ze * zd,
            torch.abs(ze - zd)
        ], dim=-1)

        # Two-layer perceptron with SELU activation
        z = F.selu(self.fc1(z))
        z = self.dropout(z)
        z = F.selu(self.fc2(z))
        z = self.dropout(z)

        return z


class ClassificationLayer(nn.Module):
    """Classification layer.

    Softmax classifier over applications.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vector (batch, input_dim)

        Returns:
            Logits over classes (batch, num_classes)
        """
        return self.fc(x)


class FSNet(nn.Module):
    """Flow Sequence Network (FS-Net) for encrypted traffic classification.

    End-to-end model that:
    1. Embeds packet length sequences
    2. Encodes sequences with bi-GRU
    3. Decodes with bi-GRU for reconstruction
    4. Combines features for classification

    From paper:
    - Embedding dimension: 128
    - Hidden dimension: 128
    - Number of bi-GRU layers: 2
    - Dropout: 0.3
    - Loss: L = LC + α * LR (α = 1)
    """

    def __init__(self, num_classes: int, config: FSNetConfig = None, class_weight: torch.Tensor = None):
        super().__init__()
        if config is None:
            config = DEFAULT_FSNET_CONFIG

        self.config = config
        self.num_classes = num_classes
        self.register_buffer('class_weight', class_weight)  # For class imbalance

        # Calculate feature dimensions
        encoder_feature_dim = (
            config.encoder.num_layers *
            (2 if config.encoder.bidirectional else 1) *
            config.encoder.hidden_dim
        )
        decoder_output_dim = (
            (2 if config.decoder.bidirectional else 1) *
            config.decoder.hidden_dim
        )

        # Embedding layer
        self.embedding = EmbeddingLayer(config.embedding)

        # Encoder layer
        self.encoder = EncoderLayer(config.encoder)

        # Decoder layer
        self.decoder = DecoderLayer(encoder_feature_dim, config.decoder)

        # Reconstruction layer
        self.reconstruction = ReconstructionLayer(
            decoder_output_dim,
            config.embedding.vocab_size
        )

        # Dense layer
        self.dense = DenseLayer(encoder_feature_dim, config.dense)

        # Classification layer
        self.classifier = ClassificationLayer(
            config.dense.hidden_dim,
            num_classes
        )

        # Loss weight
        self.alpha = config.alpha

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Packet length sequence (batch, seq_len)
            lengths: Sequence lengths (batch,)

        Returns:
            class_logits: Classification logits (batch, num_classes)
            recon_logits: Reconstruction logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Encoder
        encoder_outputs, ze = self.encoder(embedded, lengths)

        # Decoder
        decoder_outputs, zd = self.decoder(ze, seq_len, lengths)

        # Reconstruction
        recon_logits = self.reconstruction(decoder_outputs)

        # Dense layer
        features = self.dense(ze, zd)

        # Classification
        class_logits = self.classifier(features)

        return class_logits, recon_logits

    def compute_loss(
        self,
        class_logits: torch.Tensor,
        recon_logits: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        L = LC + α * LR

        Args:
            class_logits: Classification logits (batch, num_classes)
            recon_logits: Reconstruction logits (batch, seq_len, vocab_size)
            targets: Class labels (batch,)
            x: Original input sequence (batch, seq_len)
            lengths: Sequence lengths (batch,)

        Returns:
            total_loss: Combined loss
            class_loss: Classification loss
            recon_loss: Reconstruction loss
        """
        # Classification loss (cross entropy with optional class weights)
        class_loss = F.cross_entropy(class_logits, targets, weight=self.class_weight)

        # Reconstruction loss (cross entropy per position)
        # Mask out padding positions
        batch_size, seq_len, vocab_size = recon_logits.size()
        mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Flatten for cross entropy
        recon_logits_flat = recon_logits.view(-1, vocab_size)
        x_flat = x.view(-1)
        mask_flat = mask.view(-1)

        # Compute loss only for non-padding positions
        recon_loss = F.cross_entropy(recon_logits_flat, x_flat, reduction='none')
        recon_loss = (recon_loss * mask_flat.float()).sum() / mask_flat.float().sum()

        # Combined loss
        total_loss = class_loss + self.alpha * recon_loss

        return total_loss, class_loss, recon_loss


class FSNetND(nn.Module):
    """FS-Net without Decoder (FS-ND variant).

    A simplified version that only uses encoder features for classification.
    From paper: used for ablation study.
    """

    def __init__(self, num_classes: int, config: FSNetConfig = None):
        super().__init__()
        if config is None:
            config = DEFAULT_FSNET_CONFIG

        self.config = config
        self.num_classes = num_classes

        # Calculate feature dimensions
        encoder_feature_dim = (
            config.encoder.num_layers *
            (2 if config.encoder.bidirectional else 1) *
            config.encoder.hidden_dim
        )

        # Embedding layer
        self.embedding = EmbeddingLayer(config.embedding)

        # Encoder layer
        self.encoder = EncoderLayer(config.encoder)

        # Dense layer (only uses ze)
        self.fc1 = nn.Linear(encoder_feature_dim, config.dense.hidden_dim)
        self.fc2 = nn.Linear(config.dense.hidden_dim, config.dense.hidden_dim)
        self.dropout = nn.Dropout(config.dense.dropout)

        # Classification layer
        self.classifier = ClassificationLayer(
            config.dense.hidden_dim,
            num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Packet length sequence (batch, seq_len)
            lengths: Sequence lengths (batch,)

        Returns:
            class_logits: Classification logits (batch, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)

        # Encoder
        _, ze = self.encoder(embedded, lengths)

        # Dense layer
        z = F.selu(self.fc1(ze))
        z = self.dropout(z)
        z = F.selu(self.fc2(z))
        z = self.dropout(z)

        # Classification
        class_logits = self.classifier(z)

        return class_logits


def create_fsnet(num_classes: int, **kwargs) -> FSNet:
    """Create FS-Net model with default configuration.

    Args:
        num_classes: Number of classification classes
        **kwargs: Override default config parameters

    Returns:
        FS-Net model
    """
    config = FSNetConfig()

    # Override with kwargs
    if 'embed_dim' in kwargs:
        config.embedding.embed_dim = kwargs['embed_dim']
        config.encoder.input_dim = kwargs['embed_dim']
    if 'hidden_dim' in kwargs:
        config.encoder.hidden_dim = kwargs['hidden_dim']
        config.decoder.hidden_dim = kwargs['hidden_dim']
    if 'num_layers' in kwargs:
        config.encoder.num_layers = kwargs['num_layers']
        config.decoder.num_layers = kwargs['num_layers']
    if 'dropout' in kwargs:
        config.encoder.dropout = kwargs['dropout']
        config.decoder.dropout = kwargs['dropout']
        config.dense.dropout = kwargs['dropout']
    if 'alpha' in kwargs:
        config.alpha = kwargs['alpha']

    class_weight = kwargs.get('class_weight', None)
    return FSNet(num_classes, config, class_weight)


def create_fsnet_nd(num_classes: int, **kwargs) -> FSNetND:
    """Create FS-Net-ND (no decoder) model.

    Args:
        num_classes: Number of classification classes
        **kwargs: Override default config parameters

    Returns:
        FS-Net-ND model
    """
    config = FSNetConfig()

    if 'embed_dim' in kwargs:
        config.embedding.embed_dim = kwargs['embed_dim']
        config.encoder.input_dim = kwargs['embed_dim']
    if 'hidden_dim' in kwargs:
        config.encoder.hidden_dim = kwargs['hidden_dim']
    if 'num_layers' in kwargs:
        config.encoder.num_layers = kwargs['num_layers']
    if 'dropout' in kwargs:
        config.encoder.dropout = kwargs['dropout']
        config.dense.dropout = kwargs['dropout']

    return FSNetND(num_classes, config)
