"""
FS-Net Unit Tests

Tests for verifying FS-Net implementation matches paper specifications.
"""

import unittest
import torch
import torch.nn as nn

from config import (
    DEFAULT_FSNET_CONFIG,
    EmbeddingConfig,
    EncoderConfig,
    DecoderConfig,
    DenseConfig,
    FSNetConfig,
)
from models import (
    EmbeddingLayer,
    EncoderLayer,
    DecoderLayer,
    ReconstructionLayer,
    DenseLayer,
    ClassificationLayer,
    FSNet,
    FSNetND,
    create_fsnet,
    create_fsnet_nd,
)


class TestConfig(unittest.TestCase):
    """Test configuration values match paper."""

    def test_embedding_config(self):
        config = EmbeddingConfig()
        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.vocab_size, 1501)

    def test_encoder_config(self):
        config = EncoderConfig()
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.3)
        self.assertTrue(config.bidirectional)

    def test_decoder_config(self):
        config = DecoderConfig()
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.3)

    def test_fsnet_config(self):
        config = FSNetConfig()
        self.assertEqual(config.alpha, 1.0)
        self.assertEqual(config.max_seq_len, 100)


class TestEmbeddingLayer(unittest.TestCase):
    """Test embedding layer."""

    def setUp(self):
        self.layer = EmbeddingLayer()

    def test_output_shape(self):
        x = torch.randint(0, 1500, (2, 50))  # (batch, seq_len)
        out = self.layer(x)
        self.assertEqual(out.shape, (2, 50, 128))

    def test_padding(self):
        x = torch.zeros(2, 10, dtype=torch.long)  # All padding
        out = self.layer(x)
        # Padding index should have zero embedding
        self.assertTrue(torch.allclose(out[:, 0], out[:, 1]))


class TestEncoderLayer(unittest.TestCase):
    """Test encoder layer."""

    def setUp(self):
        self.layer = EncoderLayer()

    def test_output_shape(self):
        x = torch.randn(2, 50, 128)  # (batch, seq_len, embed_dim)
        lengths = torch.tensor([50, 30])
        outputs, ze = self.layer(x, lengths)

        # outputs: (batch, seq_len, hidden_dim * 2)
        self.assertEqual(outputs.shape, (2, 50, 256))

        # ze: (batch, num_layers * 2 * hidden_dim) = (2, 512)
        self.assertEqual(ze.shape, (2, 512))

    def test_bidirectional(self):
        self.assertTrue(self.layer.bidirectional)
        self.assertEqual(self.layer.num_directions, 2)


class TestDecoderLayer(unittest.TestCase):
    """Test decoder layer."""

    def setUp(self):
        # Encoder output dim: 2 layers * 2 directions * 128 = 512
        self.layer = DecoderLayer(encoder_feature_dim=512)

    def test_output_shape(self):
        ze = torch.randn(2, 512)  # Encoder feature
        lengths = torch.tensor([50, 30])
        outputs, zd = self.layer(ze, seq_len=50, lengths=lengths)

        # outputs: (batch, seq_len, hidden_dim * 2)
        self.assertEqual(outputs.shape, (2, 50, 256))

        # zd: (batch, num_layers * 2 * hidden_dim)
        self.assertEqual(zd.shape, (2, 512))


class TestReconstructionLayer(unittest.TestCase):
    """Test reconstruction layer."""

    def setUp(self):
        self.layer = ReconstructionLayer(input_dim=256, vocab_size=1501)

    def test_output_shape(self):
        x = torch.randn(2, 50, 256)
        out = self.layer(x)
        self.assertEqual(out.shape, (2, 50, 1501))


class TestDenseLayer(unittest.TestCase):
    """Test dense layer."""

    def setUp(self):
        self.layer = DenseLayer(feature_dim=512)

    def test_output_shape(self):
        ze = torch.randn(2, 512)
        zd = torch.randn(2, 512)
        out = self.layer(ze, zd)
        self.assertEqual(out.shape, (2, 256))


class TestFSNet(unittest.TestCase):
    """Test complete FS-Net model."""

    def setUp(self):
        self.model = create_fsnet(num_classes=18)

    def test_forward_shape(self):
        x = torch.randint(1, 1500, (2, 50))
        lengths = torch.tensor([50, 30])

        class_logits, recon_logits = self.model(x, lengths)

        # Class logits: (batch, num_classes)
        self.assertEqual(class_logits.shape, (2, 18))

        # Reconstruction logits: (batch, seq_len, vocab_size)
        self.assertEqual(recon_logits.shape, (2, 50, 1501))

    def test_loss_computation(self):
        x = torch.randint(1, 1500, (2, 50))
        lengths = torch.tensor([50, 30])
        targets = torch.tensor([0, 5])

        class_logits, recon_logits = self.model(x, lengths)
        total_loss, class_loss, recon_loss = self.model.compute_loss(
            class_logits, recon_logits, targets, x, lengths
        )

        # All losses should be scalar
        self.assertEqual(total_loss.dim(), 0)
        self.assertEqual(class_loss.dim(), 0)
        self.assertEqual(recon_loss.dim(), 0)

        # Total loss = class_loss + alpha * recon_loss
        expected = class_loss + self.model.alpha * recon_loss
        self.assertTrue(torch.allclose(total_loss, expected))

    def test_gradient_flow(self):
        x = torch.randint(1, 1500, (2, 50))
        lengths = torch.tensor([50, 30])
        targets = torch.tensor([0, 5])

        class_logits, recon_logits = self.model(x, lengths)
        total_loss, _, _ = self.model.compute_loss(
            class_logits, recon_logits, targets, x, lengths
        )

        total_loss.backward()

        # Check gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestFSNetND(unittest.TestCase):
    """Test FS-Net-ND (no decoder) variant."""

    def setUp(self):
        self.model = create_fsnet_nd(num_classes=18)

    def test_forward_shape(self):
        x = torch.randint(1, 1500, (2, 50))
        lengths = torch.tensor([50, 30])

        class_logits = self.model(x, lengths)

        # Only class logits
        self.assertEqual(class_logits.shape, (2, 18))

    def test_fewer_parameters(self):
        fsnet = create_fsnet(num_classes=18)
        fsnet_nd = create_fsnet_nd(num_classes=18)

        params_fsnet = sum(p.numel() for p in fsnet.parameters())
        params_fsnet_nd = sum(p.numel() for p in fsnet_nd.parameters())

        # FS-Net-ND should have fewer parameters
        self.assertLess(params_fsnet_nd, params_fsnet)


class TestModelConfiguration(unittest.TestCase):
    """Test model with different configurations."""

    def test_different_hidden_dim(self):
        for hidden_dim in [64, 128, 256]:
            model = create_fsnet(num_classes=10, hidden_dim=hidden_dim)
            x = torch.randint(1, 1500, (2, 30))
            lengths = torch.tensor([30, 20])
            class_logits, _ = model(x, lengths)
            self.assertEqual(class_logits.shape, (2, 10))

    def test_different_num_layers(self):
        for num_layers in [1, 2, 3]:
            model = create_fsnet(num_classes=10, num_layers=num_layers)
            x = torch.randint(1, 1500, (2, 30))
            lengths = torch.tensor([30, 20])
            class_logits, _ = model(x, lengths)
            self.assertEqual(class_logits.shape, (2, 10))

    def test_different_num_classes(self):
        for num_classes in [5, 10, 18, 50]:
            model = create_fsnet(num_classes=num_classes)
            x = torch.randint(1, 1500, (2, 30))
            lengths = torch.tensor([30, 20])
            class_logits, _ = model(x, lengths)
            self.assertEqual(class_logits.shape, (2, num_classes))


class TestPaperHyperparameters(unittest.TestCase):
    """Test that default hyperparameters match paper."""

    def test_embedding_dim(self):
        """Paper: embedding dimension = 128"""
        config = EmbeddingConfig()
        self.assertEqual(config.embed_dim, 128)

    def test_hidden_dim(self):
        """Paper: hidden state dimension = 128"""
        config = EncoderConfig()
        self.assertEqual(config.hidden_dim, 128)

    def test_num_layers(self):
        """Paper: 2-layer bi-GRU"""
        config = EncoderConfig()
        self.assertEqual(config.num_layers, 2)

    def test_dropout(self):
        """Paper: dropout = 0.3"""
        config = EncoderConfig()
        self.assertEqual(config.dropout, 0.3)

    def test_alpha(self):
        """Paper: α = 1"""
        config = FSNetConfig()
        self.assertEqual(config.alpha, 1.0)


if __name__ == '__main__':
    unittest.main()
