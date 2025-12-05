"""
YaTC Unit Tests

This module contains unit tests for verifying the YaTC model implementation
matches the paper specifications.

Tests verify:
- Model architecture parameters
- Input/output shapes
- Patch embedding calculations
- Positional embedding dimensions
- Forward pass correctness
"""

import unittest
import torch
import torch.nn as nn

from config import (
    DEFAULT_MAE_CONFIG,
    DEFAULT_TRAFORMER_CONFIG,
    DEFAULT_PRETRAIN_CONFIG,
    DEFAULT_FINETUNE_CONFIG,
    MFRConfig,
    PatchEmbedConfig,
    EncoderConfig,
    DecoderConfig,
)
from models import (
    PatchEmbed,
    MAE_YaTC,
    TraFormer_YaTC,
    mae_yatc,
    traformer_yatc,
    Block,
    Attention,
    Mlp,
)


class TestConfig(unittest.TestCase):
    """Test configuration values match paper specifications."""

    def test_mfr_config(self):
        """Test MFR configuration matches paper."""
        config = MFRConfig()
        self.assertEqual(config.num_packets, 5)
        self.assertEqual(config.bytes_per_packet, 320)
        self.assertEqual(config.header_bytes, 80)
        self.assertEqual(config.payload_bytes, 240)
        self.assertEqual(config.rows_per_packet, 8)
        self.assertEqual(config.img_size, 40)
        self.assertEqual(config.in_channels, 1)

    def test_patch_embed_config(self):
        """Test patch embedding configuration."""
        config = PatchEmbedConfig()
        self.assertEqual(config.img_size, (8, 40))
        self.assertEqual(config.patch_size, (2, 2))
        self.assertEqual(config.num_patches_per_packet, 80)
        self.assertEqual(config.num_packets, 5)
        self.assertEqual(config.total_patches, 400)

    def test_encoder_config(self):
        """Test encoder configuration matches paper Table 1."""
        config = EncoderConfig()
        self.assertEqual(config.embed_dim, 192)
        self.assertEqual(config.depth, 4)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.mlp_ratio, 4.0)
        self.assertTrue(config.qkv_bias)

    def test_decoder_config(self):
        """Test decoder configuration matches paper Table 1."""
        config = DecoderConfig()
        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.depth, 2)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.mlp_ratio, 4.0)

    def test_pretrain_config(self):
        """Test pre-training configuration matches paper."""
        config = DEFAULT_PRETRAIN_CONFIG
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.base_lr, 1e-3)
        self.assertEqual(config.weight_decay, 0.05)
        self.assertEqual(config.betas, (0.9, 0.95))
        self.assertEqual(config.total_steps, 150000)
        self.assertEqual(config.warmup_steps, 10000)
        self.assertEqual(config.mask_ratio, 0.9)

    def test_finetune_config(self):
        """Test fine-tuning configuration matches paper."""
        config = DEFAULT_FINETUNE_CONFIG
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.base_lr, 2e-3)
        self.assertEqual(config.weight_decay, 0.05)
        self.assertEqual(config.betas, (0.9, 0.999))
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.layer_decay, 0.65)
        self.assertEqual(config.smoothing, 0.1)


class TestPatchEmbed(unittest.TestCase):
    """Test patch embedding module."""

    def setUp(self):
        self.patch_embed = PatchEmbed(
            img_size=40,
            patch_size=2,
            in_chans=1,
            embed_dim=192
        )

    def test_img_size(self):
        """Test per-packet image size calculation."""
        # Per-packet: (img_size/5, img_size) = (8, 40)
        self.assertEqual(self.patch_embed.img_size, (8, 40))

    def test_patch_size(self):
        """Test patch size."""
        self.assertEqual(self.patch_embed.patch_size, (2, 2))

    def test_num_patches(self):
        """Test number of patches per packet."""
        # (8/2) × (40/2) = 4 × 20 = 80
        self.assertEqual(self.patch_embed.num_patches, 80)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        # Single packet input: (B, 1, 8, 40)
        x = torch.randn(2, 1, 8, 40)
        out = self.patch_embed(x)
        # Output: (B, num_patches, embed_dim) = (2, 80, 192)
        self.assertEqual(out.shape, (2, 80, 192))


class TestMAE_YaTC(unittest.TestCase):
    """Test Masked Autoencoder model."""

    def setUp(self):
        self.model = mae_yatc()

    def test_architecture_params(self):
        """Test model architecture matches paper Table 1."""
        # Encoder
        self.assertEqual(len(self.model.blocks), 4)  # depth=4
        self.assertEqual(self.model.blocks[0].attn.num_heads, 16)  # num_heads=16

        # Check embedding dimension
        self.assertEqual(self.model.pos_embed.shape[-1], 192)  # embed_dim=192

        # Decoder
        self.assertEqual(len(self.model.decoder_blocks), 2)  # decoder_depth=2
        self.assertEqual(self.model.decoder_blocks[0].attn.num_heads, 16)
        self.assertEqual(self.model.decoder_pos_embed.shape[-1], 128)  # decoder_embed_dim=128

    def test_num_patches(self):
        """Test total number of patches."""
        # 80 patches per packet × 5 packets = 400
        self.assertEqual(self.model.num_patches, 400)

    def test_positional_embedding_shape(self):
        """Test positional embedding dimensions."""
        # (1, num_patches + 1, embed_dim) = (1, 401, 192)
        self.assertEqual(self.model.pos_embed.shape, (1, 401, 192))
        # Decoder: (1, num_patches + 1, decoder_embed_dim) = (1, 401, 128)
        self.assertEqual(self.model.decoder_pos_embed.shape, (1, 401, 128))

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        x = torch.randn(2, 1, 40, 40)
        loss, pred, mask = self.model(x, mask_ratio=0.9)

        # Loss is scalar
        self.assertEqual(loss.dim(), 0)

        # Predictions: (B, num_patches, patch_size^2 * in_chans) = (2, 400, 4)
        self.assertEqual(pred.shape, (2, 400, 4))

        # Mask: (B, num_patches) = (2, 400)
        self.assertEqual(mask.shape, (2, 400))

    def test_masking_ratio(self):
        """Test that masking ratio is approximately correct."""
        x = torch.randn(10, 1, 40, 40)
        _, _, mask = self.model(x, mask_ratio=0.9)

        # Average mask ratio should be approximately 0.9
        actual_ratio = mask.float().mean().item()
        self.assertAlmostEqual(actual_ratio, 0.9, places=1)

    def test_patchify_unpatchify(self):
        """Test patchify and unpatchify are inverse operations."""
        x = torch.randn(2, 1, 40, 40)
        patches = self.model.patchify(x)
        reconstructed = self.model.unpatchify(patches)

        # Should be able to reconstruct original
        self.assertEqual(reconstructed.shape, x.shape)
        self.assertTrue(torch.allclose(x, reconstructed, atol=1e-6))


class TestTraFormer_YaTC(unittest.TestCase):
    """Test Traffic Transformer model."""

    def setUp(self):
        self.model = traformer_yatc(num_classes=7)

    def test_architecture_params(self):
        """Test model architecture matches paper Table 1."""
        # Encoder
        self.assertEqual(len(self.model.blocks), 4)  # depth=4
        self.assertEqual(self.model.blocks[0].attn.num_heads, 16)  # num_heads=16
        self.assertEqual(self.model.pos_embed.shape[-1], 192)  # embed_dim=192

    def test_num_patches(self):
        """Test total number of patches."""
        self.assertEqual(self.model.num_patches, 400)

    def test_positional_embedding_shape(self):
        """Test positional embedding dimensions."""
        # (1, num_patches + 1, embed_dim) = (1, 401, 192)
        self.assertEqual(self.model.pos_embed.shape, (1, 401, 192))

    def test_forward_shape(self):
        """Test forward pass output shape."""
        x = torch.randn(2, 1, 40, 40)
        out = self.model(x)

        # Output: (B, num_classes) = (2, 7)
        self.assertEqual(out.shape, (2, 7))

    def test_different_num_classes(self):
        """Test model works with different number of classes."""
        for num_classes in [7, 8, 10, 20]:
            model = traformer_yatc(num_classes=num_classes)
            x = torch.randn(2, 1, 40, 40)
            out = model(x)
            self.assertEqual(out.shape, (2, num_classes))


class TestBlock(unittest.TestCase):
    """Test Transformer block."""

    def setUp(self):
        self.block = Block(
            dim=192,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True
        )

    def test_forward_shape(self):
        """Test block preserves input shape."""
        x = torch.randn(2, 401, 192)
        out = self.block(x)
        self.assertEqual(out.shape, x.shape)

    def test_mlp_hidden_dim(self):
        """Test MLP hidden dimension."""
        # hidden_dim = dim * mlp_ratio = 192 * 4 = 768
        self.assertEqual(self.block.mlp.fc1.out_features, 768)


class TestAttention(unittest.TestCase):
    """Test attention mechanism."""

    def setUp(self):
        self.attn = Attention(
            dim=192,
            num_heads=16,
            qkv_bias=True
        )

    def test_head_dim(self):
        """Test head dimension calculation."""
        # head_dim = dim / num_heads = 192 / 16 = 12
        self.assertEqual(self.attn.head_dim, 12)

    def test_scale(self):
        """Test attention scale factor."""
        # scale = head_dim ** -0.5 = 12 ** -0.5
        expected_scale = 12 ** -0.5
        self.assertAlmostEqual(self.attn.scale, expected_scale, places=6)

    def test_forward_shape(self):
        """Test attention preserves input shape."""
        x = torch.randn(2, 401, 192)
        out = self.attn(x)
        self.assertEqual(out.shape, x.shape)


class TestWeightCompatibility(unittest.TestCase):
    """Test weight compatibility between MAE and TraFormer."""

    def test_encoder_weight_transfer(self):
        """Test that encoder weights can be transferred from MAE to TraFormer."""
        mae = mae_yatc()
        traformer = traformer_yatc(num_classes=7)

        # Get MAE encoder state dict (excluding decoder)
        mae_state = {
            k: v for k, v in mae.state_dict().items()
            if not k.startswith('decoder') and not k.startswith('mask_token')
        }

        # Get TraFormer state dict
        traformer_state = traformer.state_dict()

        # Check that shared keys have same shapes
        for key in mae_state:
            if key in traformer_state:
                self.assertEqual(
                    mae_state[key].shape,
                    traformer_state[key].shape,
                    f"Shape mismatch for {key}"
                )

    def test_shared_components(self):
        """Test that MAE and TraFormer share the same encoder architecture."""
        mae = mae_yatc()
        traformer = traformer_yatc(num_classes=7)

        # Compare patch embedding
        self.assertEqual(
            mae.patch_embed.proj.weight.shape,
            traformer.patch_embed.proj.weight.shape
        )

        # Compare cls token
        self.assertEqual(mae.cls_token.shape, traformer.cls_token.shape)

        # Compare positional embeddings
        self.assertEqual(mae.pos_embed.shape, traformer.pos_embed.shape)

        # Compare encoder blocks
        for i in range(4):
            mae_block = mae.blocks[i]
            traformer_block = traformer.blocks[i]

            # Attention
            self.assertEqual(
                mae_block.attn.qkv.weight.shape,
                traformer_block.attn.qkv.weight.shape
            )

            # MLP
            self.assertEqual(
                mae_block.mlp.fc1.weight.shape,
                traformer_block.mlp.fc1.weight.shape
            )


class TestModelParameterCount(unittest.TestCase):
    """Test model parameter counts."""

    def test_mae_parameter_count(self):
        """Test MAE model has reasonable parameter count."""
        model = mae_yatc()
        num_params = sum(p.numel() for p in model.parameters())

        # Should be in millions of parameters
        self.assertGreater(num_params, 1_000_000)
        self.assertLess(num_params, 50_000_000)

    def test_traformer_parameter_count(self):
        """Test TraFormer model has reasonable parameter count."""
        model = traformer_yatc(num_classes=7)
        num_params = sum(p.numel() for p in model.parameters())

        # TraFormer should have fewer parameters than MAE (no decoder)
        mae = mae_yatc()
        mae_params = sum(p.numel() for p in mae.parameters())

        self.assertLess(num_params, mae_params)


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow through models."""

    def test_mae_gradient_flow(self):
        """Test gradients flow through MAE."""
        model = mae_yatc()
        x = torch.randn(2, 1, 40, 40, requires_grad=True)

        loss, _, _ = model(x)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_traformer_gradient_flow(self):
        """Test gradients flow through TraFormer."""
        model = traformer_yatc(num_classes=7)
        x = torch.randn(2, 1, 40, 40, requires_grad=True)
        target = torch.tensor([0, 1])

        out = model(x)
        loss = nn.CrossEntropyLoss()(out, target)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == '__main__':
    unittest.main()
