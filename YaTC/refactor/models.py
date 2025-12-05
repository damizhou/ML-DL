"""
YaTC Model Definitions

This module implements the YaTC (Yet Another Traffic Classifier) model architecture
as described in the AAAI 2023 paper:
"Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer"

Models:
- MAE_YaTC: Masked Autoencoder for pre-training
- TraFormer_YaTC: Traffic Transformer for fine-tuning/classification

All architectures and parameters are consistent with the paper and original implementation.
Compatible with PyTorch 2.9.
"""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DEFAULT_MAE_CONFIG,
    DEFAULT_TRAFORMER_CONFIG,
    EncoderConfig,
    DecoderConfig,
)


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode

    Returns:
        Tensor with paths randomly dropped during training
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) block.

    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention.

    Implements scaled dot-product attention with multiple heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block.

    Architecture:
    - LayerNorm -> Multi-Head Attention -> DropPath
    - LayerNorm -> MLP -> DropPath
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding for MFR (Multi-level Flow Representation).

    The MFR is a 40x40 grayscale image representing 5 packets.
    Each packet is represented as 8 rows × 40 columns.

    This module embeds each packet separately:
    - Per-packet image size: (8, 40) = (img_size/5, img_size)
    - Patch size: 2x2
    - Patches per packet: (8/2) × (40/2) = 4 × 20 = 80
    - Total patches for full MFR: 80 × 5 = 400
    """

    def __init__(
        self,
        img_size: int = 40,
        patch_size: int = 2,
        in_chans: int = 1,
        embed_dim: int = 192
    ):
        super().__init__()
        # Per-packet image size: (height, width) = (img_size/5, img_size)
        self.img_size = (int(img_size / 5), img_size)  # (8, 40)
        self.patch_size = (patch_size, patch_size)  # (2, 2)

        # Number of patches per packet
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * \
                          (self.img_size[1] // self.patch_size[1])  # 4 × 20 = 80

        # Convolution for patch embedding
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where H=8, W=40 (single packet)

        Returns:
            Embedded patches of shape (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int], cls_token: bool = False) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Tuple of (height, width) grid dimensions
        cls_token: Whether to include position for class token

    Returns:
        Positional embeddings of shape (grid_h * grid_w, embed_dim) or
        (1 + grid_h * grid_w, embed_dim) if cls_token is True
    """
    grid_h, grid_w = grid_size
    grid_h_pos = torch.arange(grid_h, dtype=torch.float32)
    grid_w_pos = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h_pos, grid_w_pos, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_h, grid_w)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings from grid.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid: Grid tensor of shape (2, 1, H, W)

    Returns:
        Positional embeddings of shape (H * W, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position
        pos: Position tensor of shape (M,) or flattened grid

    Returns:
        Positional embeddings of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class MAE_YaTC(nn.Module):
    """Masked Autoencoder for YaTC Pre-training.

    Architecture (from paper Table 1):
    - Encoder: 4 Transformer blocks, 192-dim embedding, 16 attention heads
    - Decoder: 2 Transformer blocks, 128-dim embedding, 16 attention heads
    - Patch size: 2x2
    - Input: 40x40 grayscale MFR image (5 packets)
    - Total patches: 400 (80 per packet × 5 packets)
    - Mask ratio: 0.9 (90% of patches masked)
    """

    def __init__(
        self,
        img_size: int = 40,
        patch_size: int = 2,
        in_chans: int = 1,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 16,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.,
        norm_layer: nn.Module = None,
        norm_pix_loss: bool = False
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding (per packet)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # Total patches = patches_per_packet × num_packets = 80 × 5 = 400
        self.num_patches = self.patch_embed.num_patches * 5  # 400

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False
        )

        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Prediction head: predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_chans,
            bias=True
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with proper initialization schemes."""
        # Initialize positional embeddings with sinusoidal encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (int(40 / 5 / 2), int(40 / 2)),  # (4, 20) per packet
            cls_token=True
        )
        # Expand for 5 packets
        pos_embed_full = torch.zeros(1, self.num_patches + 1, self.pos_embed.shape[-1])
        pos_embed_full[0, 0] = pos_embed[0]  # cls token
        for i in range(5):
            pos_embed_full[0, 1 + i * 80: 1 + (i + 1) * 80] = pos_embed[1:]
        self.pos_embed.data.copy_(pos_embed_full)

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            (int(40 / 5 / 2), int(40 / 2)),
            cls_token=True
        )
        decoder_pos_embed_full = torch.zeros(1, self.num_patches + 1, self.decoder_pos_embed.shape[-1])
        decoder_pos_embed_full[0, 0] = decoder_pos_embed[0]
        for i in range(5):
            decoder_pos_embed_full[0, 1 + i * 80: 1 + (i + 1) * 80] = decoder_pos_embed[1:]
        self.decoder_pos_embed.data.copy_(decoder_pos_embed_full)

        # Initialize patch embedding projection
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights for Linear and LayerNorm layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        Args:
            imgs: Input images of shape (B, 1, 40, 40)

        Returns:
            Patches of shape (B, num_patches, patch_size^2 * in_chans)
        """
        p = self.patch_embed.patch_size[0]  # 2
        h = int(imgs.shape[2] / 5 / p)  # 4
        w = int(imgs.shape[3] / p)  # 20

        patches_list = []
        for i in range(5):
            packet = imgs[:, :, i * 8:(i + 1) * 8, :]
            x = packet.reshape(imgs.shape[0], 1, h, p, w, p)
            x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, 1)
            x = x.reshape(imgs.shape[0], h * w, p * p * 1)
            patches_list.append(x)

        patches = torch.cat(patches_list, dim=1)  # (B, 400, 4)
        return patches

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images.

        Args:
            x: Patches of shape (B, num_patches, patch_size^2 * in_chans)

        Returns:
            Images of shape (B, 1, 40, 40)
        """
        p = self.patch_embed.patch_size[0]  # 2
        h = int(40 / 5 / p)  # 4
        w = int(40 / p)  # 20

        imgs_list = []
        for i in range(5):
            packet_patches = x[:, i * 80:(i + 1) * 80, :]
            packet_patches = packet_patches.reshape(-1, h, w, p, p, 1)
            packet_patches = packet_patches.permute(0, 5, 1, 3, 2, 4)  # (B, 1, h, p, w, p)
            packet = packet_patches.reshape(-1, 1, h * p, w * p)  # (B, 1, 8, 40)
            imgs_list.append(packet)

        imgs = torch.cat(imgs_list, dim=2)  # (B, 1, 40, 40)
        return imgs

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform random masking.

        Args:
            x: Input tensor of shape (B, N, D)
            mask_ratio: Ratio of patches to mask

        Returns:
            x_masked: Masked input
            mask: Binary mask (1 = keep, 0 = masked)
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder with masking.

        Args:
            x: Input images of shape (B, 1, 40, 40)
            mask_ratio: Ratio of patches to mask

        Returns:
            latent: Encoded representations
            mask: Binary mask
            ids_restore: Indices to restore order
        """
        # Embed each packet separately and concatenate
        embeddings = []
        for i in range(5):
            packet = x[:, :, i * 8:(i + 1) * 8, :]
            emb = self.patch_embed(packet)  # (B, 80, embed_dim)
            embeddings.append(emb)
        x = torch.cat(embeddings, dim=1)  # (B, 400, embed_dim)

        # Add positional embeddings (without cls token)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepend cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            x: Encoded representations
            ids_restore: Indices to restore original order

        Returns:
            Decoded predictions for all patches
        """
        # Embed to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixel values
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            imgs: Original images
            pred: Predicted patches
            mask: Binary mask (1 = masked, 0 = visible)

        Returns:
            Mean squared error loss on masked patches
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            imgs: Input images of shape (B, 1, 40, 40)
            mask_ratio: Ratio of patches to mask (default: 0.9)

        Returns:
            loss: Reconstruction loss
            pred: Predicted patches
            mask: Binary mask
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class TraFormer_YaTC(nn.Module):
    """Traffic Transformer for YaTC Fine-tuning.

    Architecture (from paper Table 1):
    - Encoder: 4 Transformer blocks, 192-dim embedding, 16 attention heads
    - Patch size: 2x2
    - Input: 40x40 grayscale MFR image (5 packets)
    - Total patches: 400 (80 per packet × 5 packets)

    Inherits encoder architecture from MAE pre-training.
    Adds classification head for traffic classification.
    """

    def __init__(
        self,
        img_size: int = 40,
        patch_size: int = 2,
        in_chans: int = 1,
        num_classes: int = 7,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = None
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding (per packet)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # Total patches = patches_per_packet × num_packets = 80 × 5 = 400
        self.num_patches = self.patch_embed.num_patches * 5  # 400

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with proper initialization schemes."""
        # Initialize positional embeddings with sinusoidal encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (int(40 / 5 / 2), int(40 / 2)),  # (4, 20) per packet
            cls_token=True
        )
        # Expand for 5 packets
        pos_embed_full = torch.zeros(1, self.num_patches + 1, self.pos_embed.shape[-1])
        pos_embed_full[0, 0] = pos_embed[0]  # cls token
        for i in range(5):
            pos_embed_full[0, 1 + i * 80: 1 + (i + 1) * 80] = pos_embed[1:]
        self.pos_embed.data.copy_(pos_embed_full)

        # Initialize patch embedding projection
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights for Linear and LayerNorm layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.

        Args:
            x: Input images of shape (B, 1, 40, 40)

        Returns:
            cls_token: Class token representation
        """
        # Embed each packet separately and concatenate
        embeddings = []
        for i in range(5):
            packet = x[:, :, i * 8:(i + 1) * 8, :]
            emb = self.patch_embed(packet)  # (B, 80, embed_dim)
            embeddings.append(emb)
        x = torch.cat(embeddings, dim=1)  # (B, 400, embed_dim)

        # Prepend cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Return cls token
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (B, 1, 40, 40)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


def mae_yatc(**kwargs) -> MAE_YaTC:
    """Create MAE_YaTC model with default configuration.

    Default architecture (from paper):
    - img_size: 40
    - patch_size: 2
    - embed_dim: 192
    - depth: 4
    - num_heads: 16
    - decoder_embed_dim: 128
    - decoder_depth: 2
    - decoder_num_heads: 16
    - mlp_ratio: 4
    """
    model = MAE_YaTC(
        img_size=40,
        patch_size=2,
        embed_dim=192,
        depth=4,
        num_heads=16,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def traformer_yatc(num_classes: int = 7, **kwargs) -> TraFormer_YaTC:
    """Create TraFormer_YaTC model with default configuration.

    Default architecture (from paper):
    - img_size: 40
    - patch_size: 2
    - embed_dim: 192
    - depth: 4
    - num_heads: 16
    - mlp_ratio: 4
    - qkv_bias: True

    Args:
        num_classes: Number of classification classes
    """
    model = TraFormer_YaTC(
        img_size=40,
        patch_size=2,
        in_chans=1,
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
