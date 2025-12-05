#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed2d(nn.Module):
    """
    2D Patch Embedding 模块（对应 YaTC 论文 3.2 中的 Embedding Module）：

    - 输入:  (B, 1, H, W)
    - 输出:  tokens (B, N, D), 其中 N = (H / P) * (W / P)

    使用 kernel_size = stride = patch_size 的 Conv2d，相当于对每个 P×P patch 做线性映射。
    """

    def __init__(
        self,
        img_size: int = 40,
        patch_size: int = 2,
        in_chans: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: (B, 1, H, W)
        x = self.proj(x)                     # (B, D, H_p, W_p)
        B, D, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, N, D)
        return x, H_p  # H_p == W_p == grid_size


class TransformerBlock(nn.Module):
    """
    标准 Transformer Encoder block：
        LN -> Multi-head Self-Attention -> 残差
        LN -> MLP(Linear-GELU-Linear)  -> 残差
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # 输入 (B, L, D)
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class YaTC(nn.Module):
    """
    YaTC Traffic Transformer（只包含 fine-tuning 阶段的分类结构）：

    参考论文：
        "Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer
         with Multi-Level Flow Representation" (YaTC)

    结构对应论文第 3.2 小节 "Traffic Transformer"：
        - Embedding Module:
            * 输入 MFR 矩阵 x ∈ R^{H×W}，H = W = 40
            * 划分为 P×P patch（论文中 P=2），得到 N = (H/P)^2 = 400 个 patch
            * 每个 patch 通过线性映射得到 D 维向量（D=192）
            * 加上 patch 级位置编码 E_pos
        - Packet-level Attention Module:
            * 将 patch 按行划分为 num_packets 个 packet，每个 packet 内部做 self-attention
            * 使用 L=4 层 Transformer，n=16 个 attention heads
        - Row Pooling (RP):
            * 对每一行的 patch 在列方向上求平均，得到 row patches（共 sqrt(N) = 20 个）
        - Flow-level Attention Module:
            * 在 row patches 上再进行一遍 self-attention（与 packet-level 共享参数）
        - Column Pooling (CP):
            * 在行方向上求平均，得到完整 flow 的向量表示
        - 分类头：
            * LayerNorm + Linear -> logits (num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        *,
        img_size: int = 40,
        patch_size: int = 2,
        num_packets: int = 5,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须为正整数")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_packets = num_packets
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # ---- MFR 尺寸 / patch 网格大小检查 ----
        H = W = img_size
        assert H == W, "当前实现假定 MFR 为正方形矩阵 (H == W)"
        assert H % patch_size == 0, "img_size 必须能被 patch_size 整除"
        grid_size = H // patch_size

        assert H % num_packets == 0, "img_size 必须能被 num_packets 整除"
        rows_per_packet = H // num_packets
        assert rows_per_packet % patch_size == 0, "每个 packet 的行数必须能被 patch_size 整除"

        # patch 网格是 (H_p, W_p)，这里 H_p == W_p == grid_size
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        # 每个 packet 在 patch 网格中的行数
        self.patch_rows_per_packet = rows_per_packet // patch_size
        self.patches_per_packet = self.patch_rows_per_packet * grid_size

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbed2d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
        )

        # Patch 级位置编码（长度 N）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # ---- Traffic Encoder blocks（packet / flow 共用参数）----
        blocks = []
        for _ in range(depth):
            blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    # ------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------
    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------
    # 内部：依次通过共享的 Transformer blocks
    # ------------------------------------------------------------
    def _run_shared_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        for blk in self.blocks:
            x = blk(x)
        return x

    # ------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ):
        """
        参数
        ----
        x:
            MFR 矩阵，形状为 (B, H, W) 或 (B, 1, H, W)，
            H = W = img_size（默认 40）。
            一般为 0~255 的字节或归一化到 [0,1] 的浮点值。
        return_features:
            若为 True，则同时返回最后的 flow 表示向量 (B, D)。

        返回
        ----
        logits:
            (B, num_classes) 的分类 logits。
        或
        logits, feat:
            feat: (B, embed_dim) 的 flow 表示。
        """
        if x.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"期望输入形状为 (B, H, W) 或 (B, 1, H, W)，当前为 {tuple(x.shape)}")

        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"MFR 尺寸 {(H, W)} 与配置 img_size={self.img_size} 不一致"
            )

        # 转成 float（位置编码默认 float32）
        x = x.to(self.pos_embed.dtype)

        # ---- Patch Embedding ----
        tokens, H_p = self.patch_embed(x)  # tokens: (B, N, D)
        assert H_p == self.grid_size

        # 加 patch-level 位置编码
        tokens = tokens + self.pos_embed   # (B, N, D)

        # 重新 reshape 成 patch 网格：(B, H_p, W_p, D)
        W_p = self.grid_size
        x_grid = tokens.view(B, H_p, W_p, self.embed_dim)

        # ============================================================
        # Packet-level Attention
        #   将 patch 网格按行划分为 num_packets 个 packet，每个 packet 内自注意力
        # ============================================================
        M = self.num_packets
        prpp = self.patch_rows_per_packet    # patch rows per packet

        # (B, H_p, W_p, D) -> (B, M, prpp, W_p, D)
        x_packets = x_grid.view(B, M, prpp, W_p, self.embed_dim)
        # 合并 batch 和 packet 维度，得到每个 packet 的 patch 序列：
        x_packets = x_packets.view(B * M, self.patches_per_packet, self.embed_dim)  # (B*M, Lp, D)

        # 共享的 Traffic Encoder：packet-level
        x_packets = self._run_shared_encoder(x_packets)  # (B*M, Lp, D)

        # 还原回 patch 网格 (B, H_p, W_p, D)
        x_packets = x_packets.view(B, M, prpp, W_p, self.embed_dim)
        x_packets = x_packets.view(B, H_p, W_p, self.embed_dim)

        # ============================================================
        # Row Pooling (RP)：对每一行的 patch 在列方向做平均，得到 row patches
        #   结果形状 (B, H_p, D)，共 sqrt(N) = H_p 行
        # ============================================================
        x_rows = x_packets.mean(dim=2)   # (B, H_p, D)

        # ============================================================
        # Flow-level Attention：在 row patches 上做自注意力（共用同一组 blocks）
        # ============================================================
        x_flow = self._run_shared_encoder(x_rows)  # (B, H_p, D)

        # ============================================================
        # Column Pooling (CP)：在行方向求平均，得到整个 flow 的表示
        # ============================================================
        feat = x_flow.mean(dim=1)  # (B, D)

        feat = self.norm(feat)
        logits = self.classifier(feat)

        if return_features:
            return logits, feat
        return logits


if __name__ == "__main__":
    # 最小自检，确保张量维度正确
    torch.manual_seed(0)

    num_classes = 100
    model = YaTC(
        num_classes=num_classes,
        img_size=40,
        patch_size=2,
        num_packets=5,
        embed_dim=192,
        depth=4,
        num_heads=16,
    )

    B = 4
    H = W = 40
    # 模拟一个 uint8 MFR 输入
    x = torch.randint(0, 256, (B, H, W), dtype=torch.uint8)

    logits, feat = model(x, return_features=True)
    print("logits.shape =", logits.shape)  # (4, 100)
    print("feat.shape   =", feat.shape)    # (4, 192)
