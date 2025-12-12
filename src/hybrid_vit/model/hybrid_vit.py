from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import TrainConfig
from .patch_embed import PatchEmbedding
from .blocks import TransformerBlock


class HybridViT(nn.Module):
    """
    Configurable Hybrid Vision Transformer.

    architecture:
        - "all_standard"
        - "all_performer"
        - "intertwined"
        - "performer_first"
        - "standard_first"
    """
    def __init__(self, cfg: TrainConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        dim = cfg.dim

        self.patch_embed = PatchEmbedding(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_channels=in_channels,
            dim=dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        blocks: List[TransformerBlock] = []
        for i in range(cfg.depth):
            if cfg.architecture == "all_standard":
                attn_type = "standard"
            elif cfg.architecture == "all_performer":
                attn_type = "performer"
            elif cfg.architecture == "intertwined":
                attn_type = "performer" if i % 2 == 0 else "standard"
            elif cfg.architecture == "performer_first":
                attn_type = "performer" if i < cfg.depth // 2 else "standard"
            elif cfg.architecture == "standard_first":
                attn_type = "standard" if i < cfg.depth // 2 else "performer"
            else:
                raise ValueError(f"Unknown architecture: {cfg.architecture}")

            blocks.append(
                TransformerBlock(
                    dim=dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attention_type=attn_type,
                    m_features=cfg.m_features,
                    kernel_type=cfg.kernel_type
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)            # [B, N, dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, 1+N, dim]
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                  # [B, dim]
        logits = self.head(cls_out)
        return logits