from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .attention import StandardAttention, PerformerAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        attention_type: Literal["standard", "performer"] = "standard",
        m_features: int = 256,
        kernel_type: str = "relu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if attention_type == "standard":
            self.attn = StandardAttention(dim, num_heads)
        else:
            self.attn = PerformerAttention(dim, num_heads, m_features, kernel_type)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x