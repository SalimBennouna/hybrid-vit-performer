from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from performer_pytorch import FastAttention


class StandardAttention(nn.Module):
    """Standard multi-head self-attention (MHA)"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)

        out = attn @ v                                  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)      # [B, N, C]
        out = self.proj(out)
        return out


class PerformerAttention(nn.Module):
    """
    Multi-head Performer attention using performer-pytorch FastAttention.

    kernel_type:
        - 'relu'     -> generalized attention + ReLU kernel (Performer-ReLU)
        - 'softmax'  -> FAVOR+ positive random features (softmax approximation)
    """
    def __init__(self, dim, num_heads=8, m_features=256, kernel_type="relu"):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_type = kernel_type
        self.m_features = m_features

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        if kernel_type == "relu":
            self.fast_attn = FastAttention(
                dim_heads=self.head_dim,
                nb_features=m_features,
                generalized_attention=True,
                kernel_fn=nn.ReLU(),
                causal=False
            )
        elif kernel_type == "softmax":
            self.fast_attn = FastAttention(
                dim_heads=self.head_dim,
                nb_features=m_features,
                generalized_attention=False,
                causal=False
            )
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    def forward(self, x):
        """
        x: [B, N, C]  (batch, seq_len, dim)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 3, 2, 1, 4)

        q = qkv[:, :, 0]  # [B, H, N, D]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        out = self.fast_attn(q, k, v)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out