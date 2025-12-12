import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch embedding using Conv2d:
    Input:  [B, C, H, W]
    Output: [B, N, dim], where N = (H/P)*(W/P)
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                 # [B, dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, N, dim]
        return x