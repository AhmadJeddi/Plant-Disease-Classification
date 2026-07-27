"""
DCBAM: Dual Channel & Spatial Attention Module
----------------------------------------------
This module provides a channel attention (MLP on pooled features) and a
spatial attention (conv on concatenated max/avg across channel dimension).

Usage:
    from models.attention.dcbam import DCBAM
    m = DCBAM(channels=256, reduction=16)
    out = m(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import DCBAM_REDUCTION

class DCBAM(nn.Module):
    """
    Dual Channel & Spatial Attention Module.
    Args:
        channels: number of input channels.
        reduction: channel reduction for the MLP in channel-attention.
    """

    def __init__(self, channels: int, reduction: int = DCBAM_REDUCTION) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.conv1_c = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.conv2_c = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        # spatial conv: input 2 (max + avg), output 1
        self.conv_s = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            Tensor of same shape as x after applying channel and spatial attention.
        """
        
        # Channel attention (MLP over pooled descriptors)
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)

        def mlp(z: torch.Tensor) -> torch.Tensor:
            z = self.conv1_c(z)
            z = F.relu(z, inplace=True)
            z = self.conv2_c(z)
            return z

        # Fuse average and max pooled channel descriptors
        ca = torch.sigmoid(mlp(max_pool) + mlp(avg_pool))
        x_ca = x * ca

        # Spatial attention using channel-wise max and average
        max_c = torch.amax(x_ca, dim=1, keepdim=True)
        avg_c = torch.mean(x_ca, dim=1, keepdim=True)
        sa = torch.cat([max_c, avg_c], dim=1)
        sa = torch.sigmoid(self.conv_s(sa))

        return x_ca * sa
