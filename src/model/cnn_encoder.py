"""
Fallback CNN encoder with the same interface as ClimaXEncoder.
Used when ClimaX cannot be initialised (e.g. import errors, HF outage).
"""

from typing import List

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(
        self,
        variables: List[str],
        patch_size: int = 5,
        embed_dim: int = 512,
        **kwargs,  # absorb extra ClimaX-specific kwargs
    ):
        super().__init__()
        n_vars = len(variables)
        self.embed_dim = embed_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(n_vars, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),   # → (B, 256, 1, 1)
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V, H, W)
        Returns:
            embedding: (B, embed_dim)
        """
        x = self.backbone(x)          # (B, 256, 1, 1)
        x = x.flatten(1)              # (B, 256)
        x = self.proj(x)              # (B, embed_dim)
        return x
