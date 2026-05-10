"""
Risk scoring head: maps encoder embedding → 4 risk scores in [0, 1].
"""

from typing import List

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 4))   # 4 risk scores
        layers.append(nn.Sigmoid())            # outputs in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, embed_dim)
        Returns:
            risk_scores: (B, 4)  in [0, 1]
        """
        return self.net(embedding)
