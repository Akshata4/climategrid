"""
ClimateGridModel: frozen encoder + trainable risk head.

The encoder (ClimaX or CNN fallback) is frozen — only the risk head is trained.
This is intentional: ClimaX is a pre-trained foundation model; we adapt it via
the head without expensive full fine-tuning.
"""

import logging
from typing import List

import torch
import torch.nn as nn

from src.model.risk_head import RiskHead

logger = logging.getLogger(__name__)


def _build_encoder(config: dict, variables: List[str]) -> nn.Module:
    encoder_type = config["model"]["encoder"]
    embed_dim    = config["model"]["embed_dim"]
    patch_size   = config["dataset"]["patch_size"]

    if encoder_type == "climax":
        try:
            from src.model.climax_encoder import ClimaXEncoder
            return ClimaXEncoder(
                variables=variables,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=config["model"]["depth"],
                num_heads=config["model"]["num_heads"],
                mlp_ratio=config["model"]["mlp_ratio"],
                dropout=config["model"]["dropout"],
                pretrained_model_id=config["model"].get("climax_model_id"),
            )
        except Exception as e:
            logger.warning("ClimaX encoder failed (%s). Falling back to CNN.", e)

    from src.model.cnn_encoder import CNNEncoder
    logger.info("Using CNN encoder.")
    return CNNEncoder(variables=variables, patch_size=patch_size, embed_dim=embed_dim)


class ClimateGridModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        variables = config["variables"]

        self.encoder  = _build_encoder(config, variables)
        self.risk_head = RiskHead(
            embed_dim=config["model"]["embed_dim"],
            hidden_dims=config["model"]["risk_head_hidden"],
            dropout=config["model"]["dropout"],
        )

        # Freeze the encoder — only the risk head is trained
        for param in self.encoder.parameters():
            param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        logger.info(
            "Model ready. Trainable: %d / %d params (risk head only).",
            n_trainable, n_total,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_vars, patch_h, patch_w) normalised climate patch
        Returns:
            risk_scores: (B, 4) in [0, 1]
        """
        with torch.no_grad():
            embedding = self.encoder(x)
        return self.risk_head(embedding)

    def unfreeze_encoder(self) -> None:
        """Call this to enable full fine-tuning (future MLOps use)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen — full fine-tuning enabled.")
