"""
ClimaX-inspired Vision Transformer encoder.

Architecture mirrors the ClimaX design:
  - Each spatial grid cell becomes a token (H×W tokens per sample)
  - Each token is projected from the V climate variables at that cell
  - Learnable positional embeddings for the H×W spatial positions
  - Multi-layer Transformer (batch_first=True)
  - Global average pool → fixed-size embedding

We attempt to load pre-trained weights from microsoft/ClimaX on HuggingFace.
If the download fails or the weights are incompatible, the model is used
with random initialisation (still ViT-based, just not pre-trained).
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClimaXEncoder(nn.Module):
    def __init__(
        self,
        variables: List[str],
        patch_size: int = 5,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained_model_id: Optional[str] = "microsoft/ClimaX",
    ):
        super().__init__()
        self.variables  = variables
        self.n_vars     = len(variables)
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        n_tokens = patch_size * patch_size

        # Project each spatial token (V climate values) → embed_dim
        self.token_proj = nn.Linear(self.n_vars, embed_dim)

        # Learnable positional embedding for each spatial position
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN, same as ViT-based ClimaX
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

        if pretrained_model_id:
            self._try_load_pretrained(pretrained_model_id)

    def _try_load_pretrained(self, model_id: str) -> None:
        """
        Attempt to load compatible weights from a ClimaX checkpoint on HuggingFace.
        Only keys whose names and shapes match our model are loaded (strict=False).
        Any mismatch is silently skipped — random init covers the rest.
        """
        try:
            from huggingface_hub import hf_hub_download

            logger.info("Attempting to download ClimaX weights from %s …", model_id)
            # Filename comes from config; fall back to the coarser-resolution checkpoint
            import yaml, pathlib
            _cfg_path = pathlib.Path(__file__).parent.parent.parent / "configs" / "config.yaml"
            _ckpt_file = "5.625deg.ckpt"
            if _cfg_path.exists():
                _cfg = yaml.safe_load(_cfg_path.read_text())
                _ckpt_file = _cfg.get("model", {}).get("climax_checkpoint_file", _ckpt_file)

            ckpt_path = hf_hub_download(
                repo_id=model_id,
                filename=_ckpt_file,
                repo_type="model",
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Lightning checkpoints store weights under "state_dict"
            raw_sd = ckpt.get("state_dict", ckpt)

            our_sd = self.state_dict()
            compatible = {
                k: v
                for k, v in raw_sd.items()
                if k in our_sd and our_sd[k].shape == v.shape
            }
            self.load_state_dict(compatible, strict=False)
            logger.info(
                "Loaded %d / %d params from ClimaX pre-trained weights.",
                len(compatible), len(our_sd),
            )
        except Exception as e:
            logger.warning(
                "Could not load ClimaX pre-trained weights (%s). "
                "Using random initialisation.", e,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V, H, W) — normalised climate variable patches

        Returns:
            embedding: (B, embed_dim)
        """
        B, V, H, W = x.shape

        # Reshape to token sequence: (B, H*W, V)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, V)

        # Project variables → embed_dim
        x = self.token_proj(x)                    # (B, H*W, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed                    # (B, H*W, embed_dim)

        # Transformer
        x = self.transformer(x)                   # (B, H*W, embed_dim)
        x = self.norm(x)

        # Global average pool over spatial tokens → single embedding
        x = x.mean(dim=1)                         # (B, embed_dim)
        return x
