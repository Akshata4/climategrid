"""
Inference wrapper.

Loads the trained checkpoint and exposes predict(city, scenario, year) → dict.
This is the single source of truth for model inference — used by both the
FastAPI /predict endpoint and the LLM advisor tools.

When MLflow is added later, swap _load_model() to load from the registry
instead of a local checkpoint file.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch

from src.data.cities import get_city, CITIES
from src.model.labels import RISK_NAMES

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model()
        self.norm_stats = self._load_norm_stats()

    def _load_model(self) -> torch.nn.Module:
        from src.model.climategrid import ClimateGridModel

        ckpt_path = Path(self.config["paths"]["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Run training first."
            )
        model = ClimateGridModel(self.config).to(self.device)
        ckpt  = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info("Model loaded from %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))
        return model

    def _load_norm_stats(self) -> Dict[str, Dict[str, float]]:
        stats_path = Path(self.config["paths"]["processed_data"]) / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalisation stats not found at {stats_path}. "
                "Run preprocessing first."
            )
        return json.loads(stats_path.read_text())

    def _build_feature_tensor(
        self,
        city: str,
        scenario: str,
        year: int,
    ) -> torch.Tensor:
        """
        Load the preprocessed feature tensor for (city, scenario, year).
        Falls back to interpolation between adjacent years if the exact
        year is missing (relevant for inference on future years beyond
        the processed range).
        """
        processed_dir = Path(self.config["paths"]["processed_data"])
        feat_path = processed_dir / scenario / city / f"{year}_features.npy"

        if feat_path.exists():
            arr = np.load(feat_path).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        # ── Interpolation fallback for years outside the processed range ──────
        # Find the nearest two processed years and linearly interpolate.
        city_dir = processed_dir / scenario / city
        available = sorted(
            int(p.stem.split("_")[0])
            for p in city_dir.glob("*_features.npy")
        )
        if not available:
            raise FileNotFoundError(
                f"No processed data for {city}/{scenario}. Run preprocessing."
            )

        if year <= available[0]:
            arr = np.load(city_dir / f"{available[0]}_features.npy")
        elif year >= available[-1]:
            arr = np.load(city_dir / f"{available[-1]}_features.npy")
        else:
            lo = max(y for y in available if y <= year)
            hi = min(y for y in available if y >= year)
            arr_lo = np.load(city_dir / f"{lo}_features.npy").astype(np.float32)
            arr_hi = np.load(city_dir / f"{hi}_features.npy").astype(np.float32)
            alpha = (year - lo) / (hi - lo)
            arr = (1 - alpha) * arr_lo + alpha * arr_hi

        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, city: str, scenario: str, year: int) -> Dict[str, float]:
        """
        Run inference for a single (city, scenario, year) query.

        Returns:
            dict with keys: heat_risk, flood_risk, wildfire_risk, drought_risk
            Each value is a float in [0, 1].
        """
        get_city(city)  # validates city name
        if scenario not in self.config["scenarios"]:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {self.config['scenarios']}")
        year_start = self.config["years"]["start"]
        year_end   = self.config["years"]["end"]
        if not (year_start <= year <= year_end):
            raise ValueError(f"Year {year} out of supported range [{year_start}, {year_end}].")

        x = self._build_feature_tensor(city, scenario, year)
        scores = self.model(x).squeeze(0).cpu().numpy()

        return {name: float(score) for name, score in zip(RISK_NAMES, scores)}


# ── Module-level singleton (shared across API requests) ──────────────────────

_predictor: Optional[Predictor] = None


def get_predictor(config: dict) -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(config)
    return _predictor
