"""
Synthetic risk label generation.

These are documented proxies — not real risk indices.
Each formula maps raw CMIP6 variable patches to a scalar risk score in [0, 1].

Variable units:
  tas:     Kelvin
  pr:      kg m⁻² s⁻¹  (multiply × 86400 to get mm/day)
  hurs:    %
  sfcWind: m/s
"""

import numpy as np
from typing import Dict


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def heat_risk(tas: np.ndarray, cfg: dict) -> float:
    """
    Proxy: logistic function of mean temperature above a baseline.
    risk → 0.5 at baseline_celsius, → 1 for very hot, → 0 for cold.
    """
    tas_c = tas - 273.15
    score = _sigmoid((tas_c.mean() - cfg["baseline_celsius"]) / cfg["scale"])
    return float(np.clip(score, 0.0, 1.0))


def flood_risk(pr: np.ndarray, cfg: dict) -> float:
    """
    Proxy: mean + 2×std precipitation, normalised.
    High mean AND high variability → high flood risk.
    """
    pr_mm = pr * 86400.0
    proxy = (pr_mm.mean() + 2.0 * pr_mm.std()) / cfg["normalization_mm_day"]
    return float(np.clip(proxy, 0.0, 1.0))


def wildfire_risk(tas: np.ndarray, pr: np.ndarray, cfg: dict) -> float:
    """
    Proxy: weighted combination of heat component and dryness component.
    """
    tas_c = tas - 273.15
    heat_comp = np.clip(
        (tas_c.mean() - cfg["heat_baseline_celsius"]) / cfg["heat_range_celsius"],
        0.0, 1.0,
    )
    pr_mm = pr * 86400.0
    dry_comp = np.clip(
        1.0 - pr_mm.mean() / cfg["precip_dry_threshold_mm_day"],
        0.0, 1.0,
    )
    score = cfg["heat_weight"] * heat_comp + cfg["dry_weight"] * dry_comp
    return float(np.clip(score, 0.0, 1.0))


def drought_risk(tas: np.ndarray, pr: np.ndarray, cfg: dict) -> float:
    """
    Proxy: product of normalised heat and normalised water deficit.
    Requires both high temperature AND low precipitation to score high.
    """
    tas_c = tas - 273.15
    heat_norm = np.clip(
        (tas_c.mean() - cfg["temp_baseline_celsius"]) / cfg["temp_range_celsius"],
        0.0, 1.0,
    )
    pr_mm = pr * 86400.0
    water_deficit = np.clip(
        1.0 - pr_mm.mean() / cfg["precip_normalization_mm_day"],
        0.0, 1.0,
    )
    score = heat_norm * water_deficit
    return float(np.clip(score, 0.0, 1.0))


def compute_labels(raw_patches: Dict[str, np.ndarray], label_cfg: dict) -> np.ndarray:
    """
    Compute all four risk scores for a single sample.

    Args:
        raw_patches: dict mapping variable name → numpy array (patch_h, patch_w)
        label_cfg: labels section from config.yaml

    Returns:
        np.ndarray of shape (4,): [heat, flood, wildfire, drought]
    """
    tas     = raw_patches["tas"]
    pr      = raw_patches["pr"]

    scores = np.array([
        heat_risk(tas, label_cfg["heat"]),
        flood_risk(pr, label_cfg["flood"]),
        wildfire_risk(tas, pr, label_cfg["wildfire"]),
        drought_risk(tas, pr, label_cfg["drought"]),
    ], dtype=np.float32)

    return scores


RISK_NAMES = ["heat_risk", "flood_risk", "wildfire_risk", "drought_risk"]
