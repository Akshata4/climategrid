"""
Load raw city-level climate patches, normalise each variable, compute
per-variable statistics across the training set, and save processed
tensors to data/processed/.

Output per (city, scenario, year):
  - features.npy  shape (n_vars, patch_h, patch_w)  normalised climate patch
  - labels.npy    shape (4,)                         synthetic risk scores [0,1]
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _load_raw_patch(raw_dir: Path, scenario: str, variable: str, city: str, year: int) -> np.ndarray:
    path = raw_dir / scenario / variable / f"{city}_{year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return np.load(path).astype(np.float32)


def _compute_normalisation_stats(
    raw_dir: Path,
    variables: List[str],
    scenarios: List[str],
    cities: Dict[str, Dict],
    years: List[int],
) -> Dict[str, Dict[str, float]]:
    """Compute per-variable mean and std across all cities/scenarios/years."""
    stats: Dict[str, Dict[str, float]] = {}
    for variable in variables:
        all_values: List[float] = []
        for scenario in scenarios:
            for city in cities:
                for year in years:
                    try:
                        patch = _load_raw_patch(raw_dir, scenario, variable, city, year)
                        all_values.extend(patch.flatten().tolist())
                    except FileNotFoundError:
                        continue
        arr = np.array(all_values, dtype=np.float32)
        stats[variable] = {"mean": float(arr.mean()), "std": float(max(arr.std(), 1e-8))}
        logger.info("  %s: mean=%.4f std=%.4f", variable, stats[variable]["mean"], stats[variable]["std"])
    return stats


def run_preprocessing(config: dict) -> None:
    from src.data.cities import CITIES
    from src.model.labels import compute_labels

    raw_dir       = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    variables = config["variables"]
    scenarios = config["scenarios"]
    years     = list(range(config["years"]["start"], config["years"]["end"] + 1))
    patch_h = patch_w = config["dataset"]["patch_size"]
    n_vars = len(variables)

    logger.info("Computing normalisation statistics …")
    stats = _compute_normalisation_stats(raw_dir, variables, scenarios, CITIES, years)

    # Save normalisation stats for use at inference time
    import json
    stats_path = processed_dir / "norm_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Normalisation stats saved to %s", stats_path)

    logger.info("Building processed tensors …")
    count = 0
    for scenario in scenarios:
        for city in CITIES:
            for year in years:
                out_dir = processed_dir / scenario / city
                out_dir.mkdir(parents=True, exist_ok=True)
                feat_path  = out_dir / f"{year}_features.npy"
                label_path = out_dir / f"{year}_labels.npy"

                if feat_path.exists() and label_path.exists():
                    continue  # already processed

                # Build feature tensor: (n_vars, patch_h, patch_w)
                try:
                    patches = []
                    raw_patches: Dict[str, np.ndarray] = {}
                    for variable in variables:
                        patch = _load_raw_patch(raw_dir, scenario, variable, city, year)
                        raw_patches[variable] = patch
                        mean = stats[variable]["mean"]
                        std  = stats[variable]["std"]
                        patches.append((patch - mean) / std)
                    features = np.stack(patches, axis=0).astype(np.float32)  # (n_vars, H, W)
                    assert features.shape == (n_vars, patch_h, patch_w), features.shape
                except FileNotFoundError as e:
                    logger.warning("Skipping %s/%s/%d: %s", scenario, city, year, e)
                    continue

                # Compute synthetic risk labels from raw (unnormalised) variables
                labels = compute_labels(raw_patches, config["labels"]).astype(np.float32)

                np.save(feat_path, features)
                np.save(label_path, labels)
                count += 1

    logger.info("Preprocessing complete. %d samples written to %s", count, processed_dir)
