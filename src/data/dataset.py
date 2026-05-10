"""
PyTorch Dataset for ClimateGrid.

Each sample is one (city, scenario, year) triple.
  features: Tensor (n_vars, patch_h, patch_w)  normalised climate variables
  labels:   Tensor (4,)                        synthetic risk scores in [0, 1]
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class ClimateGridDataset(Dataset):
    def __init__(self, processed_dir: Path, scenarios: List[str], cities: List[str], years: List[int]):
        self.samples: List[Tuple[Path, Path]] = []

        for scenario in scenarios:
            for city in cities:
                for year in years:
                    base = processed_dir / scenario / city
                    feat_path  = base / f"{year}_features.npy"
                    label_path = base / f"{year}_labels.npy"
                    if feat_path.exists() and label_path.exists():
                        self.samples.append((feat_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No processed samples found in {processed_dir}. "
                "Run the preprocessing step first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_path, label_path = self.samples[idx]
        features = torch.from_numpy(np.load(feat_path))
        labels   = torch.from_numpy(np.load(label_path))
        return features, labels


def build_datasets(config: dict) -> Tuple[ClimateGridDataset, ClimateGridDataset]:
    """Return (train_dataset, val_dataset) split by year."""
    from src.data.cities import CITY_NAMES

    processed_dir = Path(config["paths"]["processed_data"])
    scenarios     = config["scenarios"]
    cities        = CITY_NAMES
    all_years     = list(range(config["years"]["start"], config["years"]["end"] + 1))
    val_frac      = config["training"]["val_split"]

    # Chronological split: last val_frac of years → validation
    split_idx  = int(len(all_years) * (1 - val_frac))
    train_years = all_years[:split_idx]
    val_years   = all_years[split_idx:]

    train_ds = ClimateGridDataset(processed_dir, scenarios, cities, train_years)
    val_ds   = ClimateGridDataset(processed_dir, scenarios, cities, val_years)

    return train_ds, val_ds
