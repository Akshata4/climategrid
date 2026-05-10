"""Tests for the data pipeline."""

import numpy as np
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_city_lookup():
    from src.data.cities import get_city, CITY_NAMES, CITIES
    for name in CITY_NAMES:
        c = get_city(name)
        assert "lat" in c and "lon" in c
        assert -90 <= c["lat"] <= 90
        assert -180 <= c["lon"] <= 180


def test_city_lookup_invalid():
    from src.data.cities import get_city
    with pytest.raises(ValueError, match="Unknown city"):
        get_city("Atlantis")


def test_synthetic_data_generation(tmp_path, config):
    from src.data.cities import CITIES
    from src.data.download import _generate_synthetic_data

    config["paths"]["raw_data"] = str(tmp_path)
    variables = config["variables"]
    scenarios = config["scenarios"]
    years     = [2030, 2035]
    patch_size = config["dataset"]["patch_size"]

    _generate_synthetic_data(CITIES, variables, scenarios, years, patch_size, tmp_path)

    for scenario in scenarios:
        for variable in variables:
            for city in CITIES:
                for year in years:
                    path = tmp_path / scenario / variable / f"{city}_{year}.npy"
                    assert path.exists(), f"Missing: {path}"
                    arr = np.load(path)
                    assert arr.shape == (patch_size, patch_size)
                    assert arr.dtype == np.float32


def test_label_computation(config):
    from src.model.labels import compute_labels

    patch_size = config["dataset"]["patch_size"]
    raw_patches = {
        "tas":     np.full((patch_size, patch_size), 308.0, dtype=np.float32),   # ~35°C, hot
        "pr":      np.full((patch_size, patch_size), 1e-5, dtype=np.float32),    # low rain
        "hurs":    np.full((patch_size, patch_size), 30.0, dtype=np.float32),
        "sfcWind": np.full((patch_size, patch_size), 3.5, dtype=np.float32),
    }
    labels = compute_labels(raw_patches, config["labels"])

    assert labels.shape == (4,)
    assert np.all((labels >= 0) & (labels <= 1)), f"Labels out of range: {labels}"
    # Hot + dry city → heat and drought should be elevated
    assert labels[0] > 0.5, f"Expected heat_risk > 0.5 for 35°C, got {labels[0]}"
