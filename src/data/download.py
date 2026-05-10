"""
Download climate data from ClimateSet (climateset/causalpaca) on HuggingFace,
extracting only the grid cells near the 6 target cities.

If `use_synthetic=True` in config, generates synthetic climate tensors instead
(useful for testing the full pipeline without a large download).
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _nearest_grid_index(lats: np.ndarray, lons: np.ndarray, city_lat: float, city_lon: float) -> Tuple[int, int]:
    lat_idx = int(np.argmin(np.abs(lats - city_lat)))
    lon_idx = int(np.argmin(np.abs(lons - city_lon)))
    return lat_idx, lon_idx


def _extract_patch(data: np.ndarray, lat_idx: int, lon_idx: int, half: int) -> np.ndarray:
    """Extract a (2*half+1) x (2*half+1) patch centred on (lat_idx, lon_idx)."""
    lat_start = max(0, lat_idx - half)
    lat_end   = min(data.shape[-2], lat_idx + half + 1)
    lon_start = max(0, lon_idx - half)
    lon_end   = min(data.shape[-1], lon_idx + half + 1)
    patch = data[..., lat_start:lat_end, lon_start:lon_end]
    # Pad to fixed size if near the boundary
    pad_lat = (half - (lat_idx - lat_start), half - (lat_end - lat_idx - 1))
    pad_lon = (half - (lon_idx - lon_start), half - (lon_end - lon_idx - 1))
    if any(p > 0 for p in pad_lat + pad_lon):
        pad_width = [(0, 0)] * (patch.ndim - 2) + [pad_lat, pad_lon]
        patch = np.pad(patch, pad_width, mode="edge")
    return patch


def _generate_synthetic_data(
    cities: Dict[str, Dict[str, float]],
    variables: List[str],
    scenarios: List[str],
    years: List[int],
    patch_size: int,
    raw_dir: Path,
) -> None:
    """
    Generate synthetic CMIP6-like climate tensors for testing.
    Saved as: raw_dir/{scenario}/{variable}/{city}_{year}.npy
    Shape per file: (patch_size, patch_size)
    """
    rng = np.random.default_rng(seed=42)

    # Approximate "climatological" baseline per variable per city
    # so the synthetic data has city-realistic magnitudes
    baselines = {
        "tas":     {"Houston": 302, "Phoenix": 308, "Miami": 301, "Chicago": 283, "Los Angeles": 292, "Seattle": 283},
        "pr":      {"Houston": 4e-5, "Phoenix": 5e-6, "Miami": 5e-5, "Chicago": 2e-5, "Los Angeles": 8e-6, "Seattle": 3e-5},
        "hurs":    {"Houston": 75, "Phoenix": 25, "Miami": 80, "Chicago": 65, "Los Angeles": 60, "Seattle": 75},
        "sfcWind": {"Houston": 4, "Phoenix": 3.5, "Miami": 4.5, "Chicago": 5.5, "Los Angeles": 3, "Seattle": 4},
    }
    # Climate change trends per decade per variable (very rough)
    trends = {"tas": 0.3, "pr": 2e-7, "hurs": -0.5, "sfcWind": 0.0}

    for scenario in scenarios:
        ssp_factor = 1.2 if scenario == "ssp370" else 1.0
        for variable in variables:
            var_dir = raw_dir / scenario / variable
            var_dir.mkdir(parents=True, exist_ok=True)
            for city, coords in cities.items():
                for year in years:
                    baseline = baselines[variable][city]
                    decade_offset = (year - 2015) / 10.0
                    trend = trends[variable] * decade_offset * ssp_factor
                    # Use 20% relative noise so all variables (including tiny pr values) have non-zero std
                    noise = rng.normal(0, abs(baseline) * 0.20, (patch_size, patch_size))
                    patch = baseline + trend + noise
                    path = var_dir / f"{city}_{year}.npy"
                    np.save(path, patch.astype(np.float32))

    logger.info("Synthetic climate data written to %s", raw_dir)


def _download_from_huggingface(
    cities: Dict[str, Dict[str, float]],
    variables: List[str],
    scenarios: List[str],
    years: List[int],
    patch_size: int,
    cmip6_model: str,
    hf_repo: str,
    raw_dir: Path,
) -> None:
    """
    Download climate data from climateset/causalpaca on HuggingFace Hub.

    ClimateSet organises files as:
      {model}/{scenario}/{variable}/{model}_{scenario}_{variable}_{year}.nc

    We download only the years we need, extract city patches, and save as .npy.
    """
    try:
        import xarray as xr
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("xarray and huggingface_hub are required for HF download") from e

    half = patch_size // 2

    for scenario in scenarios:
        for variable in variables:
            var_dir = raw_dir / scenario / variable
            var_dir.mkdir(parents=True, exist_ok=True)
            for year in years:
                # Skip if already downloaded
                all_city_done = all(
                    (var_dir / f"{city}_{year}.npy").exists()
                    for city in cities
                )
                if all_city_done:
                    continue

                filename = f"{cmip6_model}/{scenario}/{variable}/{cmip6_model}_{scenario}_{variable}_{year}.nc"
                logger.info("Downloading %s from %s", filename, hf_repo)
                try:
                    local_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(raw_dir / "_hf_cache"),
                    )
                except Exception as e:
                    logger.warning("Could not download %s: %s. Skipping year %d.", filename, e, year)
                    continue

                ds = xr.open_dataset(local_path)
                # Identify the data variable (usually the variable name itself)
                data_var = variable if variable in ds else list(ds.data_vars)[0]
                lats = ds["lat"].values
                lons = ds["lon"].values
                # Annual mean across time dimension
                data = ds[data_var].mean(dim="time").values  # (lat, lon)

                for city, coords in cities.items():
                    lat_idx, lon_idx = _nearest_grid_index(lats, lons, coords["lat"], coords["lon"])
                    patch = _extract_patch(data, lat_idx, lon_idx, half)
                    np.save(var_dir / f"{city}_{year}.npy", patch.astype(np.float32))

                ds.close()


def download_climate_data(config: dict) -> None:
    """
    Entry point for the data download step.
    Reads config and delegates to synthetic or HuggingFace download.
    """
    from src.data.cities import CITIES

    raw_dir      = Path(config["paths"]["raw_data"])
    variables    = config["variables"]
    scenarios    = config["scenarios"]
    year_start   = config["years"]["start"]
    year_end     = config["years"]["end"]
    patch_size   = config["dataset"]["patch_size"]
    use_synthetic = config["dataset"].get("use_synthetic", False)

    years = list(range(year_start, year_end + 1))

    if use_synthetic:
        logger.info("Generating synthetic climate data (use_synthetic=True)")
        _generate_synthetic_data(CITIES, variables, scenarios, years, patch_size, raw_dir)
    else:
        logger.info("Downloading from HuggingFace: %s", config["dataset"]["hf_repo"])
        _download_from_huggingface(
            cities=CITIES,
            variables=variables,
            scenarios=scenarios,
            years=years,
            patch_size=patch_size,
            cmip6_model=config["dataset"]["cmip6_model"],
            hf_repo=config["dataset"]["hf_repo"],
            raw_dir=raw_dir,
        )

    logger.info("Download complete. Raw data at: %s", raw_dir)
