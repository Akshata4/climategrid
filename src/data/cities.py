from typing import Dict

CITIES: Dict[str, Dict[str, float]] = {
    "Houston":     {"lat": 29.7604, "lon": -95.3698},
    "Phoenix":     {"lat": 33.4484, "lon": -112.0740},
    "Miami":       {"lat": 25.7617, "lon": -80.1918},
    "Chicago":     {"lat": 41.8781, "lon": -87.6298},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "Seattle":     {"lat": 47.6062, "lon": -122.3321},
}

CITY_NAMES = list(CITIES.keys())

SSP_SCENARIOS = ["ssp245", "ssp370"]

# CMIP6 variable names → human-readable labels for display
VARIABLE_LABELS: Dict[str, str] = {
    "tas":     "Surface Temperature (K)",
    "pr":      "Precipitation (kg/m²/s)",
    "hurs":    "Relative Humidity (%)",
    "sfcWind": "Wind Speed (m/s)",
}


def get_city(name: str) -> Dict[str, float]:
    if name not in CITIES:
        raise ValueError(f"Unknown city '{name}'. Choose from: {CITY_NAMES}")
    return CITIES[name]
