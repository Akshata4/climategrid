"""Smoke tests for the FastAPI endpoints (no model checkpoint required)."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_predictor():
    pred = MagicMock()
    pred.predict.return_value = {
        "heat_risk": 0.72,
        "flood_risk": 0.55,
        "wildfire_risk": 0.31,
        "drought_risk": 0.44,
    }
    return pred


@pytest.fixture
def mock_advisor(mock_predictor):
    adv = MagicMock()
    adv.run.return_value = {
        "risk_scores":      mock_predictor.predict.return_value,
        "mitigation_brief": "## Test Brief\n\nTest mitigation content.",
    }
    return adv


@pytest.fixture
def client(config, mock_predictor, mock_advisor):
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    # Patch at the lifespan import sites so no real checkpoint is needed
    with patch("src.inference.predictor.get_predictor", return_value=mock_predictor), \
         patch("src.llm.advisor.Advisor", return_value=mock_advisor):
        from src.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid(client):
    resp = client.post("/predict", json={"city": "Houston", "scenario": "ssp245", "year": 2045})
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "Houston"
    scores = data["risk_scores"]
    for key in ["heat_risk", "flood_risk", "wildfire_risk", "drought_risk"]:
        assert 0 <= scores[key] <= 1


def test_predict_invalid_city(client, mock_predictor):
    from src.data.cities import get_city
    mock_predictor.predict.side_effect = ValueError("Unknown city 'Atlantis'.")
    resp = client.post("/predict", json={"city": "Atlantis", "scenario": "ssp245", "year": 2045})
    assert resp.status_code == 400


def test_advise_valid(client):
    resp = client.post("/advise", json={"city": "Phoenix", "scenario": "ssp370", "year": 2050})
    assert resp.status_code == 200
    data = resp.json()
    assert "mitigation_brief" in data
    assert len(data["mitigation_brief"]) > 10


def test_predict_year_out_of_range(client):
    resp = client.post("/predict", json={"city": "Miami", "scenario": "ssp245", "year": 2099})
    # Pydantic validates year ≤ 2060 before reaching the handler
    assert resp.status_code == 422
