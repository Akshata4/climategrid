"""
FastAPI application — two endpoints:

  POST /predict  → direct model inference (city, scenario, year → risk scores)
  POST /advise   → LLM agentic advisor   (city, scenario, year → risk scores + mitigation brief)
  GET  /health   → liveness probe
"""

import logging
import os
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.api.schemas import (
    AdviseRequest, AdviseResponse,
    PredictRequest, PredictResponse, RiskScores,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shared application state ─────────────────────────────────────────────────

_state: dict = {}


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()
    _state["config"] = config

    from src.inference.predictor import get_predictor
    _state["predictor"] = get_predictor(config)
    logger.info("Predictor loaded.")

    from src.llm.advisor import Advisor
    _state["advisor"] = Advisor(config)
    logger.info("Advisor ready.")

    yield  # app runs here

    _state.clear()


app = FastAPI(
    title="ClimateGrid API",
    description="Urban climate disaster risk intelligence — inference and advisory endpoints.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialised.")
    try:
        scores = predictor.predict(req.city, req.scenario, req.year)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        city=req.city,
        scenario=req.scenario,
        year=req.year,
        risk_scores=RiskScores(**scores),
    )


@app.post("/advise", response_model=AdviseResponse)
def advise(req: AdviseRequest) -> AdviseResponse:
    advisor = _state.get("advisor")
    if advisor is None:
        raise HTTPException(status_code=503, detail="Advisor not initialised.")
    try:
        result = advisor.run(req.city, req.scenario, req.year, req.question)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AdviseResponse(
        city=req.city,
        scenario=req.scenario,
        year=req.year,
        risk_scores=RiskScores(**result["risk_scores"]),
        mitigation_brief=result["mitigation_brief"],
    )
