from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    city:     str = Field(..., examples=["Houston"])
    scenario: str = Field(..., examples=["ssp245"])
    year:     int = Field(..., ge=2015, le=2060, examples=[2045])


class RiskScores(BaseModel):
    heat_risk:     float = Field(..., ge=0.0, le=1.0)
    flood_risk:    float = Field(..., ge=0.0, le=1.0)
    wildfire_risk: float = Field(..., ge=0.0, le=1.0)
    drought_risk:  float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    city:        str
    scenario:    str
    year:        int
    risk_scores: RiskScores


class AdviseRequest(BaseModel):
    city:     str = Field(..., examples=["Houston"])
    scenario: str = Field(..., examples=["ssp245"])
    year:     int = Field(..., ge=2015, le=2060, examples=[2045])
    question: Optional[str] = Field(
        default=None,
        examples=["What should we prioritise for flood resilience?"],
    )


class AdviseResponse(BaseModel):
    city:             str
    scenario:         str
    year:             int
    risk_scores:      RiskScores
    mitigation_brief: str
