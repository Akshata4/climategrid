# ClimateGrid

Urban climate disaster risk intelligence for city administrators and planners.

Given a city, a future year, and a climate emission scenario — ClimateGrid predicts how severe climate-related disasters will be and what the city should do about it.

---

## What it does

City planners make infrastructure decisions today that need to remain effective for the next 30–50 years. ClimateGrid bridges the gap between raw climate model outputs and actionable planning guidance.

A city administrator selects:
- **City** — Houston, Phoenix, Miami, Chicago, Los Angeles, or Seattle
- **Scenario** — SSP2-4.5 (moderate emissions) or SSP3-7.0 (high emissions)
- **Year** — any year from 2025 to 2060

And receives:
- Four hazard risk scores (heat, flood, wildfire, drought) on a 0–1 scale
- An AI-written mitigation brief with short, medium, and long-term recommended actions

---

## Architecture

```
ClimateSet (HuggingFace)
        ↓
  Data pipeline          download.py → preprocess.py → dataset.py
        ↓
  ClimaX encoder         Frozen ViT (microsoft/ClimaX) — feature extraction only
        +
  Risk head              Trainable MLP → 4 risk scores in [0, 1]
        ↓
  FastAPI                POST /predict  (direct inference)
                         POST /advise   (LLM agentic loop)
        ↓
  Claude Sonnet          Agentic tool-use loop → mitigation brief
        ↓
  Gradio dashboard       City planner UI
```

**Data:** [ClimateSet](https://huggingface.co/datasets/climateset/causalpaca) — CMIP6 climate model outputs (NorESM2-LM), variables: surface temperature (`tas`), precipitation (`pr`), relative humidity (`hurs`), wind speed (`sfcWind`), scenarios SSP2-4.5 and SSP3-7.0.

**Model:** [ClimaX](https://huggingface.co/microsoft/ClimaX) (frozen) as a spatial feature extractor, plus a trainable MLP risk scoring head. Only the head is trained — no full fine-tuning.

**Labels:** Synthetic risk proxies derived from raw climate variables (documented limitation — real labels would require observational damage data).

**LLM:** Claude Sonnet via the Anthropic API. The advisor agent autonomously calls `get_risk_scores`, `compare_scenarios`, and `get_city_profile` tools before writing any response.

---

## Project structure

```
climategrid/
├── configs/
│   └── config.yaml            # all settings: cities, model, training, paths
├── data/
│   ├── raw/                   # downloaded climate files (gitignored)
│   └── processed/             # normalised tensors + labels (gitignored)
├── artifacts/                 # model checkpoints (gitignored)
├── scripts/
│   └── run_pipeline.py        # orchestrates download → preprocess → train
├── src/
│   ├── data/
│   │   ├── cities.py          # city name → (lat, lon) lookup
│   │   ├── download.py        # HuggingFace download or synthetic data generation
│   │   ├── preprocess.py      # normalise, generate labels, save tensors
│   │   └── dataset.py         # PyTorch Dataset
│   ├── model/
│   │   ├── labels.py          # synthetic risk score formulas
│   │   ├── climax_encoder.py  # ViT encoder (ClimaX-style), frozen
│   │   ├── cnn_encoder.py     # CNN fallback encoder
│   │   ├── risk_head.py       # MLP → 4 risk scores
│   │   └── climategrid.py     # encoder + head combined
│   ├── training/
│   │   ├── train.py           # training loop (programmatic, not CLI)
│   │   └── evaluate.py        # per-risk MAE and MSE
│   ├── inference/
│   │   └── predictor.py       # load checkpoint, predict(city, scenario, year)
│   ├── api/
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── main.py            # FastAPI: /predict, /advise, /health
│   ├── llm/
│   │   ├── tools.py           # Claude tool definitions and execution
│   │   └── advisor.py         # agentic loop: tools → mitigation brief
│   └── dashboard/
│       └── app.py             # Gradio UI
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_api.py
```

---

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd climategrid

# Install all dependencies
uv sync

# Add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=...
```

---

## Running the pipeline

### Quick start — synthetic data (no download required)

Use this to verify the full pipeline before committing to the HuggingFace download.

```bash
uv run python scripts/run_pipeline.py --synthetic --epochs 10
```

### Full pipeline — real ClimateSet data

```bash
uv run python scripts/run_pipeline.py
```

This downloads only the required city grid cells from `climateset/causalpaca` (NorESM2-LM, SSP2-4.5 and SSP3-7.0). Add `--epochs 50` to run the full training.

### Pipeline flags

| Flag | Description |
|---|---|
| `--synthetic` | Generate synthetic climate data instead of downloading |
| `--epochs N` | Override number of training epochs |
| `--skip-download` | Skip download step (use already-downloaded data) |
| `--skip-preprocess` | Skip preprocessing step |

---

## Running the API

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

**`GET /health`**
```json
{"status": "ok"}
```

**`POST /predict`**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Houston", "scenario": "ssp245", "year": 2045}'
```
```json
{
  "city": "Houston",
  "scenario": "ssp245",
  "year": 2045,
  "risk_scores": {
    "heat_risk": 0.51,
    "flood_risk": 0.20,
    "wildfire_risk": 0.40,
    "drought_risk": 0.38
  }
}
```

**`POST /advise`**
```bash
curl -X POST http://localhost:8000/advise \
  -H "Content-Type: application/json" \
  -d '{"city": "Phoenix", "scenario": "ssp370", "year": 2050, "question": "What should we prioritise for heat resilience?"}'
```
Returns `risk_scores` + `mitigation_brief` (AI-written structured plan).

---

## Running the dashboard

With the API server running in one terminal:

```bash
uv run python src/dashboard/app.py
```

Open [http://localhost:7860](http://localhost:7860) in a browser.

---

## Running tests

```bash
uv run pytest tests/ -q
```

---

## Synthetic risk label formulas

These are documented proxies, not real risk indices. Real labels would require observational damage records.

| Risk | Formula |
|---|---|
| Heat | `sigmoid((mean_temp_°C − 30) / 10)` — risk = 0.5 at 30 °C |
| Flood | `clamp((mean_pr + 2×std_pr) / 20 mm/day, 0, 1)` — rewards high mean and high variability |
| Wildfire | `0.6 × heat_component + 0.4 × dryness_component` |
| Drought | `clamp(heat_norm × (1 − precip_norm), 0, 1)` — requires both high temp and low precip |

---

## Cities and scenarios

| City | Primary hazards |
|---|---|
| Houston | Flooding, extreme heat, hurricanes |
| Phoenix | Extreme heat, drought, wildfire smoke |
| Miami | Sea level rise, storm surge, flooding |
| Chicago | Extreme heat, flooding, cold extremes |
| Los Angeles | Wildfire, drought, extreme heat |
| Seattle | Drought, wildfire smoke, changing precipitation |

| Scenario | Description |
|---|---|
| `ssp245` | SSP2-4.5 — moderate emissions, middle-of-the-road future |
| `ssp370` | SSP3-7.0 — high emissions, fragmented world |

---

## Limitations

- **Synthetic labels** — risk scores are derived from climate variables by formula, not from real damage or loss data. This is the core scientific limitation.
- **Single CMIP6 model** — we use NorESM2-LM only. A real system would ensemble across multiple models to capture model uncertainty.
- **City-level granularity** — predictions are based on the nearest CMIP6 grid cell (~1° resolution), not fine-grained urban microclimate data.
- **ClimaX pre-training** — the ClimaX encoder is used with random initialisation because the original checkpoint architecture differs from our simplified encoder. The model still learns from the training data via the risk head.
- **Year range** — limited to 2015–2060 by the processed dataset. Queries outside this range use linear interpolation from the nearest available years.
