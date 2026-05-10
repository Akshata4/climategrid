# ClimateGrid

Urban climate disaster risk intelligence for city administrators and planners.

Given a city, a future year, and a climate emission scenario — ClimateGrid predicts how severe climate-related disasters will be and what the city should do about it.

---

## What it does

City planners make infrastructure decisions today that need to remain effective for the next 30–50 years. ClimateGrid bridges the gap between raw climate model outputs and actionable planning guidance.

A city administrator selects:
- **City** — Houston, Phoenix, Miami, Chicago, Los Angeles, or Seattle (click on the map or use the dropdown)
- **Scenario** — SSP2-4.5 (moderate emissions) or SSP3-7.0 (high emissions)
- **Year** — any year from 2025 to 2060

And receives:
- An interactive US map with colour-coded risk markers for the selected city
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
  Gemini 3.1 Flash-Lite  Agentic tool-use loop → mitigation brief
        ↓
  Gradio dashboard       Interactive map + bar chart + AI brief
```

**Data:** [ClimateSet](https://huggingface.co/datasets/climateset/causalpaca) — CMIP6 climate model outputs (NorESM2-LM), variables: surface temperature (`tas`), precipitation (`pr`), relative humidity (`hurs`), wind speed (`sfcWind`), scenarios SSP2-4.5 and SSP3-7.0.

**Model:** [ClimaX](https://huggingface.co/microsoft/ClimaX) (frozen) as a spatial feature extractor, plus a trainable MLP risk scoring head. Only the head is trained — no full fine-tuning.

**Labels:** Synthetic risk proxies derived from raw climate variables (documented limitation — real labels would require observational damage data).

**LLM:** Google Gemini 3.1 Flash-Lite via the Gemini API (free tier — no credit card required). The advisor agent autonomously calls `get_risk_scores`, `compare_scenarios`, and `get_city_profile` tools before writing any response.

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
│   │   ├── tools.py           # Gemini tool definitions and execution
│   │   └── advisor.py         # agentic loop: tools → mitigation brief
│   └── dashboard/
│       └── app.py             # Gradio UI with interactive Leaflet map
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_api.py
```

---

## Getting started (first time on a new machine)

### 1. Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12 or 3.13 | [python.org](https://www.python.org/downloads/) |
| uv | any recent | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Mac/Linux) or see [docs](https://docs.astral.sh/uv/getting-started/installation/) |
| Git | any | already installed on most systems |

Verify before continuing:
```bash
python --version   # should say 3.12.x or 3.13.x
uv --version       # should say uv 0.x.x
```

### 2. Clone and install

```bash
git clone <repo-url>
cd climategrid-1

# Install all Python dependencies into a local virtual environment
uv sync
```

`uv sync` reads `pyproject.toml` and installs every dependency automatically. No need to activate a venv — all `uv run` commands use it automatically.

### 3. Get a free Google API key

The AI advisor uses Google Gemini 3.1 Flash-Lite, which is **free** (no credit card):

1. Go to [https://aistudio.google.com](https://aistudio.google.com) and sign in with a Google account
2. Click **"Get API key"** → **"Create API key"**
3. Copy the key

### 4. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your key:
```
GOOGLE_API_KEY=your_key_here
```

> Each teammate needs their own `.env` file with their own key. The `.env` file is gitignored — never commit it.

### 5. Train the model

You need a trained checkpoint before the API can run. Use synthetic data (fast, no download):

```bash
uv run python scripts/run_pipeline.py --synthetic --epochs 10
```

This takes about 1–2 minutes and saves a checkpoint to `artifacts/best_model.pt`.

> You only need to do this once per machine. After that, skip straight to step 6.

### 6. Run the app

Open **two terminals** in the project folder:

**Terminal 1 — API server:**
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Gradio dashboard:**
```bash
uv run python src/dashboard/app.py
```

Open [http://localhost:7860](http://localhost:7860) in a browser. You should see the interactive map with all 6 cities.

---

## Using the dashboard

1. **Select a city** — click any marker on the map, or use the City dropdown
2. **Choose a scenario** — SSP2-4.5 (moderate) or SSP3-7.0 (high emissions)
3. **Set a year** — drag the slider to any year between 2025 and 2060
4. **Ask a question** (optional) — e.g. *"What should we prioritise for flood resilience?"*
5. **Click Analyse** — the map marker updates colour based on risk level, a bar chart appears, and the AI writes a mitigation brief below

**Map colour legend:**

| Colour | Average risk |
|---|---|
| Red | High (≥ 0.65) |
| Orange | Medium (0.40–0.65) |
| Yellow | Low (0.20–0.40) |
| Green | Minimal (< 0.20) |
| Grey | Not yet analysed |

---

## Running the pipeline (reference)

### Quick start — synthetic data

```bash
uv run python scripts/run_pipeline.py --synthetic --epochs 10
```

### Full pipeline — real ClimateSet data

```bash
uv run python scripts/run_pipeline.py
```

Downloads city grid cells from `climateset/causalpaca` on HuggingFace (NorESM2-LM, SSP2-4.5 and SSP3-7.0). Add `--epochs 50` for full training.

### Pipeline flags

| Flag | Description |
|---|---|
| `--synthetic` | Generate synthetic climate data instead of downloading |
| `--epochs N` | Override number of training epochs |
| `--skip-download` | Skip download, use already-downloaded data |
| `--skip-preprocess` | Skip preprocessing, use already-processed tensors |

---

## API reference

**`GET /health`**
```json
{"status": "ok"}
```

**`POST /predict`** — direct model inference
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

**`POST /advise`** — AI mitigation brief
```bash
curl -X POST http://localhost:8000/advise \
  -H "Content-Type: application/json" \
  -d '{"city": "Phoenix", "scenario": "ssp370", "year": 2050, "question": "What should we prioritise for heat resilience?"}'
```
Returns `risk_scores` + `mitigation_brief` (AI-written structured plan).

---

## Running tests

```bash
uv run pytest tests/ -q
```

Tests mock the model and the Gemini API, so no checkpoint or API key is needed to run them.

---

## Making changes

### Changing the LLM model

Edit `configs/config.yaml`:
```yaml
llm:
  model: "gemini-3.1-flash-lite"   # change this to any Gemini model ID
```

No code changes needed — the model name is read from config at startup.

### Adding a new city

1. Add coordinates to `configs/config.yaml` under `cities:`
2. Add a city profile to `src/llm/tools.py` in `_CITY_PROFILES`
3. Re-run the pipeline (`--synthetic` is fine for testing)

### Changing training settings

Edit `configs/config.yaml` under `training:` (batch size, epochs, learning rate). Then re-run:
```bash
uv run python scripts/run_pipeline.py --synthetic --skip-download --skip-preprocess --epochs 20
```

### Changing the risk head (MLP architecture)

Edit `model.risk_head_hidden` in `configs/config.yaml` — it's a list of hidden layer sizes:
```yaml
model:
  risk_head_hidden:
    - 256
    - 128
```

### Switching encoder (ClimaX → CNN)

Edit `configs/config.yaml`:
```yaml
model:
  encoder: cnn   # "climax" or "cnn"
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run commands with `uv run` from the project root (`climategrid-1/`), not from inside `src/`.

**`FileNotFoundError: artifacts/best_model.pt`**
The model has not been trained yet. Run the pipeline first:
```bash
uv run python scripts/run_pipeline.py --synthetic --epochs 10
```

**`Cannot reach the ClimateGrid API`** (in dashboard)
The API server is not running. Open a second terminal and run:
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**`GOOGLE_API_KEY is not set`**
Your `.env` file is missing or the key is wrong. Check that `.env` exists in the project root and contains `GOOGLE_API_KEY=...`. Get a free key at [aistudio.google.com](https://aistudio.google.com).

**Map not loading / city markers not appearing**
The map tiles and Leaflet load from a CDN. Make sure you have internet access when running the dashboard.

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
