"""
Gradio dashboard for ClimateGrid.

City planner workflow:
  1. Select city, SSP scenario, future year
  2. (Optional) type a specific question
  3. Click Analyse
  4. See a bar chart of the 4 risk scores
  5. Read the AI-written mitigation brief below
"""

import os
import yaml
import httpx
import gradio as gr

from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

CITY_NAMES  = ["Houston", "Phoenix", "Miami", "Chicago", "Los Angeles", "Seattle"]
SCENARIOS   = {"SSP2-4.5 (moderate emissions)": "ssp245", "SSP3-7.0 (high emissions)": "ssp370"}
RISK_LABELS = ["Heat Risk", "Flood Risk", "Wildfire Risk", "Drought Risk"]
RISK_KEYS   = ["heat_risk", "flood_risk", "wildfire_risk", "drought_risk"]
RISK_COLORS = ["#e74c3c", "#3498db", "#e67e22", "#f39c12"]

def _api_url() -> str:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    host = cfg["api"]["host"]
    port = cfg["api"]["port"]
    # Gradio and FastAPI run on the same machine during demo
    return f"http://localhost:{port}"


def analyse(city: str, scenario_label: str, year: int, question: str) -> tuple:
    """Call POST /advise and return (bar chart figure, mitigation brief text)."""
    scenario = SCENARIOS[scenario_label]
    payload = {
        "city": city,
        "scenario": scenario,
        "year": int(year),
        "question": question.strip() if question.strip() else None,
    }

    try:
        resp = httpx.post(f"{_api_url()}/advise", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return None, "⚠️ Cannot reach the ClimateGrid API. Make sure `uvicorn src.api.main:app` is running."
    except Exception as e:
        return None, f"⚠️ Error: {e}"

    scores = data["risk_scores"]
    values = [scores[k] for k in RISK_KEYS]
    brief  = data["mitigation_brief"]

    # Build bar chart with Gradio's native BarPlot via a simple dict
    chart_data = {
        "Risk Type": RISK_LABELS,
        "Score":     values,
    }

    return chart_data, brief


with gr.Blocks(title="ClimateGrid", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌍 ClimateGrid
        ### Urban Climate Disaster Risk Intelligence
        Select a city, emission scenario, and future year to get an AI-powered risk assessment
        and mitigation brief tailored for city administrators.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            city_dd     = gr.Dropdown(CITY_NAMES, label="City", value="Houston")
            scenario_dd = gr.Dropdown(list(SCENARIOS), label="Emission Scenario", value=list(SCENARIOS)[0])
            year_sl     = gr.Slider(minimum=2025, maximum=2060, step=1, value=2045, label="Future Year")
            question_tb = gr.Textbox(
                label="Specific question (optional)",
                placeholder="e.g. What should we prioritise for flood resilience?",
                lines=2,
            )
            analyse_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=2):
            risk_chart = gr.BarPlot(
                value=None,
                x="Risk Type",
                y="Score",
                color="Risk Type",
                y_lim=[0, 1],
                title="Climate Hazard Risk Scores",
                tooltip=["Risk Type", "Score"],
                height=300,
            )
            brief_md = gr.Markdown(value="*Click Analyse to generate a risk assessment.*")

    analyse_btn.click(
        fn=analyse,
        inputs=[city_dd, scenario_dd, year_sl, question_tb],
        outputs=[risk_chart, brief_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
