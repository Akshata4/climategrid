"""
Gradio dashboard for ClimateGrid — with interactive Leaflet map.

Map interaction:
  - All 6 cities shown as circle markers on a US basemap
  - Clicking a marker (or its popup button) selects the city in the dropdown
  - After Analyse, the selected city is coloured by risk level; a legend explains the scale
"""

import yaml
import httpx
import gradio as gr
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

CITY_NAMES  = ["Houston", "Phoenix", "Miami", "Chicago", "Los Angeles", "Seattle"]
SCENARIOS   = {"SSP2-4.5 (moderate emissions)": "ssp245", "SSP3-7.0 (high emissions)": "ssp370"}
RISK_LABELS = ["Heat Risk", "Flood Risk", "Wildfire Risk", "Drought Risk"]
RISK_KEYS   = ["heat_risk", "flood_risk", "wildfire_risk", "drought_risk"]

CITY_COORDS = {
    "Houston":     (29.76, -95.37),
    "Phoenix":     (33.45, -112.07),
    "Miami":       (25.76, -80.19),
    "Chicago":     (41.88, -87.63),
    "Los Angeles": (34.05, -118.24),
    "Seattle":     (47.61, -122.33),
}


def _api_url() -> str:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return f"http://localhost:{cfg['api']['port']}"


def _risk_color(avg: float) -> str:
    if avg >= 0.65: return "#e74c3c"   # red
    if avg >= 0.40: return "#e67e22"   # orange
    if avg >= 0.20: return "#f1c40f"   # yellow
    return "#27ae60"                   # green


def _make_map_html(selected_city: str = None, city_risks: dict = None) -> str:
    """Return a self-contained Leaflet HTML page for gr.HTML (runs in iframe)."""
    city_risks = city_risks or {}
    markers = []

    for city, (lat, lon) in CITY_COORDS.items():
        is_selected = city == selected_city
        has_risk    = city in city_risks

        if has_risk:
            scores = city_risks[city]
            avg    = sum(scores.get(k, 0) for k in RISK_KEYS) / 4
            fill   = _risk_color(avg)
            radius = 14
            rows   = "".join(
                f"<tr><td style='padding:2px 8px 2px 0'>{lbl}</td>"
                f"<td><b>{scores.get(key, 0):.2f}</b></td></tr>"
                for lbl, key in zip(RISK_LABELS, RISK_KEYS)
            )
            popup = (
                f"<div style='font-family:sans-serif;min-width:170px'>"
                f"<b style='font-size:14px'>{city}</b>"
                f"<table style='margin-top:6px;font-size:12px;width:100%'>{rows}</table>"
                f"<button onclick=\"window.parent.postMessage({{type:'city_select',city:'{city}'}},'*')\" "
                f"style='margin-top:8px;width:100%;padding:5px 0;background:{fill};color:white;"
                f"border:none;border-radius:4px;cursor:pointer;font-size:13px'>"
                f"&#10003; Select {city}</button></div>"
            )
        else:
            fill   = "#95a5a6"
            radius = 10
            popup  = (
                f"<div style='font-family:sans-serif;text-align:center'>"
                f"<b>{city}</b><br>"
                f"<button onclick=\"window.parent.postMessage({{type:'city_select',city:'{city}'}},'*')\" "
                f"style='margin-top:8px;width:100%;padding:5px 0;background:#3498db;color:white;"
                f"border:none;border-radius:4px;cursor:pointer;font-size:13px'>Select {city}</button></div>"
            )

        popup_js    = popup.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "")
        border_col  = "#2c3e50" if is_selected else "#ffffff"
        border_w    = 3 if is_selected else 1.5

        markers.append(
            f"L.circleMarker([{lat},{lon}],"
            f"{{radius:{radius},color:'{border_col}',weight:{border_w},"
            f"fillColor:'{fill}',fillOpacity:0.85}}).addTo(map)"
            f".bindTooltip('<b>{city}</b>',{{permanent:false,direction:'top'}})"
            f".bindPopup('{popup_js}')"
            f".on('click',function(){{"
            f"window.parent.postMessage({{type:'city_select',city:'{city}'}},'*');}});"
        )

    markers_block = "\n".join(markers)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;padding:0;}} #map{{height:430px;width:100%;}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map',{{zoomControl:true}}).setView([39,-96],4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com">CARTO</a>',
    subdomains:'abcd', maxZoom:19
}}).addTo(map);

var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
    var d = L.DomUtil.create('div');
    d.style = 'background:white;padding:8px 12px;border-radius:6px;font-family:sans-serif;'
            + 'font-size:12px;box-shadow:0 1px 5px rgba(0,0,0,.3);line-height:2';
    d.innerHTML = '<b>Avg Risk</b><br>'
        + '<span style="color:#e74c3c;font-size:15px">&#11044;</span> High (&ge;0.65)<br>'
        + '<span style="color:#e67e22;font-size:15px">&#11044;</span> Medium (0.4&ndash;0.65)<br>'
        + '<span style="color:#f1c40f;font-size:15px">&#11044;</span> Low (0.2&ndash;0.4)<br>'
        + '<span style="color:#27ae60;font-size:15px">&#11044;</span> Minimal (&lt;0.2)<br>'
        + '<span style="color:#95a5a6;font-size:15px">&#11044;</span> Not analysed';
    return d;
}};
legend.addTo(map);

{markers_block}
</script>
</body></html>"""


def analyse(city: str, scenario_label: str, year: int, question: str) -> tuple:
    """Call POST /advise and return (chart data, brief text, updated map HTML)."""
    scenario = SCENARIOS[scenario_label]
    payload = {
        "city":     city,
        "scenario": scenario,
        "year":     int(year),
        "question": question.strip() if question and question.strip() else None,
    }
    try:
        resp = httpx.post(f"{_api_url()}/advise", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return (None,
                "⚠️ Cannot reach the API — make sure `uvicorn src.api.main:app` is running.",
                _make_map_html())
    except Exception as e:
        return None, f"⚠️ Error: {e}", _make_map_html()

    scores     = data["risk_scores"]
    chart_data = {"Risk Type": RISK_LABELS, "Score": [scores[k] for k in RISK_KEYS]}
    brief      = data["mitigation_brief"]
    new_map    = _make_map_html(selected_city=city, city_risks={city: scores})
    return chart_data, brief, new_map


# Injected at page load: listens for postMessage from the map iframe and clicks
# the matching off-screen Gradio button to update the city dropdown.
_SETUP_JS = """
() => {
    window.addEventListener('message', function(e) {
        if (!e.data || e.data.type !== 'city_select') return;
        var safeId = 'city_btn_' + e.data.city.replace(/ /g, '_');
        var container = document.getElementById(safeId);
        if (container) {
            var btn = container.querySelector('button');
            if (btn) btn.click();
        }
    });
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="ClimateGrid") as demo:

    # Push the hidden city-select buttons off-screen so they stay in the DOM
    # but are completely invisible to the user.
    gr.HTML("""<style>
    .city-select-btn {
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    </style>""")

    gr.Markdown("""
# 🌍 ClimateGrid
### Urban Climate Disaster Risk Intelligence
**Click any city on the map** to select it — or use the dropdown. Adjust the scenario and year, then click **Analyse**.
    """)

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
            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=2):
            map_component = gr.HTML(value=_make_map_html())
            risk_chart    = gr.BarPlot(
                value=None,
                x="Risk Type", y="Score", color="Risk Type",
                y_lim=[0, 1], title="Climate Hazard Risk Scores",
                tooltip=["Risk Type", "Score"], height=260,
            )

    brief_md = gr.Markdown("*Click a city on the map or use the dropdown, then click **Analyse**.*")

    # Off-screen hidden buttons (one per city) — the JS listener above clicks
    # whichever one matches the postMessage city name.
    for city in CITY_NAMES:
        _btn = gr.Button(
            city,
            elem_id=f"city_btn_{city.replace(' ', '_')}",
            elem_classes=["city-select-btn"],
        )
        _btn.click(fn=lambda c=city: c, outputs=city_dd)

    analyse_btn.click(
        fn=analyse,
        inputs=[city_dd, scenario_dd, year_sl, question_tb],
        outputs=[risk_chart, brief_md, map_component],
    )

    demo.load(fn=None, js=_SETUP_JS)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
