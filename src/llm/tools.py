"""
Gemini tool definitions for the agentic advisor.

Three tools:
  get_risk_scores     — run the climate model for one (city, scenario, year)
  compare_scenarios   — run the model for both SSP2 and SSP3 for a given city/year
  get_city_profile    — return static city metadata (geography, known climate risks)
"""

from typing import Any, Dict

from google.genai import types

from src.data.cities import CITIES, SSP_SCENARIOS

# ── Tool schemas (passed to the Gemini API) ───────────────────────────────────

_city_list = str(list(CITIES))

_get_risk_scores = types.FunctionDeclaration(
    name="get_risk_scores",
    description=(
        "Run the ClimateGrid model to retrieve climate disaster risk scores "
        "for a specific city, emission scenario, and future year. "
        "Returns four risk scores in [0, 1]: heat, flood, wildfire, drought."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city":     types.Schema(type=types.Type.STRING,  description=f"City name. One of: {_city_list}"),
            "scenario": types.Schema(type=types.Type.STRING,  description="SSP scenario: 'ssp245' (moderate) or 'ssp370' (high emissions)."),
            "year":     types.Schema(type=types.Type.INTEGER, description="Future year between 2015 and 2060."),
        },
        required=["city", "scenario", "year"],
    ),
)

_compare_scenarios = types.FunctionDeclaration(
    name="compare_scenarios",
    description=(
        "Compare climate risk under SSP2-4.5 (moderate emissions) vs SSP3-7.0 "
        "(high emissions, fragmented world) for the same city and year. "
        "Returns risk scores for both scenarios side by side."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(type=types.Type.STRING,  description=f"City name. One of: {_city_list}"),
            "year": types.Schema(type=types.Type.INTEGER, description="Future year between 2015 and 2060."),
        },
        required=["city", "year"],
    ),
)

_get_city_profile = types.FunctionDeclaration(
    name="get_city_profile",
    description=(
        "Return geographic and climate context for a city: coordinates, "
        "primary known climate hazards, and relevant infrastructure notes."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(type=types.Type.STRING, description=f"City name. One of: {_city_list}"),
        },
        required=["city"],
    ),
)

TOOL_DEFINITIONS = [
    types.Tool(function_declarations=[_get_risk_scores, _compare_scenarios, _get_city_profile])
]

# ── Static city profiles ──────────────────────────────────────────────────────

_CITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Houston": {
        "lat": 29.76, "lon": -95.37,
        "primary_hazards": ["flooding", "extreme heat", "hurricanes"],
        "notes": "Located in a low-lying coastal plain. Harris County is one of the most flood-prone counties in the US. Experiences intense hurricane-season rainfall.",
    },
    "Phoenix": {
        "lat": 33.45, "lon": -112.07,
        "primary_hazards": ["extreme heat", "drought", "wildfire smoke"],
        "notes": "Urban heat island effect is severe. Regularly records the most extreme-heat days of any major US city. Water supply depends heavily on the Colorado River.",
    },
    "Miami": {
        "lat": 25.76, "lon": -80.19,
        "primary_hazards": ["sea level rise", "storm surge", "flooding", "extreme heat"],
        "notes": "Built on porous limestone, making it uniquely vulnerable to saltwater intrusion and sunny-day flooding. One of the US cities most exposed to sea level rise.",
    },
    "Chicago": {
        "lat": 41.88, "lon": -87.63,
        "primary_hazards": ["extreme heat", "flooding", "cold extremes"],
        "notes": "The 1995 heat wave killed over 700 people. Combined sewer system is prone to overflow flooding. Climate change is increasing summer heat frequency.",
    },
    "Los Angeles": {
        "lat": 34.05, "lon": -118.24,
        "primary_hazards": ["wildfire", "drought", "extreme heat"],
        "notes": "Surrounded by fire-prone chaparral. The Santa Ana wind events drive rapid fire spread. Long-term drought is worsening water scarcity.",
    },
    "Seattle": {
        "lat": 47.61, "lon": -122.33,
        "primary_hazards": ["drought", "wildfire smoke", "changing precipitation"],
        "notes": "Historically mild climate is shifting. The 2021 heat dome killed hundreds in the Pacific Northwest. Snowpack decline threatens summer water supplies.",
    },
}


# ── Tool execution functions ──────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: Dict[str, Any], predictor) -> Dict[str, Any]:
    """
    Dispatch a tool call from the Gemini agent to the appropriate function.

    Returns a plain Python dict with only JSON-serializable native types
    (protobuf Struct requires no numpy floats).
    """
    if tool_name == "get_risk_scores":
        scores = predictor.predict(
            city=tool_input["city"],
            scenario=tool_input["scenario"],
            year=int(tool_input["year"]),
        )
        return {
            "city":        tool_input["city"],
            "scenario":    tool_input["scenario"],
            "year":        int(tool_input["year"]),
            "risk_scores": {k: float(v) for k, v in scores.items()},
        }

    elif tool_name == "compare_scenarios":
        results = {}
        for scenario in SSP_SCENARIOS:
            scores = predictor.predict(
                city=tool_input["city"],
                scenario=scenario,
                year=int(tool_input["year"]),
            )
            results[scenario] = {k: float(v) for k, v in scores.items()}
        return {
            "city":   tool_input["city"],
            "year":   int(tool_input["year"]),
            "ssp245": results.get("ssp245", {}),
            "ssp370": results.get("ssp370", {}),
        }

    elif tool_name == "get_city_profile":
        city = tool_input["city"]
        if city not in _CITY_PROFILES:
            return {"error": f"No profile for city '{city}'."}
        return _CITY_PROFILES[city]

    else:
        return {"error": f"Unknown tool: {tool_name}"}
