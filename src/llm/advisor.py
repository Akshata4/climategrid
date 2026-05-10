"""
Agentic LLM advisor.

Runs an agentic Claude loop: Claude decides which tools to call,
in what order, and writes a structured mitigation brief.

The advisor NEVER answers without calling get_risk_scores first —
this is enforced by the system prompt.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

from src.llm.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ClimateGrid, an expert climate risk advisor for city planners and administrators.

Your job is to answer questions about climate disaster risk in specific cities under different future emission scenarios.

RULES YOU MUST FOLLOW:
1. You MUST call get_risk_scores (or compare_scenarios) before writing any risk assessment. Never guess or hallucinate risk scores.
2. You SHOULD call get_city_profile to understand the city's geography and known vulnerabilities before writing recommendations.
3. After retrieving data, write a concise, structured mitigation brief in plain language — no jargon.

BRIEF FORMAT:
## Climate Risk Assessment: {city} — {scenario} — {year}

**Risk Scores** (0 = no risk, 1 = extreme risk):
- Heat risk: X.XX
- Flood risk: X.XX
- Wildfire risk: X.XX
- Drought risk: X.XX

**Key Finding**: One sentence summarising the dominant risk.

**Short-term actions (0–5 years)**:
- Bullet point recommendations

**Medium-term actions (5–15 years)**:
- Bullet point recommendations

**Long-term actions (15–30 years)**:
- Bullet point recommendations

**Scenario note** (if comparing SSP2 vs SSP3): brief comparison.

Keep language accessible to a non-technical city administrator. Be specific and actionable."""


class Advisor:
    def __init__(self, config: dict):
        self.config = config
        self._client: anthropic.Anthropic | None = None
        self.model  = config["llm"]["model"]
        self.max_tokens = config["llm"]["max_tokens"]

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Copy .env.example to .env and add your key."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def run(
        self,
        city: str,
        scenario: str,
        year: int,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agentic loop for a single query.

        Returns:
            dict with keys:
              risk_scores:      {heat_risk, flood_risk, wildfire_risk, drought_risk}
              mitigation_brief: str
        """
        from src.inference.predictor import get_predictor
        predictor = get_predictor(self.config)

        user_content = (
            f"City: {city}\n"
            f"Scenario: {scenario} ({'moderate' if scenario == 'ssp245' else 'high'} emissions)\n"
            f"Year: {year}\n"
        )
        if question:
            user_content += f"\nSpecific question: {question}"
        else:
            user_content += "\nPlease provide a full climate risk assessment and mitigation brief."

        messages: List[Dict] = [{"role": "user", "content": user_content}]

        # ── Agentic tool-use loop ─────────────────────────────────────────────
        final_text: str = ""
        last_risk_scores: Optional[Dict[str, float]] = None
        max_iterations = 10

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Append assistant message
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract the final text block
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input))
                    result = execute_tool(block.name, block.input, predictor)
                    logger.info("Tool result: %s", json.dumps(result))

                    # Track the most recent risk scores for the response envelope
                    if block.name == "get_risk_scores" and "risk_scores" in result:
                        last_risk_scores = result["risk_scores"]
                    elif block.name == "compare_scenarios":
                        # Use the requested scenario's scores
                        last_risk_scores = result.get(scenario, {})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                messages.append({"role": "user", "content": tool_results})

        if not final_text:
            logger.warning("Advisor loop ended without a final text response.")
            final_text = "Unable to generate mitigation brief. Please try again."

        # If tool was never called or scores weren't captured, run predict directly
        if last_risk_scores is None:
            last_risk_scores = predictor.predict(city, scenario, year)

        return {
            "risk_scores":      last_risk_scores,
            "mitigation_brief": final_text,
        }
