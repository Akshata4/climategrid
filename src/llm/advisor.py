"""
Agentic LLM advisor — powered by Google Gemini (free tier).

Runs an agentic Gemini loop: the model decides which tools to call,
in what order, and writes a structured mitigation brief.

Free tier: gemini-1.5-flash — 1,500 requests/day, 15 RPM.
Get a key at https://aistudio.google.com (no credit card required).
"""

import logging
import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

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
        self._client: genai.Client | None = None
        self.model_name  = config["llm"]["model"]
        self.max_tokens  = config["llm"]["max_tokens"]
        self.temperature = config["llm"].get("temperature", 0.3)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. "
                    "Get a free key at https://aistudio.google.com and add it to .env as GOOGLE_API_KEY=..."
                )
            self._client = genai.Client(api_key=api_key)
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

        chat_config = types.GenerateContentConfig(
            tools=TOOL_DEFINITIONS,
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        chat = self.client.chats.create(model=self.model_name, config=chat_config)
        response = chat.send_message(user_content)

        # ── Agentic tool-use loop ─────────────────────────────────────────────
        final_text: str = ""
        last_risk_scores: Optional[Dict[str, float]] = None
        max_iterations = 10

        for _ in range(max_iterations):
            if not response.candidates or not response.candidates[0].content.parts:
                break

            parts = response.candidates[0].content.parts

            # Collect function calls from this response turn
            function_calls = [
                part.function_call
                for part in parts
                if part.function_call and part.function_call.name
            ]

            if not function_calls:
                # No tool calls — extract final text
                final_text = "".join(p.text for p in parts if p.text)
                break

            # Execute each tool and build function_response parts
            fn_response_parts = []
            for fc in function_calls:
                tool_input = dict(fc.args)
                logger.info("Tool call: %s(%s)", fc.name, tool_input)
                result = execute_tool(fc.name, tool_input, predictor)
                logger.info("Tool result: %s", result)

                if fc.name == "get_risk_scores" and "risk_scores" in result:
                    last_risk_scores = result["risk_scores"]
                elif fc.name == "compare_scenarios":
                    last_risk_scores = result.get(scenario, {})

                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response=result,
                    )
                )

            response = chat.send_message(fn_response_parts)

        if not final_text:
            logger.warning("Advisor loop ended without a final text response.")
            final_text = "Unable to generate mitigation brief. Please try again."

        # Fallback: run predict directly if no tool was ever called
        if last_risk_scores is None:
            last_risk_scores = predictor.predict(city, scenario, year)

        return {
            "risk_scores":      last_risk_scores,
            "mitigation_brief": final_text,
        }
