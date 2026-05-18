"""LLM agent template — Local Ollama Version"""

from __future__ import annotations

import json
import sys

import ollama

from agents.runner import run_game

SYSTEM_PROMPT = """\
You manage an Italian restaurant for 30 simulated days. Each day you receive
an observation (JSON) describing your restaurant's state: cash, inventory,
suppliers, menu, reputation, yesterday's service results, and more.

Respond ONLY with a valid JSON object containing an "actions" array of tool calls.
No explanation, no markdown.

Format Example:
{
  "actions": [
    {"tool": "set_staff_level", "args": {"level": 5}},
    {"tool": "run_happy_hour", "args": {}}
  ]
}

Available tools:
- place_order: {"tool": "place_order", "args": {"supplier": "...", "ingredient": "...", "quantity_kg": N}}
- set_staff_level: {"tool": "set_staff_level", "args": {"level": N}}  (range: 3-15)
- set_price: {"tool": "set_price", "args": {"dish": "...", "price": N}}  (0.8x-1.2x base)
- set_menu: {"tool": "set_menu", "args": {"dishes": [...]}}  (min 5 dishes)
- set_marketing_spend: {"tool": "set_marketing_spend", "args": {"amount": N}}  (0-500 EUR)
- run_happy_hour: {"tool": "run_happy_hour", "args": {}}
- offer_daily_special: {"tool": "offer_daily_special", "args": {"dish": "..."}}
- save_notes: {"tool": "save_notes", "args": {"text": "..."}}  (up to 4000 chars, persists)

Your score = net_profit - penalties (satisfaction, reputation, walkouts, waste).
Going bankrupt (cash < 0) = -100,000 score. Survival is priority #1.

Use the exact supplier, ingredient, and dish names from the observation."""


def strategy(observation: dict, day: int) -> list[dict]:
    user_msg = f"Day {day}/30. Here is today's observation:\n\n{json.dumps(observation, indent=2)}"

    try:
        response = ollama.chat(
            model="qwen3.6:27b-coding-mxfp8",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            stream=False,
            think=False,
            options={
                "temperature": 0.0,
                "num_predict": 512,
                "num_ctx": 4096,
            },
            keep_alive="1h",
        )

        content = response["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed_json = json.loads(content)

        tool_calls = parsed_json.get("actions", [])

        if not isinstance(tool_calls, list):
            return []

        return tool_calls

    except Exception as e:
        print(f"  LLM error on day {day}: {e}")
        return []


if __name__ == "__main__":
    """
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY first.")
            print(f"Using model: {MODEL} (override with AGENT_MODEL env var)")
            sys.exit(1)

    print(f"Using model: {MODEL}")
    """

    # Removed the API key checks since you are running locally
    print("Using local Ollama model...")
    result = run_game(strategy, team_name="local_ai_chef", seed=42)
