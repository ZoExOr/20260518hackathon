"""DemandAdvisor — picks today's posture ("push" | "hold" | "discount").

The PricingController consumes this and chooses the numbers
(marketing amount, happy hour on/off, price multiplier).

Fallback: deterministic posture chooser identical in spirit to the
HeuristicRegime — so the agent never blocks on LLM failure.
"""
from __future__ import annotations

from .client import LLMClient
from .prompts import DEMAND_SYSTEM, demand_user

VALID_POSTURES = {"push", "hold", "discount"}


def _heuristic_posture(obs, belief) -> str:
    """Deterministic fallback matching the prompt's rules."""
    if obs.cash < 4000:
        return "hold"
    if obs.reputation_band in {"Poor", "Fair"}:
        return "hold"
    is_weekend = obs.day_of_week in {"Friday", "Saturday", "Sunday"}
    good_weather = obs.weather_today in {"sunny", "cloudy"}
    growing = obs.customer_trend == "Growing"
    if obs.cash >= 5000 and obs.reputation_band in {"Very Good", "Excellent"} \
            and (is_weekend or good_weather or growing):
        return "push"
    slow_weekday = obs.day_of_week in {"Monday", "Tuesday", "Wednesday"}
    if slow_weekday and obs.reputation_band in {"Very Good", "Excellent"}:
        return "discount"
    return "hold"


class DemandAdvisor:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def choose_posture(self, obs, belief) -> str:
        fallback_posture = _heuristic_posture(obs, belief)
        data = self.client.chat_json(
            system=DEMAND_SYSTEM,
            user=demand_user(obs, belief),
            fallback={"posture": fallback_posture, "why": "fallback"},
            max_tokens=80,
            temperature=0.1,
        )
        posture = data.get("posture", fallback_posture)
        if posture not in VALID_POSTURES:
            posture = fallback_posture
        # Hard safety rails — never let LLM override survival logic.
        if posture == "push" and (obs.cash < 5000
                                  or obs.reputation_band in {"Poor", "Fair"}):
            posture = "hold"
        if posture == "discount" and obs.reputation_band in {"Poor", "Fair"}:
            posture = "hold"
        return posture