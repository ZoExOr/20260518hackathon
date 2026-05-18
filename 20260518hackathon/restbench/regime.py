"""RegimeSupervisor — sets the high-level Mode and (optionally) param
overrides. This is the ONLY place an LLM is allowed, and it runs at most
once every `params.llm_every_k_days` days — never per turn, never 6 agents.

OWNER: adaptation workstream (optional; ship HeuristicRegime first).

Contract: `decide(obs, belief, params) -> (Mode, dict_of_param_overrides)`.
The agent applies the overrides on top of the tuned Params for that turn.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .types import Observation, BeliefState, Mode
from .params import Params
from .llm_client import DEFAULT_LLM_MODEL, chat_completion
from .renovation import renovation_active, renovation_recovery


class RegimeSupervisor(Protocol):
    def decide(self, obs: Observation, belief: BeliefState,
               params: Params) -> tuple[Mode, dict]:
        ...


class HeuristicRegime:
    """Zero-cost, deterministic, reproducible. The sensible default."""

    def decide(self, obs: Observation, belief: BeliefState,
               params: Params) -> tuple[Mode, dict]:
        if renovation_active(obs, belief):
            return Mode.DEMAND_SLUMP, {
                "target_days": max(6.0, params.target_days - 1.0),
                "safety_days": max(1.5, params.safety_days),
                "marketing_slump": 0.0,
            }

        if renovation_recovery(obs, belief):
            return Mode.DEMAND_SURGE, {
                "target_days": params.target_days + 1.5,
                "safety_days": params.safety_days + 0.5,
                "marketing_default": max(params.marketing_default, 125.0),
            }

        if obs.days_remaining <= params.endgame_window:
            # Protect FINAL reputation: stop chasing risky margin.
            return Mode.ENDGAME, {"base_price_mult": min(
                1.0, params.base_price_mult), "waste_aversion": 1.5}

        alerts = " ".join(obs.alerts).lower()
        if "supplier" in alerts or "outage" in alerts or "halt" in alerts:
            return Mode.SUPPLY_DEFENSIVE, {
                "target_days": params.target_days + 3.0,
                "reliability_inflation": params.reliability_inflation + 1.0}

        if obs.customer_trend == "Growing":
            return Mode.DEMAND_SURGE, {"target_days": params.target_days + 2.0}
        if obs.customer_trend == "Declining" or \
                belief.reputation_trajectory == "Falling":
            return Mode.DEMAND_SLUMP, {"marketing_slump": max(
                params.marketing_slump, 250.0)}
        return Mode.NORMAL, {}


class LLMRegime:
    """Optional. Calls one LLM every k days with a COMPACT summary. Falls
    back to HeuristicRegime on any error so a flaky API never breaks a run.

    Uses the team's OpenAI-compatible LiteLLM proxy. Set RESTBENCH_LLM_API_KEY
    or OPENAI_API_KEY, plus optional AGENT_MODEL. Keep the prompt tiny — this
    is a supervisor, not a controller.
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AGENT_MODEL", DEFAULT_LLM_MODEL)
        self._fallback = HeuristicRegime()
        self._last_day = -999
        self._last = (Mode.NORMAL, {})

    def decide(self, obs: Observation, belief: BeliefState,
               params: Params) -> tuple[Mode, dict]:
        if obs.day - self._last_day < params.llm_every_k_days:
            return self._last
        self._last_day = obs.day
        try:
            summary = {
                "day": obs.day, "days_remaining": obs.days_remaining,
                "cash": round(obs.cash), "reputation_band": obs.reputation_band,
                "trend": obs.customer_trend, "alerts": obs.alerts,
                "weather_forecast": obs.weather_forecast,
                "rep_est": round(belief.reputation_est, 2),
            }
            prompt = (
                "You supervise a restaurant agent. Pick ONE mode from "
                "[normal, supply_defensive, demand_surge, demand_slump, "
                "endgame] and optional numeric overrides for keys "
                "{target_days,reliability_inflation,base_price_mult,"
                "marketing_slump}. Reply ONLY JSON: "
                '{"mode": "...", "overrides": {...}}.\nState: '
                + json.dumps(summary))
            r = chat_completion(
                [{"role": "user", "content": prompt}],
                model=self.model, max_tokens=200,
                temperature=0.2)
            txt = r.choices[0].message.content
            txt = txt[txt.find("{"): txt.rfind("}") + 1]
            data = json.loads(txt)
            mode = Mode(data.get("mode", "normal"))
            self._last = (mode, dict(data.get("overrides", {})))
        except Exception:
            self._last = self._fallback.decide(obs, belief, params)
        return self._last
