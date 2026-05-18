"""Evaluation adapter for the team's RestBench agent.

`agents.evaluate` expects a module-level strategy(observation, day) function.
The production agent is stateful, so this wrapper keeps one Agent per worker
thread; evaluate runs each game sequentially inside its worker thread.
"""
from __future__ import annotations

import os
import threading

from restbench.agent import Agent
from restbench.multiagent import MultiAgentAdvisor
from restbench.params import Params
from restbench.regime import HeuristicRegime, LLMRegime
from restbench.types import Observation

_LOCAL = threading.local()


def _new_agent() -> Agent:
    use_llm_regime = os.environ.get("RESTBENCH_MY_AGENT_LLM", "0") == "1"
    use_multi_agent = os.environ.get("RESTBENCH_MY_AGENT_MULTI_AGENT", "0") == "1"
    strict_llm = os.environ.get("RESTBENCH_MY_AGENT_STRICT_LLM", "0") == "1"
    regime = LLMRegime() if use_llm_regime else HeuristicRegime()
    advisor = MultiAgentAdvisor(allow_fallback=not strict_llm) if use_multi_agent else None
    return Agent(
        params=Params(),
        regime=regime,
        multi_agent=use_multi_agent,
        multi_agent_advisor=advisor,
    )


def _agent_for_day(day: int) -> Agent:
    current = getattr(_LOCAL, "agent", None)
    last_day = int(getattr(_LOCAL, "last_day", 0) or 0)
    if current is None or day <= 1 or day <= last_day:
        current = _new_agent()
        _LOCAL.agent = current
    _LOCAL.last_day = day
    return current


def strategy(observation: dict, day: int) -> list[dict]:
    agent = _agent_for_day(day)
    actions = agent.decide(Observation(observation))
    return [a.to_payload() for a in actions]
