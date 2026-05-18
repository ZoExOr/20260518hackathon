"""Entrypoint — mirrors the starter kit convention.

    python -m agents.team_agent                       # one baseline game
    python -m agents.team_agent --scenario supply_crisis --seed 88
    python -m agents.team_agent --tune                # offline BO over Params
    python -m agents.team_agent --llm                 # use LLM regime supervisor

Set RESTBENCH_URL (defaults to the kit's server) and, for --llm,
ANTHROPIC_API_KEY/OPENAI_API_KEY + AGENT_MODEL.
"""
from __future__ import annotations

import argparse
import os
import threading

from restbench.params import Params
from restbench.replay import ReplayStore
from restbench.runner import play_game
from restbench.regime import HeuristicRegime, LLMRegime
from restbench.agent import Agent
from restbench.types import Observation
from restbench.tuning.harness import EvalConfig, optimise

TEAM = os.environ.get("RESTBENCH_TEAM", "pppp")
_STRATEGY_STATE = threading.local()
_STRATEGY_USE_LLM = False


def _prepare_llm_env() -> None:
    """Make LLM mode explicit and fail fast if no provider key is configured."""
    if os.getenv("OPENAI_API_KEY"):
        os.environ.setdefault("AGENT_MODEL", "openai/gpt-4.1-mini")
    elif os.getenv("ANTHROPIC_API_KEY"):
        os.environ.setdefault("AGENT_MODEL", "anthropic/claude-haiku-4-5")
    else:
        raise RuntimeError(
            "LLM mode requires an API key. In PowerShell run: "
            "$env:OPENAI_API_KEY='sk-...' or $env:ANTHROPIC_API_KEY='...'"
        )
    os.environ["RESTBENCH_REQUIRE_LLM"] = "1"


def configure_strategy(*, use_llm: bool = False) -> None:
    """Configure the evaluate-compatible strategy entrypoint."""
    global _STRATEGY_USE_LLM
    _STRATEGY_USE_LLM = use_llm
    if use_llm:
        _prepare_llm_env()
    if hasattr(_STRATEGY_STATE, "agent"):
        delattr(_STRATEGY_STATE, "agent")


def _strategy_agent(day: int) -> Agent:
    if day == 1 or not hasattr(_STRATEGY_STATE, "agent"):
        use_llm = _STRATEGY_USE_LLM or os.getenv("RESTBENCH_USE_LLM") == "1"
        if use_llm:
            _prepare_llm_env()
        regime = LLMRegime() if use_llm else HeuristicRegime()
        _STRATEGY_STATE.agent = Agent(params=Params(), regime=regime)
    return _STRATEGY_STATE.agent


def strategy(observation: dict, day: int) -> list[dict]:
    """Starter-kit evaluate entrypoint.

    `agents.evaluate` expects a module-level strategy(observation, day)
    function returning raw tool-call dictionaries. Use thread-local state so
    parallel evaluations do not share belief memory across games.
    """
    obs = Observation({**observation, "day": observation.get("day", day)})
    return [action.to_payload() for action in _strategy_agent(day).decide(obs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    if args.tune:
        optimise(EvalConfig(team_name=TEAM))
        return

    if args.llm:
        _prepare_llm_env()
    regime = LLMRegime() if args.llm else HeuristicRegime()
    replay = None if args.no_replay else ReplayStore()
    try:
        play_game(TEAM, args.scenario, args.seed, params=Params(),
                  regime=regime, replay=replay, verbose=True)
    finally:
        if replay:
            replay.close()


if __name__ == "__main__":
    main()
