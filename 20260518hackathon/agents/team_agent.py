"""Entrypoint — mirrors the starter kit convention.

    python -m agents.team_agent                       # one baseline game
    python -m agents.team_agent --scenario supply_crisis --seed 88
    python -m agents.team_agent --tune                # offline BO over Params
    python -m agents.team_agent --llm                 # use LLM regime supervisor
    python -m agents.team_agent --multi-agent         # LLM advice + risk layer

Set RESTBENCH_URL (defaults to the kit's server) and, for LLM modes,
RESTBENCH_LLM_API_KEY or OPENAI_API_KEY + optional AGENT_MODEL.
"""
from __future__ import annotations

import argparse
import os

from restbench.params import Params
from restbench.replay import ReplayStore
from restbench.runner import play_game
from restbench.regime import HeuristicRegime, LLMRegime
from restbench.tuning.harness import EvalConfig, optimise
from restbench.multiagent import MultiAgentAdvisor

TEAM = os.environ.get("RESTBENCH_TEAM", "prosus-team")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--multi-agent", action="store_true",
                    help="Enable the optional five-agent LLM advice workflow")
    ap.add_argument("--no-llm-fallback", action="store_true",
                    help="Raise on LLM failures instead of falling back to heuristics")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    if args.tune:
        optimise(EvalConfig(team_name=TEAM))
        return

    regime = LLMRegime() if args.llm else HeuristicRegime()
    advisor = MultiAgentAdvisor(
        allow_fallback=not args.no_llm_fallback) if args.multi_agent else None
    replay = None if args.no_replay else ReplayStore()
    try:
        play_game(TEAM, args.scenario, args.seed, params=Params(),
                  regime=regime, replay=replay, verbose=True,
                  multi_agent=args.multi_agent,
                  multi_agent_advisor=advisor)
    finally:
        if replay:
            replay.close()


if __name__ == "__main__":
    main()
