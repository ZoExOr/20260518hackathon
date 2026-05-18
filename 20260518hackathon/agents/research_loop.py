"""Offline autoresearch-style loop for bounded RestBench strategy tuning.

Usage:
    python -m agents.research_loop --budget 5 --scenarios baseline,supply_crisis --seeds 42,88
"""
from __future__ import annotations

import argparse
import os

from restbench.autoresearch import ResearchAgent, SupplyResearchAgent, live_evaluator

TEAM = os.environ.get("RESTBENCH_TEAM", "prosus-team-research")


def _split_csv(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _split_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1,
                        help="Number of candidate experiments to run")
    parser.add_argument("--scenarios", default="baseline,supply_crisis")
    parser.add_argument("--seeds", default="42,88")
    parser.add_argument("--run-dir", default="research_runs")
    parser.add_argument("--lambda", dest="robustness_lambda",
                        type=float, default=1.0)
    parser.add_argument("--multi-agent", action="store_true",
                        help="Evaluate candidates with the multi-agent runtime enabled")
    parser.add_argument("--supply-only", action="store_true",
                        help="Use the supply-scoped frontier/objective")
    args = parser.parse_args()

    cls = SupplyResearchAgent if args.supply_only else ResearchAgent
    agent = cls(
        run_dir=args.run_dir,
        robustness_lambda=args.robustness_lambda,
    )
    evaluator = live_evaluator(TEAM, multi_agent=args.multi_agent)
    scenarios = _split_csv(args.scenarios)
    seeds = _split_int_csv(args.seeds)

    for i in range(args.budget):
        result = agent.run_once(evaluator, scenarios, seeds)
        print(
            f"[{i + 1}/{args.budget}] {result.status} "
            f"score={result.score:,.0f} mean={result.mean:,.0f} "
            f"std={result.std:,.0f} hypothesis={result.candidate.hypothesis}"
        )


if __name__ == "__main__":
    main()
