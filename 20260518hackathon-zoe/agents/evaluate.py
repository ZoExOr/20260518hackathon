"""Evaluation harness — run an agent against multiple scenarios and seeds.

Usage:
    # Run against all scenarios available on the server
    python -m agents.evaluate agents.naive_rule

    # Specific scenarios
    python -m agents.evaluate agents.naive_rule --scenarios baseline,inflation

    # Multiple seeds for robustness
    python -m agents.evaluate agents.naive_rule --seeds 42,88,123

    # Custom server
    python -m agents.evaluate agents.naive_rule --url http://eval-server:8001

    # Quiet mode (summary table only)
    python -m agents.evaluate agents.naive_rule --quiet

    # Control parallelism (default: 5, matches server limit)
    python -m agents.evaluate agents.naive_rule --parallel 3

Scenarios are fetched from the server's GET /scenarios endpoint.
The server controls which scenarios are available (admin can unlock hidden ones).
Final ranking = average score across all (scenario, seed) combinations.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from agents.runner import run_game

FALLBACK_SCENARIOS = [
    "baseline", "supply_crisis", "tourist_season",
    "inflation", "renovation", "health_scare",
]
DEFAULT_SEEDS = [42, 88, 123]


def fetch_scenarios(base_url: str) -> list[str]:
    """Fetch available scenario names from the server."""
    try:
        r = httpx.get(f"{base_url}/scenarios", timeout=10.0)
        r.raise_for_status()
        return [s["name"] for s in r.json()]
    except Exception as e:
        print(f"Warning: Could not fetch scenarios from {base_url}/scenarios: {e}")
        print("Falling back to default known scenarios.")
        return list(FALLBACK_SCENARIOS)


def load_strategy(module_path: str):
    """Dynamically import a strategy function from a dotted module path."""
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "strategy"):
        print(f"Error: {module_path} has no 'strategy' function")
        sys.exit(1)
    return mod.strategy


MAX_PARALLEL = 10


def _run_one(
    strategy,
    scenario: str,
    seed: int,
    base_url: str,
    team_name: str,
    verbose: bool,
    label: str,
) -> dict:
    if verbose:
        print(f"\n{'=' * 60}")
        print(label)
        print("=" * 60)

    try:
        result = run_game(
            strategy,
            base_url=base_url,
            team_name=team_name,
            scenario=scenario,
            seed=seed,
            verbose=verbose,
        )
        score = result["score"]["total_score"]
        return {
            "scenario": scenario,
            "seed": seed,
            "score": score,
            "days": result["days_survived"],
            "cash": result["final_cash"],
            "profit": result["score"]["net_profit"],
            "sat_pen": result["score"]["satisfaction_penalty"],
            "rep_pen": result["score"]["reputation_penalty"],
            "walk_pen": result["score"]["walkout_penalty"],
            "waste_pen": result["score"]["waste_penalty"],
            "status": result["status"],
        }
    except Exception as e:
        print(f"  FAILED ({scenario} seed={seed}): {e}")
        return {
            "scenario": scenario,
            "seed": seed,
            "score": -100_000,
            "days": 0,
            "cash": 0,
            "profit": 0,
            "sat_pen": 0,
            "rep_pen": 0,
            "walk_pen": 0,
            "waste_pen": 0,
            "status": "error",
        }


def evaluate(
    strategy,
    *,
    scenarios: list[str],
    seeds: list[int],
    base_url: str,
    team_name: str,
    verbose: bool,
    parallel: int = MAX_PARALLEL,
) -> dict:
    jobs = []
    total_games = len(scenarios) * len(seeds)
    game_num = 0
    for scenario in scenarios:
        for seed in seeds:
            game_num += 1
            label = f"[{game_num}/{total_games}] {scenario} seed={seed}"
            jobs.append((scenario, seed, label))

    results: list[dict] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_one, strategy, scenario, seed,
                base_url, team_name, verbose, label,
            ): (scenario, seed)
            for scenario, seed, label in jobs
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += 1
            print(f"  Completed {completed}/{total_games}: {r['scenario']} seed={r['seed']} → {r['score']:,.0f}")

    results.sort(key=lambda r: (r["scenario"], r["seed"]))

    scenario_totals: dict[str, list[float]] = {}
    for r in results:
        scenario_totals.setdefault(r["scenario"], []).append(r["score"])

    return {"results": results, "scenario_totals": scenario_totals}


def print_report(data: dict, team_name: str, seeds: list[int]) -> None:
    results = data["results"]
    scenario_totals = data["scenario_totals"]

    multi_seed = len(seeds) > 1

    print("\n")
    print("=" * 80)
    print(f"  EVALUATION REPORT — {team_name}")
    print("=" * 80)

    if multi_seed:
        # Per-scenario, per-seed detail table
        print(f"\n{'Scenario':<20} {'Seed':>6} {'Score':>10} {'Profit':>10} {'Walk':>7} {'Rep':>9} {'Days':>5} {'Status':<10}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['scenario']:<20} {r['seed']:>6} {r['score']:>10.0f} "
                f"{r['profit']:>10.0f} {r['walk_pen']:>7.0f} {r['rep_pen']:>9.0f} "
                f"{r['days']:>5} {r['status']:<10}"
            )

    # Per-scenario summary
    print(f"\n{'Scenario':<20} {'Avg Score':>10} {'Min':>10} {'Max':>10} {'Games':>6}")
    print("-" * 60)
    for scenario, scores in scenario_totals.items():
        avg = sum(scores) / len(scores)
        print(
            f"{scenario:<20} {avg:>10.0f} {min(scores):>10.0f} "
            f"{max(scores):>10.0f} {len(scores):>6}"
        )

    # Grand total
    all_scores = [r["score"] for r in results]
    avg_score = sum(all_scores) / len(all_scores)
    bankrupt_count = sum(1 for r in results if r["status"] == "bankrupt")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'─' * 60}")
    print(f"  Games played:    {len(results)}")
    print(f"  Bankruptcies:    {bankrupt_count}")
    if error_count:
        print(f"  Errors:          {error_count}")
    print(f"  Average score:   {avg_score:,.0f}")
    print(f"  Best scenario:   {max(scenario_totals, key=lambda s: sum(scenario_totals[s])/len(scenario_totals[s]))}")
    print(f"  Worst scenario:  {min(scenario_totals, key=lambda s: sum(scenario_totals[s])/len(scenario_totals[s]))}")
    print(f"\n  *** FINAL SCORE: {avg_score:,.0f} ***")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an agent across scenarios and seeds",
    )
    parser.add_argument(
        "agent",
        help="Dotted module path to agent (e.g., agents.naive_rule)",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario names (default: fetch from server)",
    )
    parser.add_argument(
        "--seeds",
        default="42,88,123",
        help="Comma-separated seeds (default: 42,88,123)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("RESTBENCH_URL", "http://localhost:8001"),
        help="Server URL (default: RESTBENCH_URL env var or http://localhost:8001)",
    )
    parser.add_argument(
        "--team-name",
        default=None,
        help="Team name for leaderboard (default: agent module name)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show summary table, not per-game output",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=MAX_PARALLEL,
        help=f"Max parallel games (default: {MAX_PARALLEL})",
    )
    args = parser.parse_args()

    strategy = load_strategy(args.agent)

    if args.scenarios:
        scenarios = args.scenarios.split(",")
    else:
        scenarios = fetch_scenarios(args.url)

    seeds = [int(s) for s in args.seeds.split(",")]
    team_name = args.team_name or args.agent.split(".")[-1]

    print(f"Agent:     {args.agent}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Seeds:     {', '.join(str(s) for s in seeds)}")
    print(f"Games:     {len(scenarios) * len(seeds)}")
    print(f"Server:    {args.url}")

    data = evaluate(
        strategy,
        scenarios=scenarios,
        seeds=seeds,
        base_url=args.url,
        team_name=team_name,
        verbose=not args.quiet,
        parallel=args.parallel,
    )

    print_report(data, team_name, seeds)


if __name__ == "__main__":
    main()
