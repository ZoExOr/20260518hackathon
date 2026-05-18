"""Tuning harness — searches Params under a ROBUST, sample-bounded objective.

OWNER: tuning workstream.

Hard reality: ~60 games/hour, a few hundred episodes total INCLUDING
debugging. So:
  * Optimise a robust objective (mean - lambda*std, or min) across a SMALL
    set of (scenario, seed) pairs — never tune toward one seed.
  * Prefer Bayesian optimisation (50-150 evals) over CMA-ES (400-600).
  * Track a global eval budget and stop when it's exhausted.

`evaluate_params` is the objective. The optimiser driver is a thin wrapper;
swap scikit-optimize / Optuna / cma as the team prefers.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..params import Params
from ..runner import play_game


@dataclass
class EvalConfig:
    team_name: str
    scenarios: tuple[str, ...] = ("baseline", "supply_crisis")
    seeds: tuple[int, ...] = (42, 88)
    robustness_lambda: float = 1.0   # penalise variance across runs
    max_evals: int = 80              # hard cap on optimiser iterations


def evaluate_params(params: Params, cfg: EvalConfig) -> float:
    """Robust score for one Params across the (scenario, seed) grid.

    Bankruptcy returns -100000 from the API, so a single collapse tanks the
    mean — exactly the behaviour we want the optimiser to avoid.
    """
    scores: list[float] = []
    for sc in cfg.scenarios:
        for sd in cfg.seeds:
            final = play_game(cfg.team_name, sc, sd, params=params,
                              verbose=False)
            scores.append(float(final.get("total_score", -100000)))
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return mean - cfg.robustness_lambda * std


def optimise(cfg: EvalConfig, base: Params | None = None) -> Params:
    """Bayesian optimisation over Params.TUNABLE within Params.bounds().

    Requires scikit-optimize (`pip install scikit-optimize`). Falls back to
    random search if it's unavailable so the harness always runs.
    """
    base = base or Params()
    bounds = Params.bounds()
    keys = list(Params.TUNABLE)
    space = [bounds[k] for k in keys]

    best_p, best_v = base, float("-inf")
    evals = {"n": 0}

    def objective(vec: list[float]) -> float:
        nonlocal best_p, best_v
        if evals["n"] >= cfg.max_evals:
            return 0.0
        evals["n"] += 1
        p = Params.from_vector(vec, base)
        v = evaluate_params(p, cfg)
        if v > best_v:
            best_v, best_p = v, p
        print(f"[tune {evals['n']:>3}/{cfg.max_evals}] score={v:,.0f} "
              f"best={best_v:,.0f}")
        return -v  # skopt minimises

    try:
        from skopt import gp_minimize
        gp_minimize(objective, space, n_calls=cfg.max_evals,
                    n_initial_points=min(15, cfg.max_evals), random_state=0)
    except ImportError:
        import random
        rng = random.Random(0)
        for _ in range(cfg.max_evals):
            objective([rng.uniform(lo, hi) for lo, hi in space])

    print(f"BEST robust score={best_v:,.0f}\n{best_p}")
    return best_p
