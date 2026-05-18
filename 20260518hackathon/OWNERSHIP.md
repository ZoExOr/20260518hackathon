# Ownership, interfaces & build order

This scaffold turns the "multi-agent" idea into a **module map for parallel
work**, not a runtime swarm of LLMs. The decomposition (supply / demand /
operations / research) is kept; each piece is a deterministic module behind a
typed interface so four people can work at once without merge pain.

The single source of truth is `BeliefState`. Controllers are pure functions
`(Observation, BeliefState, Params) -> list[ProposedAction]`. They never call
the network and never mutate shared state — which is what makes them testable
in isolation and safe to develop in parallel.

## The contract (do not change without the whole team)

`restbench/types.py` — `Observation`, `BeliefState`, `Action`,
`ProposedAction`, `Mode`. Everything depends on this. One owner, reviewed
changes only.

## Workstreams

### A — Supply / inventory  (`controllers/supply.py`, `controllers/base.py`)
Owns the (s,S) policy, the delivery-calendar solver, supplier choice.
Highest score leverage. `earliest_delivery_day` MUST stay correct — it has
tests; add more. Interface: `SupplyController.propose(...) -> [ProposedAction]`.

### B — Demand / pricing  (`controllers/pricing.py`, `tuning/analyze.py`)
Owns menu width, the price multiplier, marketing/happy-hour/special policy,
and the offline elasticity estimation from replays. Interface:
`PricingController.propose(...)`. Coordinate the menu-conflict rule with A
via the composer (documented in `composer.py`).

### C — Operations + Belief  (`controllers/operations.py`, `belief.py`)
Owns staffing AND the belief estimator (they share the demand forecaster).
The belief estimator is the "online research" — uncensoring demand, supplier
posteriors, reputation reconstruction. Improving `_uncensor_covers` and the
demand model is the single best non-supply task. Interfaces:
`OperationsController.propose(...)`, `BeliefEstimator.update(obs) ->
BeliefState`, `BeliefEstimator.forecast_covers(obs, offset) -> float`.

### D — Safety + Tuning + Infra  (`safety.py`, `tuning/`, `runner.py`, `api.py`)
Owns the bankruptcy invariant (most important single function in the repo),
the offline BO/CMA-ES harness, replay logging, and the API client. The
SafetyGate has tests; treat any live `[REJECTED]` log line as a P0 bug.

The optional `regime.py` (LLM supervisor, every k days) is a shared stretch
goal — ship `HeuristicRegime` first; `LLMRegime` only after the deterministic
agent reliably survives all known scenarios.

## Build order (maps to the STRATEGY_GUIDE's four levels)

1. **Survive** — `api.py` + `runner.py` + `SafetyGate` + naive controllers.
   Goal: 30 days, no bankruptcy, beat the −15k baseline. (Scaffold already
   does this with safe defaults.)
2. **Optimise** — real `BeliefEstimator` (demand uncensoring, reliability) +
   real (s,S) supply + newsvendor staffing. Goal: positive score on baseline.
3. **Adapt** — offline BO over `Params` on a 2-scenario / 2-seed subset
   (robust objective), then `HeuristicRegime` thresholds. Goal: positive
   across known scenarios.
4. **Anticipate** — replay analysis (elasticity, scenario signatures),
   optional `LLMRegime`, optional surrogate + lookahead.

## Sample-budget discipline (read this before tuning)

~60 games/hour, 10 concurrent; a few hundred episodes TOTAL including
debugging. Therefore:

- Tune on a **small (scenario, seed) subset** with a **robust objective**
  (`mean − λ·std`). Never optimise toward a single seed — evaluation uses
  hidden scenarios.
- Prefer Bayesian optimisation (50–150 evals) over CMA-ES (400–600).
- Use determinism for **debugging** (same seed reproduces exactly), not as a
  tuning target.
- Hypotheses get tested on **logged replays** (`tuning/analyze.py`), not on
  fresh live games.

## Quick start

```
pip install -r requirements.txt
export RESTBENCH_URL=http://52.48.183.209:8001
export RESTBENCH_TEAM=your-team-name
pytest -q                                  # deterministic pieces, offline
python -m agents.team_agent                # one baseline game
python -m agents.team_agent --tune         # offline BO over Params
```
