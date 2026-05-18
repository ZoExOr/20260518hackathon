"""Autoresearch-style offline loop for bounded strategy experiments.

This borrows the useful shape of karpathy/autoresearch: fixed metric, fixed
scope, explicit experiment log, and keep/discard promotion. It deliberately
does not let an LLM edit core code.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Callable

from .params import Params
from .runner import play_game


@dataclass
class ResearchCandidate:
    params_patch: dict[str, float] = field(default_factory=dict)
    prompt_patch: dict[str, str] = field(default_factory=dict)
    hypothesis: str = ""
    expected_metric_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchResult:
    tag: str
    score: float
    mean: float
    std: float
    status: str
    candidate: ResearchCandidate


Evaluator = Callable[[Params, tuple[str, ...], tuple[int, ...]], list[float]]
MetricSample = float | dict[str, float]
SupplyEvaluator = Callable[
    [Params, tuple[str, ...], tuple[int, ...]], list[MetricSample]
]


def _extract_total_score(final: dict) -> float:
    if isinstance(final.get("score"), dict):
        return float(final["score"].get("total_score", -100000))
    return float(final.get("total_score", -100000))


def live_evaluator(team_name: str, *, multi_agent: bool = False) -> Evaluator:
    def evaluate(params: Params, scenarios: tuple[str, ...],
                 seeds: tuple[int, ...]) -> list[float]:
        scores = []
        for scenario in scenarios:
            for seed in seeds:
                final = play_game(
                    team_name, scenario, seed, params=params,
                    verbose=False, multi_agent=multi_agent)
                scores.append(_extract_total_score(final))
        return scores
    return evaluate


class ResearchAgent:
    def __init__(self, run_dir: str = "research_runs",
                 robustness_lambda: float = 1.0,
                 replay_pattern: str = "replays/*.jsonl"):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frontier_path = self.dir / "frontier.json"
        self.results_path = self.dir / "results.tsv"
        self.robustness_lambda = robustness_lambda
        self.replay_pattern = replay_pattern
        if not self.results_path.exists():
            self.results_path.write_text(
                "tag\tscore\tmean\tstd\tstatus\tparams_patch\thypothesis\n",
                encoding="utf-8",
            )

    def load_frontier(self) -> dict[str, Any]:
        if not self.frontier_path.exists():
            return {
                "best_score": float("-inf"),
                "params": asdict(Params()),
                "prompt_patch": {},
                "history": [],
            }
        return json.loads(self.frontier_path.read_text(encoding="utf-8"))

    def _base_params(self, frontier: dict[str, Any]) -> Params:
        data = frontier.get("params", {}) or {}
        valid = {k: v for k, v in data.items() if k in Params.__dataclass_fields__}
        return Params(**valid)

    def propose_candidate(self, frontier: dict[str, Any]) -> ResearchCandidate:
        signals = self._replay_signals()
        if signals["stockout_days"] >= max(2, signals["turns"] // 8):
            return ResearchCandidate(
                params_patch={"target_days": 8.5, "safety_days": 2.5},
                hypothesis="Replay shows recurring stockouts; test more cover.",
                expected_metric_change=0.0,
            )
        if signals["waste_cost"] > 500:
            return ResearchCandidate(
                params_patch={"waste_aversion": 1.5, "target_days": 6.0},
                hypothesis="Replay shows high waste; test smaller fresh inventory.",
                expected_metric_change=0.0,
            )
        if signals["walkout_pressure"] >= 2:
            return ResearchCandidate(
                params_patch={"covers_per_staff": 14.0},
                hypothesis="Replay shows walkout pressure; test higher staffing.",
                expected_metric_change=0.0,
            )
        if signals["low_cash_days"] >= 2:
            return ResearchCandidate(
                params_patch={"cash_reserve_floor": 3500.0,
                              "max_order_cash_frac": 0.28},
                hypothesis="Replay shows low-cash risk; test tighter cash guard.",
                expected_metric_change=0.0,
            )

        history = frontier.get("history", []) or []
        step = len(history) % 5
        if step == 0:
            patch = {"target_days": 8.5, "safety_days": 2.5}
            hypothesis = "Hold more cover to reduce hidden-scenario stockouts."
        elif step == 1:
            patch = {"waste_aversion": 1.4, "target_days": 6.5}
            hypothesis = "Shrink fresh overstock to reduce waste penalties."
        elif step == 2:
            patch = {"covers_per_staff": 14.0}
            hypothesis = "Slightly higher staffing protects reputation and walkouts."
        elif step == 3:
            patch = {"base_price_mult": 1.05, "marketing_slump": 250.0}
            hypothesis = "Small price lift plus slump marketing may improve profit."
        else:
            patch = {"cash_reserve_floor": 3000.0, "max_order_cash_frac": 0.3}
            hypothesis = "Tighter cash guard improves survival robustness."
        return ResearchCandidate(
            params_patch=patch,
            prompt_patch={},
            hypothesis=hypothesis,
            expected_metric_change=0.0,
        )

    def _replay_signals(self) -> dict[str, float]:
        signals = {
            "turns": 0,
            "stockout_days": 0,
            "waste_cost": 0.0,
            "walkout_pressure": 0,
            "low_cash_days": 0,
        }
        for raw_path in sorted(glob(self.replay_pattern))[-5:]:
            path = Path(raw_path)
            try:
                rows = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in rows:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "final_score":
                    continue
                signals["turns"] += 1
                obs = row.get("observation", {}) or {}
                day_result = row.get("day_result", {}) or {}
                if day_result.get("dishes_unavailable_at"):
                    signals["stockout_days"] += 1
                costs = obs.get("cost_breakdown", {}) or {}
                signals["waste_cost"] += float(costs.get("waste", 0) or 0)
                if day_result.get("walkout_band") in ("Some", "Many"):
                    signals["walkout_pressure"] += 1
                if float(obs.get("cash", 0) or 0) < 3500:
                    signals["low_cash_days"] += 1
        return signals

    def _candidate_params(self, base: Params,
                          candidate: ResearchCandidate) -> Params:
        data = asdict(base)
        bounds = Params.bounds()
        for key, value in candidate.params_patch.items():
            if key not in data:
                continue
            if key in bounds:
                lo, hi = bounds[key]
                value = max(lo, min(hi, float(value)))
            data[key] = value
        return Params(**{
            k: v for k, v in data.items() if k in Params.__dataclass_fields__
        })

    def evaluate_candidate(self, candidate: ResearchCandidate,
                           evaluator: Evaluator,
                           scenarios: tuple[str, ...],
                           seeds: tuple[int, ...]) -> tuple[float, float, float]:
        frontier = self.load_frontier()
        params = self._candidate_params(self._base_params(frontier), candidate)
        scores = evaluator(params, scenarios, seeds)
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        return mean - self.robustness_lambda * std, mean, std

    def run_once(self, evaluator: Evaluator, scenarios: tuple[str, ...],
                 seeds: tuple[int, ...],
                 candidate: ResearchCandidate | None = None) -> ResearchResult:
        frontier = self.load_frontier()
        candidate = candidate or self.propose_candidate(frontier)
        tag = f"r{int(time.time())}"
        score, mean, std = self.evaluate_candidate(
            candidate, evaluator, scenarios, seeds)
        best = float(frontier.get("best_score", float("-inf")))
        status = "keep" if score > best else "discard"
        self._append_result(tag, score, mean, std, status, candidate)

        if status == "keep":
            params = self._candidate_params(self._base_params(frontier), candidate)
            history = list(frontier.get("history", []) or [])
            history.append({
                "tag": tag,
                "score": score,
                "mean": mean,
                "std": std,
                "candidate": candidate.to_dict(),
            })
            self.frontier_path.write_text(json.dumps({
                "best_score": score,
                "params": asdict(params),
                "prompt_patch": candidate.prompt_patch,
                "history": history,
            }, indent=2, sort_keys=True), encoding="utf-8")

        return ResearchResult(tag, score, mean, std, status, candidate)

    def _append_result(self, tag: str, score: float, mean: float, std: float,
                       status: str, candidate: ResearchCandidate) -> None:
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{tag}\t{score:.3f}\t{mean:.3f}\t{std:.3f}\t{status}\t"
                f"{json.dumps(candidate.params_patch, sort_keys=True)}\t"
                f"{candidate.hypothesis}\n"
            )


class SupplyResearchAgent(ResearchAgent):
    """Supply-scoped autoresearch with a stockout/waste-aware objective."""

    def __init__(self, run_dir: str = "research_runs",
                 robustness_lambda: float = 0.8,
                 replay_pattern: str = "replays/*.jsonl"):
        super().__init__(
            run_dir=run_dir,
            robustness_lambda=robustness_lambda,
            replay_pattern=replay_pattern,
        )
        self.frontier_path = self.dir / "supply_frontier.json"
        self.results_path = self.dir / "supply_results.tsv"
        if not self.results_path.exists():
            self.results_path.write_text(
                "tag\tscore\tmean\tstd\tstockout_days\twaste_cost\tstatus\t"
                "params_patch\tprompt_patch\thypothesis\n",
                encoding="utf-8",
            )

    def propose_candidate(self, frontier: dict[str, Any]) -> ResearchCandidate:
        signals = self._replay_signals()
        history = frontier.get("history", []) or []
        if signals["stockout_days"] >= max(1, signals["turns"] // 10):
            return ResearchCandidate(
                params_patch={
                    "reorder_days": 4.5,
                    "target_days": 9.0,
                    "safety_days": 3.0,
                    "reliability_inflation": 1.7,
                },
                prompt_patch={
                    "supply": "Prefer fastest ETA over cheapest supplier when cover_days < ETA guard."
                },
                hypothesis="Supply replay shows stockouts; test faster defensive replenishment.",
            )
        if signals["waste_cost"] > 500:
            return ResearchCandidate(
                params_patch={"target_days": 6.0, "waste_aversion": 1.7},
                prompt_patch={
                    "supply": "Cap fresh orders to shelf-life cover unless risk is critical."
                },
                hypothesis="Supply replay shows waste; test tighter fresh inventory sizing.",
            )
        step = len(history) % 4
        if step == 0:
            patch = {"reorder_days": 4.0, "target_days": 8.5, "safety_days": 2.75}
            prompt = {"supply": "During renovation recovery, restore a 5-6 dish menu first."}
            hyp = "Renovation recovery may need a larger recovery buffer."
        elif step == 1:
            patch = {"reliability_inflation": 1.8, "max_order_cash_frac": 0.42}
            prompt = {"supply": "Under supplier alerts, diversify and inflate unreliable suppliers."}
            hyp = "Supply crisis may reward reliability-weighted ordering."
        elif step == 2:
            patch = {"target_days": 5.5, "safety_days": 1.0, "waste_aversion": 1.8}
            prompt = {"supply": "In endgame, trim late fresh orders aggressively."}
            hyp = "Endgame trim can reduce waste and cash drag."
        else:
            patch = {"reorder_days": 3.75, "target_days": 7.5, "safety_days": 2.25}
            prompt = {"supply": "Use balanced coverage unless serviceable_dish_count < 5."}
            hyp = "Balanced coverage may preserve baseline profit while preventing stockouts."
        return ResearchCandidate(
            params_patch=patch,
            prompt_patch=prompt,
            hypothesis=hyp,
        )

    def evaluate_candidate(self, candidate: ResearchCandidate,
                           evaluator: SupplyEvaluator,
                           scenarios: tuple[str, ...],
                           seeds: tuple[int, ...]) -> tuple[float, float, float]:
        params = self._candidate_params(self._base_params(self.load_frontier()), candidate)
        samples = evaluator(params, scenarios, seeds)
        scores, stockouts, waste = self._supply_metrics(samples)
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        objective = (
            mean
            - self.robustness_lambda * std
            - 300.0 * stockouts
            - 0.5 * waste
        )
        return objective, mean, std

    def run_once(self, evaluator: SupplyEvaluator, scenarios: tuple[str, ...],
                 seeds: tuple[int, ...],
                 candidate: ResearchCandidate | None = None) -> ResearchResult:
        frontier = self.load_frontier()
        candidate = candidate or self.propose_candidate(frontier)
        tag = f"s{int(time.time())}"
        params = self._candidate_params(self._base_params(frontier), candidate)
        samples = evaluator(params, scenarios, seeds)
        scores, stockouts, waste = self._supply_metrics(samples)
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        score = (
            mean
            - self.robustness_lambda * std
            - 300.0 * stockouts
            - 0.5 * waste
        )
        best = float(frontier.get("best_score", float("-inf")))
        status = "keep" if score > best else "discard"
        self._append_supply_result(
            tag, score, mean, std, stockouts, waste, status, candidate)

        if status == "keep":
            history = list(frontier.get("history", []) or [])
            history.append({
                "tag": tag,
                "score": score,
                "mean": mean,
                "std": std,
                "stockout_days": stockouts,
                "waste_cost": waste,
                "candidate": candidate.to_dict(),
            })
            self.frontier_path.write_text(json.dumps({
                "best_score": score,
                "params": asdict(params),
                "prompt_patch": candidate.prompt_patch,
                "history": history,
            }, indent=2, sort_keys=True), encoding="utf-8")

        return ResearchResult(tag, score, mean, std, status, candidate)

    def _supply_metrics(self, samples: list[MetricSample]
                        ) -> tuple[list[float], float, float]:
        scores = []
        stockouts = 0.0
        waste = 0.0
        for sample in samples:
            if isinstance(sample, dict):
                scores.append(float(sample.get("score", sample.get("total_score", 0))))
                stockouts += float(sample.get("stockout_days", 0) or 0)
                waste += float(sample.get("waste_cost", 0) or 0)
            else:
                scores.append(float(sample))
        return scores or [float("-inf")], stockouts, waste

    def _append_supply_result(self, tag: str, score: float, mean: float,
                              std: float, stockouts: float, waste: float,
                              status: str,
                              candidate: ResearchCandidate) -> None:
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{tag}\t{score:.3f}\t{mean:.3f}\t{std:.3f}\t"
                f"{stockouts:.3f}\t{waste:.3f}\t{status}\t"
                f"{json.dumps(candidate.params_patch, sort_keys=True)}\t"
                f"{json.dumps(candidate.prompt_patch, sort_keys=True)}\t"
                f"{candidate.hypothesis}\n"
            )
