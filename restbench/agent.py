"""Agent — the decision core. Pure function of the observation stream.

    obs ->  BeliefEstimator.update
        ->  RegimeSupervisor.decide  (mode + param overrides)
        ->  controllers.propose      (supply / operations / pricing)
        ->  Coordinator.compose      (deterministic merge)
        ->  SafetyGate.filter        (bankruptcy-proof, API-valid)
        ->  list[Action]

No network here — `decide` is fully unit-testable with a fake Observation.
OWNER: integration workstream (thin; the work lives in the modules).
"""
from __future__ import annotations

from dataclasses import replace

from .types import Observation, Action, BeliefState
from .params import Params
from .belief import BeliefEstimator
from .controllers import SupplyController, OperationsController, PricingController
from .composer import Coordinator
from .safety import SafetyGate
from .regime import RegimeSupervisor, HeuristicRegime


class Agent:
    def __init__(self, params: Params | None = None,
                 regime: RegimeSupervisor | None = None):
        self.params = params or Params()
        self.belief_est = BeliefEstimator()
        self.regime = regime or HeuristicRegime()
        self.supply = SupplyController()
        self.ops = OperationsController(
            forecaster=self.belief_est.forecast_covers)
        self.pricing = PricingController()
        self.composer = Coordinator()
        self.safety = SafetyGate()

    def decide(self, obs: Observation) -> list[Action]:
        belief = self.belief_est.update(obs)

        mode, overrides = self.regime.decide(obs, belief, self.params)
        belief.mode = mode
        p = replace(self.params, **{k: v for k, v in overrides.items()
                                    if hasattr(self.params, k)})

        proposals = {
            self.supply.name: self.supply.propose(obs, belief, p),
            self.ops.name: self.ops.propose(obs, belief, p),
            self.pricing.name: self.pricing.propose(obs, belief, p),
        }
        composed = self.composer.compose(proposals, obs, belief)
        actions = self.safety.filter(composed, obs, belief, p)

        # Persist a compact crumb for LLM-context continuity (Python-side
        # memory is already carried in `belief`; this is only for `notes`).
        actions.append(Action("save_notes", {"text": (
            f"d{obs.day} mode={mode.value} rep~{belief.reputation_est:.1f} "
            f"cash={obs.cash:.0f}")[:4000]}))
        return actions
