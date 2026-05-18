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

from .types import Observation, Action, BeliefState, ProposedAction
from .params import Params
from .belief import BeliefEstimator
from .controllers import SupplyController, OperationsController, PricingController
from .composer import Coordinator
from .safety import SafetyGate
from .regime import RegimeSupervisor, HeuristicRegime
from .multiagent import (MultiAgentAdvisor, MultiAgentTrace,
                         advice_to_proposals)
from .risk_gate import DeterministicRiskGate


class Agent:
    def __init__(self, params: Params | None = None,
                 regime: RegimeSupervisor | None = None,
                 multi_agent: bool = False,
                 multi_agent_advisor: MultiAgentAdvisor | None = None):
        self.params = params or Params()
        self.belief_est = BeliefEstimator()
        self.regime = regime or HeuristicRegime()
        self.supply = SupplyController()
        self.ops = OperationsController(
            forecaster=self.belief_est.forecast_covers)
        self.pricing = PricingController()
        self.composer = Coordinator()
        self.safety = SafetyGate()
        self.risk_gate = DeterministicRiskGate()
        self.multi_agent = multi_agent
        self.multi_agent_advisor = multi_agent_advisor or (
            MultiAgentAdvisor() if multi_agent else None
        )
        self.last_multiagent_trace: MultiAgentTrace | None = None
        self.last_deterministic_risk: list[dict] = []

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

        self.last_multiagent_trace = None
        if self.multi_agent and self.multi_agent_advisor is not None:
            trace = MultiAgentTrace()
            advice, errors = self.multi_agent_advisor.collect_advice(
                obs, belief, p, proposals)
            trace.advice = advice
            trace.errors.extend(errors)
            overrides = {}
            for item in advice:
                overrides.update({
                    k: v for k, v in item.param_overrides.items()
                    if hasattr(p, k)
                })
                proposals.setdefault(item.role, []).extend(
                    advice_to_proposals(item))
            if overrides:
                p = replace(p, **overrides)
            self.last_multiagent_trace = trace

        composed = self.composer.compose(proposals, obs, belief)
        self.last_deterministic_risk = []
        composed, hard_report = self.risk_gate.filter(composed, obs, belief, p)
        self.last_deterministic_risk.append(hard_report.to_dict())

        if self.multi_agent and self.multi_agent_advisor is not None:
            trace = self.last_multiagent_trace or MultiAgentTrace()
            trace.deterministic_risk.extend(self.last_deterministic_risk)
            review, errors = self.multi_agent_advisor.review(
                obs, belief, p, composed)
            trace.errors.extend(errors)
            if review is not None:
                trace.risk_review = review
                if review.approved_actions or not composed:
                    composed = [
                        ProposedAction(a, priority=90,
                                       rationale="llm:risk approved")
                        for a in review.approved_actions
                    ]
                    composed, post_report = self.risk_gate.filter(
                        composed, obs, belief, p)
                    trace.deterministic_risk.append(post_report.to_dict())
                    self.last_deterministic_risk.append(post_report.to_dict())
            self.last_multiagent_trace = trace

        actions = self.safety.filter(composed, obs, belief, p)

        # Persist a compact crumb for LLM-context continuity (Python-side
        # memory is already carried in `belief`; this is only for `notes`).
        actions.append(Action("save_notes", {"text": (
            f"d{obs.day} mode={mode.value} rep~{belief.reputation_est:.1f} "
            f"cash={obs.cash:.0f}")[:4000]}))
        return actions
