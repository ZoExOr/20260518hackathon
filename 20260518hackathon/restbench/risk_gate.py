"""Deterministic hard-risk gate for strategy invariants.

This runs around the optional LLM risk reviewer. It handles rules that should
not depend on model judgment: no demand stimulation while supply is constrained,
no late endgame orders, and no low staffing after severe walkouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .controllers.base import (earliest_delivery_day, serviceability_risk,
                               usable_inventory_kg)
from .params import Params
from .types import Action, BeliefState, Observation, ProposedAction


@dataclass
class DeterministicRiskReport:
    supply_risk: dict
    dropped_actions: list[dict] = field(default_factory=list)
    adjusted_actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "supply_risk": self.supply_risk,
            "dropped_actions": self.dropped_actions,
            "adjusted_actions": self.adjusted_actions,
        }


class DeterministicRiskGate:
    def filter(self, proposals: list[ProposedAction], obs: Observation,
               belief: BeliefState, params: Params
               ) -> tuple[list[ProposedAction], DeterministicRiskReport]:
        risk = serviceability_risk(obs, belief, params)
        report = DeterministicRiskReport(supply_risk=risk.to_dict())
        out: list[ProposedAction] = []
        for proposal in proposals:
            replacement = self._filter_one(proposal, obs, risk, report)
            if replacement is not None:
                out.append(replacement)
        return out, report

    def _filter_one(self, proposal: ProposedAction, obs: Observation, risk,
                    report: DeterministicRiskReport) -> ProposedAction | None:
        action = proposal.action
        if risk.high_or_worse and action.tool == "run_happy_hour":
            report.dropped_actions.append({
                "action": action.to_payload(),
                "reason": f"supply risk {risk.level}: no demand stimulation",
            })
            return None

        if risk.high_or_worse and action.tool == "set_marketing_spend" \
                and float(action.args.get("amount", 0) or 0) > 0:
            adjusted = Action("set_marketing_spend", {"amount": 0.0})
            report.adjusted_actions.append({
                "from": action.to_payload(),
                "to": adjusted.to_payload(),
                "reason": f"supply risk {risk.level}: marketing disabled",
            })
            return ProposedAction(adjusted, proposal.priority, proposal.rationale)

        if action.tool == "set_staff_level":
            level = int(action.args.get("level", obs.staff_level))
            walk = obs.service_summary.get("walkout_band", "None")
            if not risk.critical and (
                walk == "Many" or obs.reputation_band in ("Poor", "Fair")
            ) and level < 8:
                adjusted = Action("set_staff_level", {"level": 8})
                report.adjusted_actions.append({
                    "from": action.to_payload(),
                    "to": adjusted.to_payload(),
                    "reason": "protective staffing after walkouts/reputation drop",
                })
                return ProposedAction(adjusted, proposal.priority, proposal.rationale)

        if action.tool == "place_order" and self._late_endgame_order(action, obs):
            report.dropped_actions.append({
                "action": action.to_payload(),
                "reason": "late endgame order cannot help before game end",
            })
            return None

        if action.tool == "offer_daily_special" and not self._dish_serviceable(
                str(action.args.get("dish", "")), obs):
            report.dropped_actions.append({
                "action": action.to_payload(),
                "reason": "daily special dish is not currently serviceable",
            })
            return None

        return proposal

    def _late_endgame_order(self, action: Action, obs: Observation) -> bool:
        supplier_name = action.args.get("supplier")
        ingredient = action.args.get("ingredient")
        supplier = next(
            (s for s in obs.supplier_catalog if s.get("name") == supplier_name),
            None,
        )
        if supplier is None:
            return False
        eta = earliest_delivery_day(
            obs.day,
            int(supplier.get("lead_time_days", 1)),
            supplier.get("delivery_days", []),
        )
        if eta is None:
            return True
        if eta > 30 or eta > obs.day + obs.days_remaining:
            return True
        qty = float(action.args.get("quantity_kg", 0) or 0)
        min_order = float(supplier.get("min_order_kg", 0) or 0)
        return obs.days_remaining <= 5 and eta >= 30 and qty > max(2 * min_order, 12.0) \
            and ingredient not in ("Flour", "Pepperoni")

    def _dish_serviceable(self, dish: str, obs: Observation) -> bool:
        rec = obs.recipe(dish)
        if not rec:
            return False
        usable = {inv["ingredient"]: usable_inventory_kg(inv) for inv in obs.inventory}
        return all(
            usable.get(i["ingredient"], 0.0) >= float(i.get("quantity_kg", 0) or 0)
            for i in rec.get("ingredients", []) or []
        )
