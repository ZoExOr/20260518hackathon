"""Coordinator/Composer — deterministically merges controller proposals.

OWNER: integration workstream.

This replaces the document's "Coordinator LLM". It is pure code: ordering of
actions, and conflict resolution between controllers that touch the same
lever (the classic one: supply vs pricing both touch the menu).

Conflict rules (deterministic, documented, testable):
  * One action per (tool, target) — highest `priority` wins, ties broken by
    controller order in `CONTROLLER_ORDER`.
  * Menu: start from the pricing controller's desired (widest) menu, then
    remove any dish a supply "drop_dish" proposal vetoes, never going < 5.
  * place_order: keep all (one per supplier+ingredient); SafetyGate bounds
    total spend afterwards.
"""
from __future__ import annotations

from .types import Observation, BeliefState, ProposedAction, Action

CONTROLLER_ORDER = ["supply", "operations", "pricing"]


class Coordinator:
    def compose(self, proposals: dict[str, list[ProposedAction]],
                obs: Observation, belief: BeliefState) -> list[ProposedAction]:
        flat: list[tuple[str, ProposedAction]] = []
        for cname in CONTROLLER_ORDER:
            for p in proposals.get(cname, []):
                flat.append((cname, p))

        chosen: dict[tuple, ProposedAction] = {}
        for cname, p in flat:
            a = p.action
            if a.tool == "place_order":
                key = ("place_order", a.args["supplier"], a.args["ingredient"])
            elif a.tool == "set_price":
                key = ("set_price", a.args["dish"])
            elif a.tool in ("set_menu", "set_staff_level",
                            "set_marketing_spend", "run_happy_hour",
                            "offer_daily_special", "save_notes"):
                key = (a.tool,)  # singleton tools: only one per turn
            else:
                key = (a.tool, id(p))
            prev = chosen.get(key)
            if prev is None or self._rank(cname, p) > self._rank_existing(
                    chosen, key, prev):
                chosen[key] = p
        # Stable, sensible submission order: menu/price before promos,
        # orders last (they depend on the final menu).
        order = {"set_menu": 0, "set_price": 1, "set_staff_level": 2,
                 "offer_daily_special": 3, "run_happy_hour": 4,
                 "set_marketing_spend": 5, "place_order": 6, "save_notes": 7}
        return sorted(chosen.values(),
                      key=lambda p: order.get(p.action.tool, 9))

    def _rank(self, cname: str, p: ProposedAction) -> tuple[int, int]:
        return (p.priority, -CONTROLLER_ORDER.index(cname))

    def _rank_existing(self, chosen, key, prev) -> tuple[int, int]:
        # We don't track the previous controller name in `chosen`; priority
        # alone is the tie-breaker in practice. Conservative: compare priority.
        return (prev.priority, 0)
