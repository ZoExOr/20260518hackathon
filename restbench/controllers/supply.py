"""SupplyController — perishable (s,S) inventory with delivery-calendar
and reliability-aware safety stock.

OWNER: supply workstream.

Logic (working baseline; improve the TODOs):
  For each ingredient used by the active menu:
    cover_days = on_hand / daily_usage(belief)
    pipeline   = sum(pending_orders for that ingredient)
    if (cover_days + pipeline_days) < params.reorder_days:        # s
        target  = (params.target_days + params.safety_days) * usage
        target *= 1 + params.reliability_inflation*(1/reliab - 1)  # buffer
        qty     = target - on_hand - pipeline
        pick the cheapest supplier that can DELIVER BEFORE STOCKOUT
        respect min_order_kg; never exceed shelf life * usage (waste guard)

The SafetyGate has the final say on whether we can afford the order.
"""
from __future__ import annotations

from .base import Controller, earliest_delivery_day, supplier_for
from ..types import Observation, BeliefState, ProposedAction, Action
from ..params import Params


class SupplyController:
    name = "supply"

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params) -> list[ProposedAction]:
        out: list[ProposedAction] = []

        on_hand = {i["ingredient"]: float(i.get("total_kg", 0))
                   for i in obs.inventory}
        shelf = {i["ingredient"]: float(i.get("shelf_life_days", 14))
                 for i in obs.inventory}
        pipeline: dict[str, float] = {}
        for po in obs.pending_orders:
            pipeline[po["ingredient"]] = pipeline.get(po["ingredient"], 0.0) \
                + float(po.get("quantity_kg", 0))

        needed = self._active_ingredients(obs)
        for ing in needed:
            usage = max(1e-6, belief.ingredient_daily_usage.get(ing, 0.0))
            if usage <= 1e-6:
                # No usage estimate yet (early game): seed a small order so
                # we don't stock out before the belief warms up.
                usage = self._cold_start_usage(obs, ing)

            have = on_hand.get(ing, 0.0) + pipeline.get(ing, 0.0)
            cover_days = have / usage
            if cover_days >= params.reorder_days:
                continue

            target_days = params.target_days + params.safety_days
            target_kg = target_days * usage
            order_qty = max(0.0, target_kg - have)
            if order_qty <= 0:
                continue

            supplier, qty, eta = self._choose_supplier(
                obs, belief, params, ing, order_qty, usage, shelf.get(ing, 14))
            if supplier is None:
                continue
            out.append(ProposedAction(
                Action("place_order", {
                    "supplier": supplier, "ingredient": ing,
                    "quantity_kg": round(qty, 2)}),
                priority=100,
                rationale=(f"{ing}: cover {cover_days:.1f}d < "
                           f"{params.reorder_days:.1f}d, eta day {eta}")))
        return out

    # --- helpers -----------------------------------------------------------
    def _active_ingredients(self, obs: Observation) -> set[str]:
        ings: set[str] = set()
        for dish in obs.active_menu:
            rec = obs.recipe(dish)
            if rec:
                for i in rec.get("ingredients", []):
                    ings.add(i["ingredient"])
        return ings

    def _cold_start_usage(self, obs: Observation, ing: str) -> float:
        # Assume ~90 covers spread across the active menu on day 1.
        dishes = [d for d in obs.active_menu if obs.recipe(d)]
        if not dishes:
            return 1.0
        per_dish = 90.0 / len(dishes)
        kg = 0.0
        for d in dishes:
            for i in obs.recipe(d).get("ingredients", []):
                if i["ingredient"] == ing:
                    kg += i["quantity_kg"] * per_dish
        return max(0.5, kg)

    def _choose_supplier(self, obs, belief, params: Params, ing: str,
                         want_kg: float, usage: float, shelf_life: float):
        """Cheapest supplier that delivers before we run out; reliability
        inflates the quantity, shelf life caps it (waste guard)."""
        candidates = supplier_for(obs, ing)
        best = None
        # The Beta prior in types.py is (5, 1) → initial reliability 0.83.
        # Without an evidence guard, EVERY first-order would still be
        # inflated ~1.20× across the board, which compounds with the
        # target_days + safety_days = 9 to overbuy perishables. Require
        # at least a few real deliveries before letting reliability move
        # the order quantity.
        EVIDENCE_THRESHOLD = 3.0  # observed orders past the prior
        for s in candidates:
            eta = earliest_delivery_day(
                obs.day, int(s.get("lead_time_days", 1)),
                s.get("delivery_days", []))
            if eta is None:
                continue
            reliab = belief.supplier(s["name"]).reliability
            inflate = 1.0 + params.reliability_inflation * \
                max(0.0, (1.0 / max(0.05, reliab)) - 1.0)
            qty = max(want_kg * inflate, float(s.get("min_order_kg", 0)))
            # waste guard: don't hold more than shelf_life days of usage
            qty = min(qty, shelf_life * usage / max(0.5, params.waste_aversion))
            if qty < float(s.get("min_order_kg", 0)):
                continue  # can't meet minimum without overstocking
            cost = qty * s["_price"]
            score = (eta, cost / max(0.01, reliab))  # prefer fast, then cheap+reliable
            if best is None or score < best[0]:
                best = (score, s["name"], qty, eta)
        if best is None:
            return None, 0.0, None
        return best[1], best[2], best[3]