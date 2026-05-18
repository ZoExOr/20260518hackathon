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

from .base import (Controller, earliest_delivery_day, planning_shelf_life_days,
                   supplier_for, structural_stockout, usable_inventory_kg)
from ..types import Observation, BeliefState, ProposedAction, Action
from ..params import Params


class SupplyController:
    name = "supply"

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params) -> list[ProposedAction]:
        out: list[ProposedAction] = []

        emergency = structural_stockout(obs)
        recovery = self._recovery_ingredients(obs)
        on_hand = {i["ingredient"]: usable_inventory_kg(i)
                   for i in obs.inventory}
        shelf = {i["ingredient"]: planning_shelf_life_days(i)
                 for i in obs.inventory}
        pipeline: dict[str, float] = {}
        for po in obs.pending_orders:
            pipeline[po["ingredient"]] = pipeline.get(po["ingredient"], 0.0) \
                + float(po.get("quantity_kg", 0))

        planning_dishes = self._planning_dishes(obs, belief, params)
        needed = self._ingredients_for_dishes(obs, planning_dishes)
        for ing in needed:
            usage = belief.ingredient_daily_usage.get(ing, 0.0)
            if usage <= 1e-6:
                # Early uncertainty is exactly when stockout risk is highest.
                # Seed usage from the likely widened menu, not just today's
                # active menu, so supply stays ahead of pricing/menu changes.
                usage = self._cold_start_usage(obs, ing, planning_dishes, params)
            usage = max(1e-6, usage)

            have = on_hand.get(ing, 0.0) + pipeline.get(ing, 0.0)
            cover_days = have / usage
            earliest_eta = self._earliest_eta(obs, ing)
            eta_guard = params.reorder_days + (2.0 if emergency else 0.0)
            if earliest_eta is not None:
                eta_guard = max(
                    eta_guard,
                    float(earliest_eta - obs.day) + params.delivery_guard_days,
                )
            if cover_days >= eta_guard:
                continue

            target_days = max(
                params.target_days + params.safety_days + (2.0 if emergency else 0.0),
                eta_guard + params.safety_days,
            )
            target_days = min(target_days, self._endgame_target_days(obs, ing, shelf.get(ing, 14)))
            target_kg = target_days * usage
            order_qty = max(0.0, target_kg - have)
            if order_qty <= 0:
                continue

            supplier, qty, eta = self._choose_supplier(
                obs, belief, params, ing, order_qty, usage, shelf.get(ing, 14),
                emergency=emergency)
            if supplier is None:
                continue
            shortage_days = max(0.0, eta_guard - cover_days)
            priority = 100 + int(round(shortage_days * 10))
            if ing in recovery:
                priority += 35
            if shelf.get(ing, 14) <= 4:
                priority += 25
            if on_hand.get(ing, 0.0) <= 0.05:
                priority += 20
            out.append(ProposedAction(
                Action("place_order", {
                    "supplier": supplier, "ingredient": ing,
                    "quantity_kg": round(qty, 2)}),
                priority=priority,
                rationale=(
                    f"{ing}: cover {cover_days:.1f}d < trigger {eta_guard:.1f}d, "
                    f"eta day {eta}"
                )))
        return out

    # --- helpers -----------------------------------------------------------
    def _planning_dishes(self, obs: Observation, belief: BeliefState,
                         params: Params) -> list[str]:
        all_dishes = [m["name"] for m in obs.menu_book]
        if (
            obs.day <= params.bootstrap_full_menu_days
            or len(belief.ingredient_daily_usage) < 3
        ):
            return all_dishes or obs.active_menu
        return obs.active_menu or all_dishes

    def _ingredients_for_dishes(self, obs: Observation,
                                dishes: list[str]) -> set[str]:
        ings: set[str] = set()
        for dish in dishes:
            rec = obs.recipe(dish)
            if rec:
                for i in rec.get("ingredients", []):
                    ings.add(i["ingredient"])
        return ings

    def _recovery_ingredients(self, obs: Observation) -> set[str]:
        """Ingredients whose replenishment helps restore at least five dishes."""
        usable = {i["ingredient"]: usable_inventory_kg(i) for i in obs.inventory}
        serviceable = 0
        blockers: dict[str, int] = {}
        for dish in obs.active_menu:
            rec = obs.recipe(dish)
            if not rec:
                continue
            missing = [
                i["ingredient"] for i in rec.get("ingredients", [])
                if usable.get(i["ingredient"], 0.0) < float(i.get("quantity_kg", 0) or 0)
            ]
            if not missing:
                serviceable += 1
            for ing in missing:
                blockers[ing] = blockers.get(ing, 0) + 1
        if serviceable >= 5:
            return set()
        return {
            ing for ing, count in blockers.items()
            if count >= 2 or len(blockers) <= 3
        }

    def _endgame_target_days(self, obs: Observation, ing: str,
                             shelf_life: float) -> float:
        """Avoid large late-game orders that arrive after they can help."""
        days_left = max(1.0, float(obs.days_remaining))
        cap = min(float(shelf_life), days_left + 1.0)
        # Flour/pepperoni are less perishable; keep a little more flexibility.
        if ing in ("Flour", "Pepperoni"):
            cap = min(cap + 2.0, float(shelf_life))
        return max(2.0, cap)

    def _cold_start_usage(self, obs: Observation, ing: str,
                          dishes: list[str], params: Params) -> float:
        # Conservative prior: early on we know less than we think. Provision
        # for the likely widened menu and a slightly higher-than-baseline
        # number of covers to avoid day-4 starvation before belief warms up.
        dishes = [d for d in dishes if obs.recipe(d)]
        if not dishes:
            return 1.0
        per_dish = params.cold_start_covers / len(dishes)
        kg = 0.0
        for d in dishes:
            for i in obs.recipe(d).get("ingredients", []):
                if i["ingredient"] == ing:
                    kg += i["quantity_kg"] * per_dish
        return max(0.5, kg)

    def _earliest_eta(self, obs: Observation, ing: str) -> int | None:
        etas = []
        for s in supplier_for(obs, ing):
            eta = earliest_delivery_day(
                obs.day, int(s.get("lead_time_days", 1)),
                s.get("delivery_days", []))
            if eta is not None:
                etas.append(eta)
        return min(etas) if etas else None

    def _choose_supplier(self, obs, belief, params: Params, ing: str,
                         want_kg: float, usage: float, shelf_life: float,
                         *, emergency: bool = False):
        """Cheapest supplier that delivers before we run out; reliability
        inflates the quantity, shelf life caps it (waste guard)."""
        candidates = supplier_for(obs, ing)
        best = None
        for s in candidates:
            eta = earliest_delivery_day(
                obs.day, int(s.get("lead_time_days", 1)),
                s.get("delivery_days", []))
            if eta is None:
                continue
            if eta > 30 or eta > obs.day + obs.days_remaining:
                continue
            reliab = belief.supplier(s["name"]).reliability
            inflate = 1.0 + params.reliability_inflation * \
                max(0.0, (1.0 / max(0.05, reliab)) - 1.0)
            min_order = float(s.get("min_order_kg", 0))
            qty = max(want_kg * inflate, min_order)
            # waste guard: don't hold more than shelf_life days of usage
            shelf_cap = shelf_life * usage / max(0.5, params.waste_aversion)
            if emergency:
                shelf_cap = max(shelf_cap, min_order)
            qty = min(qty, shelf_cap) if shelf_cap > 0 else qty
            if qty < min_order and not emergency:
                continue  # can't meet minimum without overstocking
            if emergency:
                qty = max(qty, min_order)
            cost = qty * s["_price"]
            current_cover = 0.0
            if usage > 0:
                on_hand = 0.0
                for inv in obs.inventory:
                    if inv.get("ingredient") == ing:
                        on_hand = usable_inventory_kg(inv)
                        break
                current_cover = on_hand / usage
            if current_cover < max(0.0, eta - obs.day + params.delivery_guard_days):
                score = (eta, cost / max(0.01, reliab))
            else:
                score = (cost / max(0.01, reliab), eta)
            if best is None or score < best[0]:
                best = (score, s["name"], qty, eta)
        if best is None:
            return None, 0.0, None
        return best[1], best[2], best[3]
