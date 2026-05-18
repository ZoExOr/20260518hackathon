"""PricingController — menu, prices, marketing, happy hour, daily special.

OWNER: demand workstream.

Keeps a wide menu (variety drives demand), then adapts price and promotion
pressure to observed demand and service health. Price elasticity is real and
asymmetric — offline analysis (tuning/analyze.py) estimates priors from
replays; this controller applies a conservative state policy online.

Note the menu-conflict contract: BOTH this controller and supply care about
the menu. Resolution rule = the composer keeps the WIDEST safe menu and
drops only dishes whose ingredients are structurally unavailable. So this
controller proposes the *desired* menu; supply may veto specific dishes via
a lower-priority "drop dish" proposal. See composer.py.
"""
from __future__ import annotations

from .base import Controller, structural_stockout, usable_inventory_kg
from ..types import Observation, BeliefState, ProposedAction, Action, Mode
from ..params import Params

_WALKOUT_RISK = {"Some", "Many"}
_GOOD_REP = {"Good", "Very Good", "Excellent"}


class PricingController:
    name = "pricing"

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params) -> list[ProposedAction]:
        out: list[ProposedAction] = []
        emergency = structural_stockout(obs)

        # 1) Menu: prefer the widest *currently serviceable* menu.
        all_dishes = [m["name"] for m in obs.menu_book]
        usable = {inv["ingredient"]: usable_inventory_kg(inv) for inv in obs.inventory}
        safe = []
        for dish in all_dishes:
            rec = obs.recipe(dish)
            if not rec:
                continue
            if all(usable.get(i["ingredient"], 0.0) >= i["quantity_kg"]
                   for i in rec.get("ingredients", [])):
                safe.append(dish)
        desired = all_dishes if len(all_dishes) >= 5 else obs.active_menu
        if emergency and len(safe) >= 5:
            desired = safe
        if set(desired) != set(obs.active_menu) and len(desired) >= 5:
            out.append(ProposedAction(
                Action("set_menu", {"dishes": desired}),
                priority=20,
                rationale="structural stockout -> widest safe menu"
                if emergency else "maximise variety"))

        state = self._demand_state(obs, belief)

        # 2) Prices: state-aware multiplier, still within the 0.8-1.2 band.
        mult = self._target_price_mult(obs, belief, params, state)
        for m in obs.menu_book:
            if not m.get("is_active"):
                continue
            base = float(m["base_price"])
            price = round(min(1.2, max(0.8, mult)) * base, 2)
            if abs(price - float(m.get("current_price", base))) > 0.01:
                out.append(ProposedAction(
                    Action("set_price", {"dish": m["name"], "price": price}),
                    priority=20,
                    rationale=f"{state} demand -> x{mult:.2f}"))

        # 3) Marketing / happy hour / special — no demand stimulation during
        # structural stockout emergencies.
        spend = 0.0 if emergency else self._marketing_amount(
            obs, belief, params, state)
        out.append(ProposedAction(
            Action("set_marketing_spend", {"amount": spend}),
            priority=10,
            rationale=f"state={state}, emergency={emergency}"))

        if not emergency and self._should_run_happy_hour(
                obs, belief, params, state):
            out.append(ProposedAction(
                Action("run_happy_hour", {}), priority=10,
                rationale=f"{state} demand, service has room"))

        pick = None if emergency else self._pick_daily_special(obs)
        if pick:
            out.append(ProposedAction(
                Action("offer_daily_special", {"dish": pick}),
                priority=10, rationale="satisfaction bonus"))
        return out

    # --- state policy -----------------------------------------------------
    def _demand_state(self, obs: Observation, belief: BeliefState) -> str:
        ss = obs.service_summary
        covers = float(ss.get("total_covers", 0) or 0)
        walk = ss.get("walkout_band", "None")
        peak_wait = float(ss.get("peak_wait_minutes", 0) or 0)

        if walk in _WALKOUT_RISK or peak_wait > 18:
            return "service_risk"
        if obs.customer_trend == "Declining" or \
                belief.reputation_trajectory == "Falling":
            return "weak"
        if obs.customer_trend == "Growing" and obs.reputation_band in _GOOD_REP:
            return "strong"
        if covers >= 110 and obs.reputation_band in {"Very Good", "Excellent"}:
            return "strong"
        if 0 < covers < 60:
            return "weak"
        return "normal"

    def _target_price_mult(self, obs: Observation, belief: BeliefState,
                           params: Params, state: str) -> float:
        mult = params.base_price_mult
        if state == "strong":
            mult = max(mult, params.price_mult_good_demand)
        elif state == "service_risk":
            mult = min(mult, params.price_mult_bad_service)
        elif state == "weak":
            # Preserve demand while reputation recovers; low prices do not fix
            # operational failures by themselves.
            mult = min(mult, 1.0)

        if belief.mode == Mode.ENDGAME:
            mult = min(mult, 1.0)
        if obs.reputation_band in {"Poor", "Fair"}:
            mult = min(mult, 1.0)
        return max(0.8, min(1.2, mult))

    def _marketing_amount(self, obs: Observation, belief: BeliefState,
                          params: Params, state: str) -> float:
        if state in {"strong", "service_risk"}:
            return 0.0
        if state == "weak":
            spend = max(params.marketing_slump, params.marketing_recovery)
        elif obs.customer_trend == "Declining":
            spend = params.marketing_slump
        else:
            spend = params.marketing_default
        if belief.mode == Mode.ENDGAME:
            spend = min(spend, params.marketing_recovery * 0.5)
        return float(max(0.0, min(500.0, spend)))

    def _should_run_happy_hour(self, obs: Observation, belief: BeliefState,
                               params: Params, state: str) -> bool:
        if not params.happy_hour_on_weak_days or state != "weak":
            return False
        ss = obs.service_summary
        walk = ss.get("walkout_band", "None")
        peak_wait = float(ss.get("peak_wait_minutes", 0) or 0)
        if walk in _WALKOUT_RISK or peak_wait > params.happy_hour_max_peak_wait:
            return False
        return belief.mode != Mode.ENDGAME

    def _pick_daily_special(self, obs: Observation) -> str | None:
        active = obs.active_menu
        if not active:
            return None

        sold = obs.service_summary.get("dishes_sold", {}) or {}
        inv = {i["ingredient"]: i for i in obs.inventory}

        def score(dish: str) -> float:
            rec = obs.recipe(dish)
            if not rec:
                return -1.0
            s = float(sold.get(dish, 0))
            margin = float(rec.get("base_price", 0.0))
            availability = 0.0
            expiry_bonus = 0.0
            for item in rec.get("ingredients", []):
                ing = item["ingredient"]
                need = max(1e-6, float(item.get("quantity_kg", 0.0)))
                row = inv.get(ing, {})
                availability += float(row.get("total_kg", 0.0)) / need
                for batch in row.get("batches", []) or []:
                    if int(batch.get("expires_in_days", 99)) <= 2:
                        expiry_bonus += float(batch.get("quantity_kg", 0.0)) * need
            return s * 3.0 + min(availability, 30.0) + margin * 0.15 + expiry_bonus

        return max(active, key=score)
