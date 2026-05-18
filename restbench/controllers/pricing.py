"""PricingController — menu, prices, marketing, happy hour, daily special.

OWNER: demand workstream.

Baseline keeps a wide menu (variety drives demand), a near-base global
price multiplier, and uses marketing / happy hour / specials reactively
when demand is weak. Price elasticity is real and asymmetric — the offline
analysis (tuning/analyze.py) estimates it from replays; this controller just
applies `params.base_price_mult` until that lands.

Note the menu-conflict contract: BOTH this controller and supply care about
the menu. Resolution rule = the composer keeps the WIDEST safe menu and
drops only dishes whose ingredients are structurally unavailable. So this
controller proposes the *desired* menu; supply may veto specific dishes via
a lower-priority "drop dish" proposal. See composer.py.
"""
from __future__ import annotations

from .base import Controller
from .base import Controller
from ..types import Observation, BeliefState, ProposedAction, Action
from ..params import Params


class PricingController:
    name = "pricing"

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params,
                posture: str = "hold") -> list[ProposedAction]:
        """`posture` comes from DemandAdvisor (LLM-picked) or the regime
        fallback. Maps to a recipe of marketing / happy_hour / special
        without ever asking the LLM for a number."""
        out: list[ProposedAction] = []

        # 1) Menu: keep all dishes whose ingredients we could plausibly stock.
        #    (Variety matters; never silently shrink below 5.)
        all_dishes = [m["name"] for m in obs.menu_book]
        desired = all_dishes if len(all_dishes) >= 5 else obs.active_menu
        if set(desired) != set(obs.active_menu) and len(desired) >= 5:
            out.append(ProposedAction(
                Action("set_menu", {"dishes": desired}),
                priority=20, rationale="maximise variety"))

        # 2) Prices: apply the global multiplier within the 0.8-1.2 band.
        #    Posture "push" allows mult up to +5%; "discount" allows -5%.
        mult = params.base_price_mult
        if posture == "push":
            mult = min(1.05, mult * 1.03)
        elif posture == "discount":
            mult = max(0.95, mult * 0.97)
        if abs(mult - 1.0) > 1e-3:
            for m in obs.menu_book:
                if not m.get("is_active"):
                    continue
                base = m["base_price"]
                price = round(min(1.2, max(0.8, mult)) * base, 2)
                if abs(price - m.get("current_price", base)) > 0.01:
                    out.append(ProposedAction(
                        Action("set_price", {"dish": m["name"], "price": price}),
                        priority=20,
                        rationale=f"posture={posture} x{mult:.2f}"))

        # 3) Marketing / happy hour / special — reactive to weak demand.
        trend = obs.customer_trend
        weak = trend == "Declining" or belief.reputation_trajectory == "Falling"
        spend = params.marketing_slump if weak else params.marketing_default
        spend = max(0.0, min(500.0, spend))
        out.append(ProposedAction(
            Action("set_marketing_spend", {"amount": spend}),
            priority=10, rationale=f"trend={trend}"))

        if weak and params.happy_hour_on_weak_days:
            out.append(ProposedAction(
                Action("run_happy_hour", {}), priority=10,
                rationale="discount posture -> happy hour"))

        # Daily special: cheap satisfaction bonus; rotate a popular dish.
        sold = obs.service_summary.get("dishes_sold", {}) or {}
        if obs.active_menu:
            pick = max(obs.active_menu,
                       key=lambda d: sold.get(d, 0)) if sold else obs.active_menu[0]
            out.append(ProposedAction(
                Action("offer_daily_special", {"dish": pick}),
                priority=10, rationale="satisfaction bonus"))
        return out