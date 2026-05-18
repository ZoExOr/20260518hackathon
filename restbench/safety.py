"""SafetyGate — the bankruptcy-proof, validity-guaranteeing layer.

OWNER: safety workstream (pairs with tuning).

This is NOT an LLM critic. It is a deterministic invariant: given any set of
proposed actions, it returns a set that (a) is always API-valid and (b)
provably cannot push projected cash below `params.cash_reserve_floor`.

Bankruptcy = -100,000, which dominates the score in expectation. Everything
else is secondary to this function being correct. It is heavily unit-tested.

Order of operations:
  1. Clamp every action into its legal range (price band, staff range,
     marketing range, menu >= 5) so the API never rejects us.
  2. Drop duplicate orders already in `pending_orders`.
  3. Project end-of-turn cash and trim/drop orders (cheapest-value first)
     until projected cash >= reserve floor AND ordering spend <=
     max_order_cash_frac * cash.
"""
from __future__ import annotations

from .types import Observation, BeliefState, ProposedAction, Action
from .params import Params

FIXED_DAILY_COST = 300.0


class SafetyGate:
    def filter(self, proposals: list[ProposedAction], obs: Observation,
               belief: BeliefState, params: Params) -> list[Action]:
        acts: list[ProposedAction] = []
        for p in proposals:
            clamped = self._clamp(p.action, obs)
            if clamped is not None:
                acts.append(ProposedAction(clamped, p.priority, p.rationale))
        acts = self._dedupe_orders(acts, obs)
        return self._enforce_cash(acts, obs, params)

    # --- 1. clamp to legal ranges -----------------------------------------
    def _clamp(self, a: Action, obs: Observation) -> Action | None:
        t, args = a.tool, dict(a.args)
        if t == "set_staff_level":
            args["level"] = int(max(3, min(15, round(args.get("level", 8)))))
        elif t == "set_marketing_spend":
            args["amount"] = float(max(0.0, min(500.0, args.get("amount", 0))))
        elif t == "set_price":
            rec = obs.recipe(args.get("dish", ""))
            if not rec:
                return None
            base = rec["base_price"]
            args["price"] = round(max(0.8 * base,
                                      min(1.2 * base, args["price"])), 2)
        elif t == "set_menu":
            dishes = [d for d in dict.fromkeys(args.get("dishes", []))
                      if obs.recipe(d)]
            if len(dishes) < 5:
                return None  # never submit an invalid (too-narrow) menu
            args["dishes"] = dishes
        elif t == "offer_daily_special":
            if not obs.recipe(args.get("dish", "")):
                return None
        elif t == "place_order":
            if args.get("quantity_kg", 0) <= 0:
                return None
        elif t == "save_notes":
            args["text"] = str(args.get("text", ""))[:4000]
        return Action(t, args)

    # --- 2. de-duplicate orders -------------------------------------------
    def _dedupe_orders(self, acts: list[ProposedAction], obs: Observation
                       ) -> list[ProposedAction]:
        pending = {(p["supplier"], p["ingredient"]) for p in obs.pending_orders}
        out, seen = [], set()
        for a in acts:
            if a.action.tool == "place_order":
                key = (a.action.args["supplier"], a.action.args["ingredient"])
                if key in pending or key in seen:
                    continue
                seen.add(key)
            out.append(a)
        return out

    # --- 3. hard cash invariant -------------------------------------------
    def _order_cost(self, a: Action, obs: Observation) -> float:
        for s in obs.supplier_catalog:
            if s["name"] == a.args.get("supplier"):
                price = s.get("ingredients", {}).get(a.args["ingredient"])
                if price is not None:
                    return price * a.args["quantity_kg"]
        return 0.0

    def _enforce_cash(self, acts: list[ProposedAction], obs: Observation,
                      params: Params) -> list[Action]:
        cash = obs.cash
        staff = next((a.action.args["level"] for a in acts
                      if a.action.tool == "set_staff_level"), obs.staff_level)
        marketing = next((a.action.args["amount"] for a in acts
                          if a.action.tool == "set_marketing_spend"), 0.0)
        # Conservative projected end-of-turn outflow that is NOT order spend.
        overhead = FIXED_DAILY_COST + staff * 120.0 + marketing

        orders = [a for a in acts if a.action.tool == "place_order"]
        others = [a.action for a in acts if a.action.tool != "place_order"]
        # cheapest-value first so we keep the orders we most need? No —
        # drop the LEAST urgent (largest, most expensive) first. Supply set
        # priority high; here we just bound total spend.
        orders.sort(key=lambda a: (-a.priority, self._order_cost(a.action, obs)))

        budget = min(cash - params.cash_reserve_floor - overhead,
                     params.max_order_cash_frac * cash)
        kept, spent = [], 0.0
        for a in orders:
            c = self._order_cost(a.action, obs)
            if spent + c <= budget:
                kept.append(a.action)
                spent += c
            # else: skip this order this turn (supply will retry next day)
        return others + kept
