"""Critic — reviews the FULL proposed action list before submission.

This is the layer that catches strategic mistakes the math checker and
safety gate can't see (e.g. "you cut staff to 4 on a Saturday").

Vetos are applied; the deterministic action list is the fallback.
NEVER lets the LLM veto a place_order for an ingredient with cover<3 days
(survival > strategy).
"""
from __future__ import annotations

from .client import LLMClient
from .prompts import CRITIC_SYSTEM, critic_user


# Inline a minimal days-of-cover helper so this module is self-contained.
# It's the only deterministic check the critic needs to do (to override
# a veto on an urgent place_order).
def _days_of_cover(obs, ingredient: str, belief) -> float:
    usage = belief.ingredient_daily_usage.get(ingredient, 0.0)
    if usage <= 0:
        return float("inf")  # no usage estimate -> don't second-guess supply
    on_hand = 0.0
    for inv in obs.inventory:
        if inv.get("ingredient") == ingredient:
            on_hand = float(inv.get("total_kg", 0))
            break
    pending = sum(float(p.get("quantity_kg", 0))
                  for p in obs.pending_orders
                  if p.get("ingredient") == ingredient)
    return (on_hand + pending) / usage


class Critic:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def review(self, actions: list, obs, belief, narrative: str = ""
               ) -> tuple[list, list[str]]:
        """Returns (kept_actions, veto_reasons)."""
        if not actions:
            return actions, []

        data = self.client.chat_json(
            system=CRITIC_SYSTEM,
            user=critic_user(obs, belief, actions, narrative),
            fallback={"veto": [], "reason": "fallback"},
            max_tokens=120,
            temperature=0.0,
        )
        veto_idx = data.get("veto", [])
        if not isinstance(veto_idx, list):
            veto_idx = []
        # validate indices
        veto_idx = [int(i) for i in veto_idx
                    if isinstance(i, (int, float)) and 0 <= int(i) < len(actions)]

        kept, reasons = [], []
        for i, a in enumerate(actions):
            if i in veto_idx:
                # Survival override: don't let critic block urgent reorders.
                if a.tool == "place_order":
                    ing = a.args.get("ingredient", "")
                    if _days_of_cover(obs, ing, belief) < 3:
                        kept.append(a)  # override veto
                        reasons.append(
                            f"critic vetoed urgent place_order({ing}); "
                            f"OVERRIDDEN — cover<3d")
                        continue
                reasons.append(f"VETO[{i}] {a.tool} {a.args}: "
                               f"{data.get('reason', '')}")
                continue
            kept.append(a)
        return kept, reasons