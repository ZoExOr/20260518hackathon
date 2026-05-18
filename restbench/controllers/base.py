"""Controller interface + shared supply-chain math.

The delivery-calendar solver lives here because it is the single most
error-prone bit of the whole game (the "order Thursday from a Wed-only
supplier = 6 days" trap) and MUST be deterministic and unit-tested. Never
let an LLM do this arithmetic.
"""
from __future__ import annotations

from typing import Protocol

from ..types import Observation, BeliefState, ProposedAction, weekday_of_day
from ..params import Params


class Controller(Protocol):
    name: str

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params) -> list[ProposedAction]:
        ...


def earliest_delivery_day(order_day: int, lead_time_days: int,
                          delivery_days: list[str],
                          horizon: int = 31) -> int | None:
    """First day an order placed on `order_day` can actually arrive.

    Rule (from AGENT_CONTRACT.md): delivery happens on the first day that is
    (a) >= order_day + lead_time AND (b) one of the supplier's delivery
    weekdays. Returns None if no valid day within `horizon`.

    This is a pure function — covered by tests/test_core.py.
    """
    earliest = order_day + lead_time_days
    valid = set(delivery_days)
    for d in range(earliest, horizon + 1):
        if weekday_of_day(d) in valid:
            return d
    return None


def supplier_for(obs: Observation, ingredient: str
                 ) -> list[dict]:
    """All suppliers that stock `ingredient`, with their unit price."""
    out = []
    for s in obs.supplier_catalog:
        ings = s.get("ingredients", {})
        if ingredient in ings:
            out.append({**s, "_price": ings[ingredient]})
    return out
