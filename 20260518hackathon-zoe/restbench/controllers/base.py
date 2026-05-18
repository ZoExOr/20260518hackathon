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


def usable_inventory_kg(inv: dict, *, next_n_days: int = 1,
                        last_day_weight: float = 0.35) -> float:
    """Inventory that is realistically usable within the next service window.

    The raw `total_kg` can be misleading because batches expiring today are
    often effectively gone before the next turn's service. We therefore:
    - drop already-expired / same-day-expiring stock
    - heavily discount stock expiring tomorrow
    """
    batches = inv.get("batches", []) or []
    if not batches:
        return float(inv.get("total_kg", 0) or 0)

    usable = 0.0
    for batch in batches:
        qty = float(batch.get("quantity_kg", 0) or 0)
        exp = int(batch.get("expires_in_days", 0) or 0)
        if exp <= 0:
            continue
        if exp <= next_n_days:
            usable += qty * last_day_weight
        else:
            usable += qty
    return usable


def structural_stockout(obs: Observation, *, threshold: float = 0.6) -> bool:
    """True when the menu is effectively unserviceable from next turn's stock.

    We flag an emergency when most ingredients required by the active menu have
    no realistically usable stock left.
    """
    needed: set[str] = set()
    for dish in obs.active_menu:
        rec = obs.recipe(dish)
        if not rec:
            continue
        for ing in rec.get("ingredients", []):
            needed.add(ing["ingredient"])

    if not needed:
        return False

    usable = {
        inv["ingredient"]: usable_inventory_kg(inv)
        for inv in obs.inventory
    }
    missing = sum(1 for ing in needed if usable.get(ing, 0.0) <= 0.05)
    return (missing / len(needed)) >= threshold
