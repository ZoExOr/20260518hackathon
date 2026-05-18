"""Controller interface + shared supply-chain math.

The delivery-calendar solver lives here because it is the single most
error-prone bit of the whole game (the "order Thursday from a Wed-only
supplier = 6 days" trap) and MUST be deterministic and unit-tested. Never
let an LLM do this arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Observation, BeliefState, ProposedAction, weekday_of_day
from ..params import Params


PLANNING_SHELF_LIFE_DAYS = {
    "Chicken": 5.0,
    "Cream": 5.0,
    "Flour": 14.0,
    "Fresh Pasta": 4.0,
    "Lettuce": 4.0,
    "Mozzarella": 5.0,
    "Mushrooms": 4.0,
    "Pepperoni": 10.0,
    "Salmon": 3.0,
    "Tomato Sauce": 7.0,
}


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


def planning_shelf_life_days(inv: dict) -> float:
    """Shelf life to use for fresh purchase sizing, not current batch age.

    The API's `shelf_life_days` field behaves like remaining life of current
    stock in observations. If that value is reused as the cap for new orders,
    a stale or expired batch can permanently force future orders down to the
    supplier minimum and keep the restaurant in a stockout loop.
    """
    ingredient = inv.get("ingredient", "")
    known = PLANNING_SHELF_LIFE_DAYS.get(ingredient)
    if known is not None:
        return known
    observed = float(inv.get("shelf_life_days", 14) or 14)
    return max(1.0, observed)


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


@dataclass
class IngredientServiceability:
    ingredient: str
    usable_kg: float
    pending_kg: float
    earliest_delivery_day: int | None
    cover_days: float


@dataclass
class ServiceabilityRisk:
    level: str
    serviceable_dish_count: int
    active_menu_count: int
    missing_ingredients: list[str] = field(default_factory=list)
    ingredient_cover_days: dict[str, float] = field(default_factory=dict)
    late_arrival_risk: bool = False

    @property
    def high_or_worse(self) -> bool:
        return self.level in ("high", "critical")

    @property
    def critical(self) -> bool:
        return self.level == "critical"

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "serviceable_dish_count": self.serviceable_dish_count,
            "active_menu_count": self.active_menu_count,
            "missing_ingredients": self.missing_ingredients,
            "ingredient_cover_days": self.ingredient_cover_days,
            "late_arrival_risk": self.late_arrival_risk,
        }


def _ingredient_usage_prior(obs: Observation, belief: BeliefState,
                            ingredient: str) -> float:
    usage = belief.ingredient_daily_usage.get(ingredient, 0.0)
    if usage > 1e-6:
        return usage
    active = [d for d in obs.active_menu if obs.recipe(d)]
    if not active:
        return 1.0
    covers_prior = 100.0
    per_dish = covers_prior / max(1, len(active))
    kg = 0.0
    for dish in active:
        rec = obs.recipe(dish)
        for ing in rec.get("ingredients", []):
            if ing.get("ingredient") == ingredient:
                kg += float(ing.get("quantity_kg", 0) or 0) * per_dish
    return max(0.5, kg)


def serviceability_risk(obs: Observation, belief: BeliefState,
                        params: Params | None = None) -> ServiceabilityRisk:
    """Deterministic supply/serviceability snapshot shared across agents."""
    del params  # reserved for future tuned thresholds
    usable = {
        inv["ingredient"]: usable_inventory_kg(inv)
        for inv in obs.inventory
    }
    pending_by_ing: dict[str, float] = {}
    eta_by_ing: dict[str, int] = {}
    for po in obs.pending_orders:
        ing = po.get("ingredient")
        if not ing:
            continue
        pending_by_ing[ing] = pending_by_ing.get(ing, 0.0) + float(
            po.get("quantity_kg", 0) or 0)
        eta = po.get("delivery_day")
        if eta is not None:
            eta_by_ing[ing] = min(int(eta), eta_by_ing.get(ing, 999))

    needed: set[str] = set()
    serviceable = 0
    for dish in obs.active_menu:
        rec = obs.recipe(dish)
        if not rec:
            continue
        ingredients = rec.get("ingredients", []) or []
        for ing in ingredients:
            needed.add(ing["ingredient"])
        if all(
            usable.get(ing["ingredient"], 0.0) >= float(ing.get("quantity_kg", 0) or 0)
            for ing in ingredients
        ):
            serviceable += 1

    missing = sorted([ing for ing in needed if usable.get(ing, 0.0) <= 0.05])
    cover_days: dict[str, float] = {}
    late_arrival = False
    for ing in needed:
        usage = _ingredient_usage_prior(obs, belief, ing)
        have = usable.get(ing, 0.0) + pending_by_ing.get(ing, 0.0)
        cover = have / max(1e-6, usage)
        cover_days[ing] = round(cover, 2)
        eta = eta_by_ing.get(ing)
        if eta is not None and eta > 30:
            late_arrival = True

    active_count = len(obs.active_menu)
    min_cover = min(cover_days.values()) if cover_days else 99.0
    missing_ratio = len(missing) / max(1, len(needed))
    if serviceable < 5 or missing_ratio >= 0.5:
        level = "critical"
    elif serviceable < max(5, active_count - 2) or min_cover < 2.0:
        level = "high"
    elif min_cover < 3.5 or missing:
        level = "medium"
    else:
        level = "low"

    return ServiceabilityRisk(
        level=level,
        serviceable_dish_count=serviceable,
        active_menu_count=active_count,
        missing_ingredients=missing,
        ingredient_cover_days=cover_days,
        late_arrival_risk=late_arrival,
    )
