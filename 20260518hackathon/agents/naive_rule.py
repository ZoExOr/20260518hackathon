"""Naive rule-based agent — daily ordering from cheapest supplier.

Strategy:
- Reduce staff to 5 on day 1 (saves 360/day vs default 8)
- Order perishables daily when effective stock is below threshold
- Track pending orders to avoid double-ordering
- Keep cash reserve for overhead
- Use happy hour on slow weekdays
"""

from __future__ import annotations

from agents.runner import run_game

REORDER_POINT = {
    "Flour": 6.0,
    "Tomato Sauce": 3.0,
    "Mozzarella": 3.0,
    "Fresh Pasta": 4.0,
    "Cream": 2.0,
    "Mushrooms": 2.0,
    "Chicken": 3.0,
    "Lettuce": 2.0,
    "Pepperoni": 2.0,
    "Salmon": 2.0,
}

ORDER_QTY = {
    "Flour": 8.0,
    "Tomato Sauce": 5.0,
    "Mozzarella": 5.0,
    "Fresh Pasta": 8.0,
    "Cream": 5.0,
    "Mushrooms": 5.0,
    "Chicken": 5.0,
    "Lettuce": 5.0,
    "Pepperoni": 5.0,
    "Salmon": 5.0,
}

SLOW_DAYS = {"Monday", "Tuesday", "Wednesday"}


def strategy(observation: dict, day: int) -> list[dict]:
    actions: list[dict] = []

    if day == 1:
        actions.append({"tool": "set_staff_level", "args": {"level": 5}})

    dow = observation.get("day_of_week", "")
    if dow in SLOW_DAYS:
        actions.append({"tool": "run_happy_hour", "args": {}})

    inventory_map = {}
    fresh_stock = {}
    for inv in observation.get("inventory", []):
        inventory_map[inv["ingredient"]] = inv["total_kg"]
        long_life = sum(
            b["quantity_kg"] for b in inv.get("batches", [])
            if b["expires_in_days"] > 1
        )
        fresh_stock[inv["ingredient"]] = long_life

    cheapest: dict[str, tuple[str, float, float]] = {}
    for sup in observation.get("supplier_catalog", []):
        for ingredient, price in sup["ingredients"].items():
            if ingredient not in cheapest or price < cheapest[ingredient][1]:
                cheapest[ingredient] = (sup["name"], price, sup["min_order_kg"])

    pending_qty: dict[str, float] = {}
    for po in observation.get("pending_orders", []):
        pending_qty[po["ingredient"]] = pending_qty.get(po["ingredient"], 0) + po["quantity_kg"]

    cash = observation["cash"]
    reserve = 1500
    budget = cash - reserve
    if budget <= 0:
        return actions
    spent = 0.0

    needs = []
    for ingredient, reorder in REORDER_POINT.items():
        usable = fresh_stock.get(ingredient, 0)
        pending = pending_qty.get(ingredient, 0)
        effective = usable + pending

        if effective < reorder and ingredient in cheapest:
            supplier_name, price, min_order = cheapest[ingredient]
            qty = max(ORDER_QTY.get(ingredient, 5.0), min_order)
            cost = qty * price
            needs.append((cost, ingredient, supplier_name, qty))

    needs.sort()

    for cost, ingredient, supplier_name, qty in needs:
        if spent + cost > budget:
            continue
        actions.append({
            "tool": "place_order",
            "args": {
                "supplier": supplier_name,
                "ingredient": ingredient,
                "quantity_kg": round(qty, 1),
            },
        })
        spent += cost

    return actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="naive_rule", seed=42)
