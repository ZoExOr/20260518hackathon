"""Starter template agent — copy this file and build your strategy!

This agent demonstrates the full API contract:
- Observe the world state
- Submit tool calls (orders, menu changes, pricing, etc.)
- End the turn to advance the simulation

It makes minimal decisions to keep the restaurant running.
Your job: make it smarter!
"""

from __future__ import annotations

from agents.runner import run_game


def strategy(observation: dict, day: int) -> list[dict]:
    """Called once per day. Return a list of tool calls.

    Each tool call is: {"tool": "<name>", "args": {<args>}}

    Available tools:
      place_order      — {"supplier": str, "ingredient": str, "quantity_kg": float}
      set_staff_level  — {"level": int}  (3 to 15)
      set_menu         — {"dishes": [str, ...]}  (min 5 dishes)
      set_price        — {"dish": str, "price": float}  (0.8x to 1.2x base)
      set_marketing_spend — {"amount": float}  (0 to 500)
      run_happy_hour   — {}
      offer_daily_special — {"dish": str}
      save_notes       — {"text": str}  (up to 4000 chars, persists between turns)
    """
    actions: list[dict] = []

    # --- Day 1 setup ---
    if day == 1:
        # TODO: Choose your starting staff level (default 8, costs 120/person/day)
        actions.append({"tool": "set_staff_level", "args": {"level": 6}})

    # --- Read observation ---
    cash = observation["cash"]
    inventory = {inv["ingredient"]: inv["total_kg"] for inv in observation.get("inventory", [])}
    pending = {}
    for po in observation.get("pending_orders", []):
        pending[po["ingredient"]] = pending.get(po["ingredient"], 0) + po["quantity_kg"]

    # --- Reorder low ingredients ---
    # TODO: Improve ordering logic — check expiry dates, optimize suppliers, manage budget
    cheapest_supplier = {}
    for sup in observation.get("supplier_catalog", []):
        for ingredient, price in sup["ingredients"].items():
            if ingredient not in cheapest_supplier or price < cheapest_supplier[ingredient][1]:
                cheapest_supplier[ingredient] = (sup["name"], price, sup["min_order_kg"])

    budget = cash - 2000  # Keep a safety reserve
    for ingredient, (supplier, price, min_qty) in cheapest_supplier.items():
        stock = inventory.get(ingredient, 0) + pending.get(ingredient, 0)
        if stock < 3.0 and budget > min_qty * price:
            qty = max(5.0, min_qty)
            actions.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ingredient, "quantity_kg": qty},
            })
            budget -= qty * price

    # --- Promotions ---
    # TODO: Experiment with happy hour, daily specials, marketing spend, and pricing
    # Example: run happy hour on slow days
    # actions.append({"tool": "run_happy_hour", "args": {}})

    # Example: offer a daily special
    # active = observation.get("active_menu", [])
    # if active:
    #     actions.append({"tool": "offer_daily_special", "args": {"dish": active[0]}})

    # --- Notes (optional) ---
    # Save your own state between turns
    # actions.append({"tool": "save_notes", "args": {"text": f"Day {day}: cash={cash}"}})

    return actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="starter", seed=42)
