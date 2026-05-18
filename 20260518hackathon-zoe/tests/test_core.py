"""Tests for the deterministic, network-free pieces.

Run: `pytest -q` from the project root.

These functions cover high-risk correctness points: delivery timing, the
bankruptcy invariant, survival-oriented supply bootstrapping, and demand
controller behavior.
"""
from restbench.types import Observation, ProposedAction, Action, BeliefState
from restbench.params import Params
from restbench.controllers.base import earliest_delivery_day
from restbench.controllers.pricing import PricingController
from restbench.controllers.supply import SupplyController
from restbench.belief import BeliefEstimator
from restbench.safety import SafetyGate
from restbench.tuning.analyze import price_elasticity, promo_lift, dish_mix


# --- delivery calendar -------------------------------------------------------

def test_earliest_delivery_basic():
    # Day 1 = Monday. Order day 1, lead 1 -> earliest candidate day 2 (Tue).
    # Supplier delivers Mon/Wed/Fri -> first valid >= day2 is day 3 (Wed).
    assert earliest_delivery_day(1, 1, ["Monday", "Wednesday", "Friday"]) == 3


def test_earliest_delivery_the_readme_trap():
    # Order Thursday (day 4) from a Wednesday-only supplier, 1-day lead.
    # earliest = day 5 (Fri); next Wednesday is day 10 -> 6 days out.
    assert earliest_delivery_day(4, 1, ["Wednesday"]) == 10


def test_earliest_delivery_none_when_impossible():
    assert earliest_delivery_day(1, 1, ["Wednesday"], horizon=2) is None


# --- safety gate: the bankruptcy invariant ----------------------------------

def _obs(cash: float) -> Observation:
    return Observation({
        "day": 5, "day_of_week": "Friday", "days_remaining": 25,
        "cash": cash, "staff_level": 8, "reputation_band": "Good",
        "inventory": [], "supplier_catalog": [{
            "name": "S1", "lead_time_days": 1, "delivery_days": ["Friday"],
            "min_order_kg": 5.0, "ingredients": {"Chicken": 10.0}}],
        "pending_orders": [], "delivery_history": [], "menu_book": [],
        "active_menu": [], "recent_reviews": [], "service_summary": {},
    })


def test_safety_blocks_unaffordable_order():
    gate = SafetyGate()
    obs = _obs(cash=3000.0)                      # tight on cash
    params = Params(cash_reserve_floor=2500.0)
    # A 100kg @ 10 EUR = 1000 EUR order would breach the reserve+overhead.
    props = [ProposedAction(Action("place_order", {
        "supplier": "S1", "ingredient": "Chicken", "quantity_kg": 100.0}))]
    out = gate.filter(props, obs, BeliefState(), params)
    assert all(a.tool != "place_order" for a in out), \
        "order that breaches the cash reserve must be dropped"


def test_safety_allows_safe_order_and_clamps_staff():
    gate = SafetyGate()
    obs = _obs(cash=15000.0)
    params = Params(cash_reserve_floor=2500.0, max_order_cash_frac=0.5)
    props = [
        ProposedAction(Action("place_order", {
            "supplier": "S1", "ingredient": "Chicken", "quantity_kg": 10.0})),
        ProposedAction(Action("set_staff_level", {"level": 99})),  # illegal
    ]
    out = gate.filter(props, obs, BeliefState(), params)
    staff = [a for a in out if a.tool == "set_staff_level"][0]
    assert staff.args["level"] == 15, "staff must be clamped to <= 15"
    assert any(a.tool == "place_order" for a in out), "affordable order kept"


# --- supply bootstrap / delivery-aware reorder ------------------------------

def test_supply_bootstrap_provisions_for_likely_widened_menu():
    obs = Observation({
        "day": 1, "day_of_week": "Monday", "days_remaining": 29,
        "cash": 15000.0, "staff_level": 8, "reputation_band": "Very Good",
        "inventory": [],
        "supplier_catalog": [
            {"name": "S1", "lead_time_days": 1, "delivery_days": ["Tuesday"],
             "min_order_kg": 1.0, "ingredients": {"Flour": 2.0, "Salmon": 9.0}}
        ],
        "pending_orders": [], "delivery_history": [],
        "menu_book": [
            {"name": "Pizza", "base_price": 10.0, "is_active": True,
             "ingredients": [{"ingredient": "Flour", "quantity_kg": 0.2}]},
            {"name": "Salmon Plate", "base_price": 20.0, "is_active": False,
             "ingredients": [{"ingredient": "Salmon", "quantity_kg": 0.25}]},
        ],
        "active_menu": ["Pizza"],
        "recent_reviews": [], "service_summary": {},
    })
    out = SupplyController().propose(obs, BeliefState(), Params())
    ordered = {(p.action.args["ingredient"], p.action.args["supplier"]) for p in out}
    assert ("Salmon", "S1") in ordered, \
        "bootstrap supply should cover likely menu expansion, not just active_menu"


def test_supply_reorders_when_next_delivery_is_after_runout():
    obs = Observation({
        "day": 4, "day_of_week": "Thursday", "days_remaining": 26,
        "cash": 15000.0, "staff_level": 8, "reputation_band": "Very Good",
        "inventory": [
            {"ingredient": "Chicken", "total_kg": 8.0, "shelf_life_days": 7,
             "batches": [{"quantity_kg": 8.0, "expires_in_days": 5}]}
        ],
        "supplier_catalog": [
            {"name": "S1", "lead_time_days": 1, "delivery_days": ["Wednesday"],
             "min_order_kg": 5.0, "ingredients": {"Chicken": 8.0}}
        ],
        "pending_orders": [], "delivery_history": [],
        "menu_book": [
            {"name": "Chicken Plate", "base_price": 18.0, "is_active": True,
             "ingredients": [{"ingredient": "Chicken", "quantity_kg": 0.2}]},
        ],
        "active_menu": ["Chicken Plate"],
        "recent_reviews": [],
        "service_summary": {},
    })
    belief = BeliefState(ingredient_daily_usage={"Chicken": 2.0})
    out = SupplyController().propose(obs, belief, Params(reorder_days=3.0))
    assert any(p.action.tool == "place_order" for p in out), \
        "supply must reorder before stockout when delivery cadence is slow"


def test_belief_supplier_update_tolerates_missing_order_day():
    obs = Observation({
        "day": 8, "day_of_week": "Monday", "days_remaining": 22,
        "cash": 10000.0, "staff_level": 8, "reputation_band": "Good",
        "inventory": [], "pending_orders": [],
        "supplier_catalog": [], "menu_book": [], "active_menu": [],
        "recent_reviews": [], "service_summary": {},
        "delivery_history": [{
            "supplier": "S1",
            "ingredient": "Chicken",
            "ordered_kg": 10.0,
            "delivered_kg": 5.0,
            "delivery_day": 8,
        }],
    })
    est = BeliefEstimator()
    belief = est.update(obs)
    reliab = belief.supplier("S1").reliability
    assert reliab < 2.0 / 3.0, \
        "partial deliveries should reduce the supplier reliability estimate"


# --- demand controller --------------------------------------------------------

def _demand_obs(
    *,
    trend: str = "Growing",
    rep: str = "Very Good",
    covers: int = 125,
    walk: str = "None",
    peak_wait: float = 4.0,
    days_remaining: int = 20,
) -> Observation:
    menu = [
        {"name": "Pizza Margherita", "base_price": 14.5,
         "current_price": 14.5, "is_active": True,
         "ingredients": [{"ingredient": "Flour", "quantity_kg": 0.25},
                         {"ingredient": "Tomato Sauce", "quantity_kg": 0.09}]},
        {"name": "Chicken Parmesan", "base_price": 20.0,
         "current_price": 20.0, "is_active": True,
         "ingredients": [{"ingredient": "Chicken", "quantity_kg": 0.18}]},
        {"name": "Grilled Salmon", "base_price": 24.0,
         "current_price": 24.0, "is_active": True,
         "ingredients": [{"ingredient": "Salmon", "quantity_kg": 0.2}]},
        {"name": "Mushroom Risotto", "base_price": 19.0,
         "current_price": 19.0, "is_active": True,
         "ingredients": [{"ingredient": "Mushrooms", "quantity_kg": 0.12}]},
        {"name": "Spaghetti Carbonara", "base_price": 16.5,
         "current_price": 16.5, "is_active": True,
         "ingredients": [{"ingredient": "Fresh Pasta", "quantity_kg": 0.18}]},
    ]
    return Observation({
        "day": 8, "day_of_week": "Monday", "days_remaining": days_remaining,
        "cash": 12000.0, "staff_level": 8, "reputation_band": rep,
        "customer_trend": trend, "weather_today": "sunny",
        "weather_forecast": ["sunny", "cloudy", "rainy"],
        "alerts": [], "pending_orders": [], "delivery_history": [],
        "supplier_catalog": [], "recent_reviews": [],
        "active_menu": [m["name"] for m in menu],
        "menu_book": menu,
        "inventory": [
            {"ingredient": "Flour", "total_kg": 40.0, "batches": [
                {"quantity_kg": 15.0, "expires_in_days": 2}]},
            {"ingredient": "Tomato Sauce", "total_kg": 20.0, "batches": []},
            {"ingredient": "Chicken", "total_kg": 15.0, "batches": []},
            {"ingredient": "Salmon", "total_kg": 12.0, "batches": []},
            {"ingredient": "Mushrooms", "total_kg": 10.0, "batches": []},
            {"ingredient": "Fresh Pasta", "total_kg": 18.0, "batches": []},
        ],
        "service_summary": {
            "total_covers": covers, "total_revenue": 2400.0,
            "walkout_band": walk, "peak_wait_minutes": peak_wait,
            "dishes_sold": {
                "Pizza Margherita": 35, "Chicken Parmesan": 20,
                "Grilled Salmon": 12,
            },
        },
    })


def _actions_by_tool(actions):
    return {a.action.tool: a.action for a in actions}


def test_pricing_raises_prices_on_healthy_demand():
    obs = _demand_obs()
    belief = BeliefState(reputation_trajectory="Stable")
    out = PricingController().propose(obs, belief, Params())
    price_actions = [p.action for p in out if p.action.tool == "set_price"]
    assert price_actions, "healthy demand should propose price actions"
    assert all(a.args["price"] > obs.recipe(a.args["dish"])["base_price"]
               for a in price_actions)
    assert all(a.args["price"] <= 1.2 * obs.recipe(a.args["dish"])["base_price"]
               for a in price_actions)


def test_pricing_avoids_happy_hour_when_service_is_risky():
    obs = _demand_obs(trend="Declining", walk="Some", peak_wait=22)
    belief = BeliefState(reputation_trajectory="Falling")
    out = PricingController().propose(obs, belief, Params(base_price_mult=1.1))
    tools = _actions_by_tool(out)
    assert "run_happy_hour" not in tools
    price_actions = [p.action for p in out if p.action.tool == "set_price"]
    assert all(a.args["price"] <= obs.recipe(a.args["dish"])["base_price"] * 1.01
               for a in price_actions)


def test_pricing_markets_and_happy_hours_weak_demand():
    obs = _demand_obs(trend="Declining", covers=45, walk="None", peak_wait=4)
    belief = BeliefState(reputation_trajectory="Falling")
    out = PricingController().propose(obs, belief, Params())
    tools = _actions_by_tool(out)
    assert tools["set_marketing_spend"].args["amount"] > 0
    assert "run_happy_hour" in tools


def test_daily_special_is_active_menu_item():
    obs = _demand_obs()
    out = PricingController().propose(obs, BeliefState(), Params())
    special = _actions_by_tool(out)["offer_daily_special"]
    assert special.args["dish"] in obs.active_menu


# --- replay analysis ---------------------------------------------------------

def test_replay_analysis_helpers_on_fake_rows():
    rows = []
    for i, mult in enumerate([0.95, 1.0, 1.1, 1.15], start=1):
        price = 20.0 * mult
        units = max(5, int(30 - i * 4))
        rows.append({
            "observation": {
                "service_summary": {
                    "total_covers": 100,
                    "dishes_sold": {"Chicken Parmesan": units},
                },
                "menu_book": [{
                    "name": "Chicken Parmesan", "base_price": 20.0,
                    "current_price": price,
                }],
            },
            "actions": [{"tool": "set_marketing_spend",
                         "args": {"amount": 200}}] if i % 2 else [],
            "day_result": {
                "total_covers": 110 if i % 2 else 90,
                "total_revenue": 2100 if i % 2 else 1700,
            },
        })

    elasticity = price_elasticity(rows)
    lift = promo_lift(rows)
    mix = dish_mix(rows)

    assert "Chicken Parmesan" in elasticity
    assert "marketing" in lift
    assert mix["Chicken Parmesan"]["units"] > 0
