"""Tests for the deterministic, network-free pieces.

Run: `pytest -q` from the project root.

These two functions are the highest-risk correctness points in the whole
agent (delivery timing + the bankruptcy invariant), so they get tested
first and hardest.
"""
from restbench.types import Observation, ProposedAction, Action, BeliefState
from restbench.params import Params
from restbench.controllers.base import earliest_delivery_day
from restbench.controllers.supply import SupplyController
from restbench.belief import BeliefEstimator
from restbench.safety import SafetyGate


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


def test_supply_sizes_new_orders_from_planning_shelf_life_not_stale_stock():
    obs = Observation({
        "day": 4, "day_of_week": "Thursday", "days_remaining": 26,
        "cash": 15000.0, "staff_level": 8, "reputation_band": "Very Good",
        "inventory": [
            {"ingredient": "Chicken", "total_kg": 0.0, "shelf_life_days": -1,
             "batches": [{"quantity_kg": 0.0, "expires_in_days": -1}]}
        ],
        "supplier_catalog": [
            {"name": "Fresh Farms NL", "lead_time_days": 1,
             "delivery_days": ["Friday"], "min_order_kg": 5.0,
             "ingredients": {"Chicken": 8.0}}
        ],
        "pending_orders": [],
        "delivery_history": [],
        "menu_book": [
            {"name": "Chicken Plate", "base_price": 18.0, "is_active": True,
             "ingredients": [{"ingredient": "Chicken", "quantity_kg": 0.2}]},
        ],
        "active_menu": ["Chicken Plate"],
        "recent_reviews": [],
        "service_summary": {},
    })
    belief = BeliefState(ingredient_daily_usage={"Chicken": 4.0})
    out = SupplyController().propose(obs, belief, Params())
    chicken_orders = [
        p.action.args["quantity_kg"]
        for p in out
        if p.action.tool == "place_order"
        and p.action.args["ingredient"] == "Chicken"
    ]
    assert chicken_orders and chicken_orders[0] > 5.0, \
        "expired current stock must not cap fresh replenishment at min order"


def test_supply_uses_staple_floor_before_slow_delivery_window():
    obs = Observation({
        "day": 3, "day_of_week": "Wednesday", "days_remaining": 27,
        "cash": 17000.0, "staff_level": 5, "reputation_band": "Very Good",
        "inventory": [
            {"ingredient": "Flour", "total_kg": 47.0, "shelf_life_days": 10,
             "batches": [{"quantity_kg": 47.0, "expires_in_days": 10}]}
        ],
        "supplier_catalog": [
            {"name": "North Sea Millers", "lead_time_days": 2,
             "delivery_days": ["Monday", "Wednesday", "Friday"],
             "min_order_kg": 10.0, "ingredients": {"Flour": 1.8}}
        ],
        "pending_orders": [],
        "delivery_history": [],
        "menu_book": [
            {"name": "Pizza Margherita", "base_price": 14.5, "is_active": True,
             "ingredients": [{"ingredient": "Flour", "quantity_kg": 0.25}]},
            {"name": "Pizza Pepperoni", "base_price": 16.0, "is_active": True,
             "ingredients": [{"ingredient": "Flour", "quantity_kg": 0.25}]},
        ],
        "active_menu": ["Pizza Margherita", "Pizza Pepperoni"],
        "recent_reviews": [],
        "service_summary": {},
    })
    belief = BeliefState(ingredient_daily_usage={"Flour": 5.0})
    out = SupplyController().propose(obs, belief, Params())
    assert any(
        p.action.tool == "place_order"
        and p.action.args["ingredient"] == "Flour"
        for p in out
    ), "staple floors should trigger flour before the next slow delivery gap"


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
