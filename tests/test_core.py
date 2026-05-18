"""Tests for the deterministic, network-free pieces.

Run: `pytest -q` from the project root.

These two functions are the highest-risk correctness points in the whole
agent (delivery timing + the bankruptcy invariant), so they get tested
first and hardest.
"""
from restbench.types import Observation, ProposedAction, Action, BeliefState
from restbench.params import Params
from restbench.controllers.base import earliest_delivery_day
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
