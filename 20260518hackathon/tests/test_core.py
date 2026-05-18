"""Tests for the deterministic, network-free pieces.

Run: `pytest -q` from the project root.

These two functions are the highest-risk correctness points in the whole
agent (delivery timing + the bankruptcy invariant), so they get tested
first and hardest.
"""
from restbench.types import Observation, ProposedAction, Action, BeliefState
from restbench.params import Params
from restbench.controllers.base import earliest_delivery_day
from restbench.controllers.base import serviceability_risk
from restbench.controllers.pricing import PricingController
from restbench.controllers.supply import SupplyController
from restbench.belief import BeliefEstimator
from restbench.safety import SafetyGate
from restbench.risk_gate import DeterministicRiskGate
from restbench.multiagent import (MultiAgentAdvisor, parse_agent_advice,
                                  advice_to_proposals, parse_json_object,
                                  parse_risk_review)
from restbench.autoresearch import ResearchAgent, ResearchCandidate
from restbench.agent import Agent


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


# --- multi-agent LLM boundary -----------------------------------------------

def _rich_obs(cash: float = 15000.0) -> Observation:
    return Observation({
        "day": 6, "day_of_week": "Saturday", "days_remaining": 24,
        "cash": cash, "staff_level": 8, "reputation_band": "Good",
        "customer_trend": "Stable", "weather_today": "sunny",
        "weather_forecast": ["sunny", "cloudy", "rainy"],
        "alerts": [],
        "inventory": [
            {"ingredient": "Chicken", "total_kg": 2.0, "shelf_life_days": 5,
             "batches": [{"quantity_kg": 2.0, "expires_in_days": 3}]}
        ],
        "supplier_catalog": [{
            "name": "S1", "lead_time_days": 1, "delivery_days": ["Saturday"],
            "min_order_kg": 5.0, "ingredients": {"Chicken": 10.0}}],
        "pending_orders": [], "delivery_history": [],
        "menu_book": [
            {"name": "Chicken Plate", "base_price": 18.0,
             "current_price": 18.0, "is_active": True,
             "ingredients": [{"ingredient": "Chicken", "quantity_kg": 0.2}]},
            {"name": "Pizza", "base_price": 12.0,
             "current_price": 12.0, "is_active": True,
             "ingredients": []},
            {"name": "Pasta", "base_price": 13.0,
             "current_price": 13.0, "is_active": True,
             "ingredients": []},
            {"name": "Risotto", "base_price": 14.0,
             "current_price": 14.0, "is_active": True,
             "ingredients": []},
            {"name": "Salad", "base_price": 10.0,
             "current_price": 10.0, "is_active": True,
             "ingredients": []},
        ],
        "active_menu": ["Chicken Plate", "Pizza", "Pasta", "Risotto", "Salad"],
        "recent_reviews": [], "service_summary": {},
    })


def _critical_supply_obs(day: int = 21, walk: str = "Many",
                         rep: str = "Fair") -> Observation:
    recipes = [
        ("Chicken Plate", "Chicken", 0.2),
        ("Salmon Plate", "Salmon", 0.25),
        ("Mushroom Pasta", "Mushrooms", 0.12),
        ("Carbonara", "Cream", 0.08),
        ("Pizza", "Mozzarella", 0.12),
    ]
    return Observation({
        "day": day,
        "day_of_week": "Sunday",
        "days_remaining": max(0, 30 - day),
        "cash": 25000.0,
        "staff_level": 5,
        "reputation_band": rep,
        "customer_trend": "Stable",
        "weather_today": "sunny",
        "weather_forecast": ["sunny", "cloudy", "rainy"],
        "alerts": [],
        "inventory": [
            {"ingredient": ing, "total_kg": 0.0, "shelf_life_days": 5,
             "batches": [{"quantity_kg": 0.0, "expires_in_days": -1}]}
            for _, ing, _ in recipes
        ],
        "supplier_catalog": [{
            "name": "LateSupplier", "lead_time_days": 1,
            "delivery_days": ["Wednesday"], "min_order_kg": 5.0,
            "ingredients": {ing: 10.0 for _, ing, _ in recipes},
        }],
        "pending_orders": [],
        "delivery_history": [],
        "menu_book": [
            {"name": dish, "base_price": 18.0, "current_price": 18.0,
             "is_active": True,
             "ingredients": [{"ingredient": ing, "quantity_kg": qty}]}
            for dish, ing, qty in recipes
        ],
        "active_menu": [dish for dish, _, _ in recipes],
        "recent_reviews": [],
        "service_summary": {"walkout_band": walk, "peak_wait_minutes": 0},
    })


def test_parse_json_object_rejects_malformed_output():
    try:
        parse_json_object("not json")
    except Exception:
        pass
    else:
        assert False, "malformed LLM output must not be silently accepted"


def test_agent_advice_to_proposals_filters_invalid_actions():
    obs = _rich_obs()
    data = {
        "role": "supply",
        "summary": "order chicken",
        "confidence": 0.9,
        "proposed_actions": [
            {"tool": "place_order", "args": {
                "supplier": "S1", "ingredient": "Chicken", "quantity_kg": 5}},
            {"tool": "place_order", "args": {
                "supplier": "Missing", "ingredient": "Chicken", "quantity_kg": 5}},
            {"tool": "unknown_tool", "args": {}},
        ],
        "param_overrides": {"target_days": 8.0, "not_a_param": 123},
    }
    advice = parse_agent_advice("supply", data, obs)
    proposals = advice_to_proposals(advice)
    assert len(proposals) == 1
    assert proposals[0].action.tool == "place_order"
    assert advice.param_overrides == {"target_days": 8.0}


class _BadClient:
    def complete_json(self, system, payload):
        raise RuntimeError("bad llm")


def test_multiagent_fallback_records_errors_without_proposals():
    advisor = MultiAgentAdvisor(client=_BadClient(), allow_fallback=True)
    advice, errors = advisor.collect_advice(
        _rich_obs(), BeliefState(), Params(), {"supply": []})
    assert advice == []
    assert len(errors) == 3


def test_risk_review_still_passes_through_safety_gate():
    obs = _rich_obs(cash=3000.0)
    fallback = [ProposedAction(Action("place_order", {
        "supplier": "S1", "ingredient": "Chicken", "quantity_kg": 100.0
    }))]
    review = parse_risk_review({
        "approved_actions": [fallback[0].action.to_payload()],
        "risk_score": 0.1,
        "rationale": "looks fine",
    }, obs, fallback)
    gated = SafetyGate().filter(
        [ProposedAction(a) for a in review.approved_actions],
        obs, BeliefState(), Params(cash_reserve_floor=2500.0))
    assert all(a.tool != "place_order" for a in gated)


class _FakeAdvisor:
    def collect_advice(self, obs, belief, params, baseline):
        advice = parse_agent_advice("operations", {
            "role": "operations",
            "summary": "trim staffing",
            "confidence": 0.9,
            "proposed_actions": [
                {"tool": "set_staff_level", "args": {"level": 6}}
            ],
        }, obs)
        return [advice], []

    def review(self, obs, belief, params, composed):
        return parse_risk_review({
            "approved_actions": [p.action.to_payload() for p in composed],
            "risk_score": 0.2,
            "rationale": "approved",
        }, obs, composed), []


def test_multiagent_agent_records_trace_and_default_path_stays_off():
    obs = _rich_obs()
    default_agent = Agent()
    default_actions = default_agent.decide(obs)
    assert default_actions
    assert default_agent.last_multiagent_trace is None

    multi_agent = Agent(multi_agent=True, multi_agent_advisor=_FakeAdvisor())
    actions = multi_agent.decide(obs)
    assert actions
    assert multi_agent.last_multiagent_trace is not None
    assert multi_agent.last_multiagent_trace.advice[0].role == "operations"
    assert multi_agent.last_multiagent_trace.risk_review is not None


# --- serviceability / deterministic risk gate -------------------------------

def test_serviceability_identifies_critical_stockout():
    obs = _critical_supply_obs()
    risk = serviceability_risk(obs, BeliefState(), Params())
    assert risk.level == "critical"
    assert risk.serviceable_dish_count == 0
    assert set(risk.missing_ingredients) >= {"Chicken", "Salmon", "Mushrooms"}


def test_risk_gate_vetoes_demand_stimulation_when_supply_high():
    obs = _critical_supply_obs()
    props = [
        ProposedAction(Action("run_happy_hour", {})),
        ProposedAction(Action("set_marketing_spend", {"amount": 250.0})),
    ]
    out, report = DeterministicRiskGate().filter(
        props, obs, BeliefState(), Params())
    assert all(p.action.tool != "run_happy_hour" for p in out)
    marketing = [p.action for p in out if p.action.tool == "set_marketing_spend"]
    assert marketing and marketing[0].args["amount"] == 0.0
    assert report.dropped_actions and report.adjusted_actions


def test_risk_gate_drops_late_endgame_order():
    obs = _critical_supply_obs(day=27)
    props = [ProposedAction(Action("place_order", {
        "supplier": "LateSupplier",
        "ingredient": "Chicken",
        "quantity_kg": 30.0,
    }))]
    out, report = DeterministicRiskGate().filter(
        props, obs, BeliefState(), Params())
    assert out == []
    assert report.dropped_actions[0]["reason"].startswith("late endgame")


def test_risk_gate_enforces_staff_floor_after_walkouts():
    obs = _rich_obs()
    obs.raw["service_summary"] = {"walkout_band": "Many"}
    obs.raw["reputation_band"] = "Good"
    props = [ProposedAction(Action("set_staff_level", {"level": 5}))]
    out, _ = DeterministicRiskGate().filter(
        props, obs, BeliefState(), Params())
    assert out[0].action.args["level"] == 8


def test_pricing_avoids_promos_under_supply_risk():
    obs = _rich_obs()
    obs.raw["inventory"][0]["total_kg"] = 0.0
    obs.raw["inventory"][0]["batches"] = [{"quantity_kg": 0.0, "expires_in_days": -1}]
    belief = BeliefState()
    belief.reputation_trajectory = "Falling"
    out = PricingController().propose(obs, belief, Params())
    tools = [p.action.tool for p in out]
    assert "run_happy_hour" not in tools
    marketing = [p.action for p in out if p.action.tool == "set_marketing_spend"]
    assert marketing and marketing[0].args["amount"] == 0.0


def test_risk_gate_drops_unserviceable_daily_special():
    obs = _rich_obs()
    obs.raw["inventory"][0]["total_kg"] = 0.0
    obs.raw["inventory"][0]["batches"] = [{"quantity_kg": 0.0, "expires_in_days": -1}]
    props = [ProposedAction(Action("offer_daily_special", {"dish": "Chicken Plate"}))]
    out, report = DeterministicRiskGate().filter(
        props, obs, BeliefState(), Params())
    assert out == []
    assert "daily special" in report.dropped_actions[0]["reason"]


# --- research loop -----------------------------------------------------------

def test_research_loop_keeps_improvement_and_discards_regression(tmp_path):
    agent = ResearchAgent(run_dir=str(tmp_path))
    scenarios = ("baseline",)
    seeds = (42,)

    def good_eval(params, scenarios, seeds):
        return [100.0]

    first = agent.run_once(
        good_eval, scenarios, seeds,
        candidate=ResearchCandidate(
            params_patch={"target_days": 8.0},
            hypothesis="good candidate"))
    assert first.status == "keep"

    def bad_eval(params, scenarios, seeds):
        return [50.0]

    second = agent.run_once(
        bad_eval, scenarios, seeds,
        candidate=ResearchCandidate(
            params_patch={"target_days": 4.0},
            hypothesis="bad candidate"))
    assert second.status == "discard"
    text = (tmp_path / "results.tsv").read_text(encoding="utf-8")
    assert "tag\tscore\tmean\tstd\tstatus\tparams_patch\thypothesis" in text
    assert "keep" in text and "discard" in text
