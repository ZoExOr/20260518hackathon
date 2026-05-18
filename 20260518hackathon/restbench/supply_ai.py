"""Supply-specific LLM context, policy selection, and advice validation.

The supply LLM is intentionally a strategy optimizer, not a free-form order
writer. Deterministic helpers compute the calendar/math-heavy facts; the model
may choose a named policy and bounded adjustments that still flow through the
Coordinator, DeterministicRiskGate, and SafetyGate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controllers.base import (earliest_delivery_day, planning_shelf_life_days,
                               serviceability_risk, supplier_for,
                               usable_inventory_kg)
from .params import Params
from .renovation import renovation_prep_or_recovery
from .types import Action, BeliefState, Observation


SUPPLY_POLICIES = (
    "lean",
    "balanced",
    "defensive",
    "renovation_recovery",
    "crisis_diversified",
    "endgame_trim",
)

SUPPLY_PARAM_KEYS = {
    "max_order_cash_frac",
    "reorder_days",
    "target_days",
    "safety_days",
    "reliability_inflation",
    "waste_aversion",
}


@dataclass
class SupplyAdvice:
    policy_choice: str = "balanced"
    summary: str = ""
    param_overrides: dict[str, float] = field(default_factory=dict)
    order_adjustments: list[dict[str, Any]] = field(default_factory=list)
    menu_recovery_priority: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_choice": self.policy_choice,
            "summary": self.summary,
            "param_overrides": self.param_overrides,
            "order_adjustments": self.order_adjustments,
            "menu_recovery_priority": self.menu_recovery_priority,
            "risks": self.risks,
            "confidence": self.confidence,
        }


def select_supply_policy(obs: Observation, belief: BeliefState,
                         params: Params) -> str:
    """Deterministic fallback policy, used even when the LLM is unavailable."""
    risk = serviceability_risk(obs, belief, params)
    alerts = " ".join(obs.alerts).lower()
    if obs.days_remaining <= max(3, params.endgame_window):
        return "endgame_trim"
    if _renovation_signal(obs, belief):
        return "renovation_recovery"
    if any(token in alerts for token in (
        "supplier", "shortage", "delay", "outage", "strike", "crisis"
    )):
        return "crisis_diversified"
    if risk.critical:
        return "crisis_diversified"
    if risk.high_or_worse:
        return "defensive"
    if risk.level == "medium":
        return "balanced"
    return "lean"


def policy_param_overrides(policy: str, params: Params) -> dict[str, float]:
    """Small bounded policy patches that keep strategy influence reproducible."""
    if policy == "lean":
        return {
            "target_days": min(params.target_days, 6.5),
            "safety_days": min(params.safety_days, 1.5),
            "waste_aversion": max(params.waste_aversion, 1.15),
        }
    if policy == "defensive":
        return {
            "reorder_days": max(params.reorder_days, 4.0),
            "target_days": max(params.target_days, 8.0),
            "safety_days": max(params.safety_days, 2.5),
        }
    if policy == "renovation_recovery":
        return {
            "reorder_days": max(params.reorder_days, 4.0),
            "target_days": max(params.target_days, 8.5),
            "safety_days": max(params.safety_days, 2.75),
            "waste_aversion": max(params.waste_aversion, 1.05),
        }
    if policy == "crisis_diversified":
        return {
            "reorder_days": max(params.reorder_days, 4.5),
            "target_days": max(params.target_days, 9.0),
            "safety_days": max(params.safety_days, 3.0),
            "reliability_inflation": max(params.reliability_inflation, 1.6),
            "max_order_cash_frac": max(params.max_order_cash_frac, 0.4),
        }
    if policy == "endgame_trim":
        return {
            "target_days": min(params.target_days, max(3.0, obs_target_cap(params))),
            "safety_days": min(params.safety_days, 1.0),
            "waste_aversion": max(params.waste_aversion, 1.6),
            "max_order_cash_frac": min(params.max_order_cash_frac, 0.25),
        }
    return {}


def obs_target_cap(params: Params) -> float:
    return max(3.0, min(5.0, params.target_days))


def should_call_supply_llm(obs: Observation, belief: BeliefState,
                           params: Params) -> bool:
    risk = serviceability_risk(obs, belief, params)
    alerts = " ".join(obs.alerts).lower()
    if risk.high_or_worse:
        return True
    if _renovation_signal(obs, belief):
        return True
    if obs.day <= 4:
        return True
    if obs.days_remaining <= max(3, params.endgame_window):
        return True
    return any(token in alerts for token in (
        "supplier", "shortage", "delay", "outage", "strike", "crisis"
    ))


def load_supply_prompt_patch() -> str:
    """Optional prompt patch promoted by the supply autoresearch loop."""
    path = os.environ.get(
        "RESTBENCH_SUPPLY_FRONTIER",
        os.path.join("research_runs", "supply_frontier.json"),
    )
    if not path:
        return ""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    patch = data.get("prompt_patch", {}) or {}
    text = str(patch.get("supply", "") or "").strip()
    if not text:
        return ""
    return "\nPromoted supply research note: " + text[:1200]


def build_supply_context(obs: Observation, belief: BeliefState,
                         params: Params) -> dict[str, Any]:
    risk = serviceability_risk(obs, belief, params)
    pending_by_ing: dict[str, float] = {}
    pending_eta_by_ing: dict[str, int] = {}
    for po in obs.pending_orders:
        ing = str(po.get("ingredient", ""))
        if not ing:
            continue
        pending_by_ing[ing] = pending_by_ing.get(ing, 0.0) + float(
            po.get("quantity_kg", 0) or 0)
        if po.get("delivery_day") is not None:
            pending_eta_by_ing[ing] = min(
                int(po["delivery_day"]), pending_eta_by_ing.get(ing, 999))

    usage = _usage_by_ingredient(obs, belief)
    inventory_rows = []
    for inv in obs.inventory:
        ing = str(inv.get("ingredient", ""))
        usable = usable_inventory_kg(inv)
        daily = max(1e-6, usage.get(ing, 0.0))
        shelf = planning_shelf_life_days(inv)
        inventory_rows.append({
            "ingredient": ing,
            "usable_kg": round(usable, 2),
            "pending_kg": round(pending_by_ing.get(ing, 0.0), 2),
            "earliest_pending_delivery_day": pending_eta_by_ing.get(ing),
            "earliest_supplier_delivery_day": _earliest_supplier_eta(obs, ing),
            "cover_days": round(
                (usable + pending_by_ing.get(ing, 0.0)) / daily, 2),
            "daily_usage_est": round(daily, 3),
            "shelf_life_days": shelf,
            "expiry_risk": _expiry_risk(inv, daily),
            "supplier_options": _supplier_options(obs, belief, ing),
        })

    return {
        "policy_selected_by_code": select_supply_policy(obs, belief, params),
        "policy_options": list(SUPPLY_POLICIES),
        "supply_risk": risk.to_dict(),
        "renovation_or_recovery": _renovation_signal(obs, belief),
        "ingredient_dashboard": inventory_rows,
        "menu_recovery_ranking": menu_recovery_ranking(obs, belief),
        "delivery_calendar_note": (
            "All ETA fields are deterministic earliest valid delivery days; "
            "do not recompute calendar dates in the LLM."
        ),
        "endgame": {
            "day": obs.day,
            "days_remaining": obs.days_remaining,
            "last_service_day": obs.day + obs.days_remaining,
        },
    }


def menu_recovery_ranking(obs: Observation,
                          belief: BeliefState | None = None) -> list[dict[str, Any]]:
    del belief
    usable = {i["ingredient"]: usable_inventory_kg(i) for i in obs.inventory}
    rows: dict[str, dict[str, Any]] = {}
    for dish in obs.active_menu or [m.get("name", "") for m in obs.menu_book]:
        rec = obs.recipe(dish)
        if not rec:
            continue
        ingredients = rec.get("ingredients", []) or []
        missing = [
            i["ingredient"] for i in ingredients
            if usable.get(i["ingredient"], 0.0) < float(i.get("quantity_kg", 0) or 0)
        ]
        if not missing:
            continue
        price = float(rec.get("current_price", rec.get("base_price", 0)) or 0)
        for ing in missing:
            row = rows.setdefault(ing, {
                "ingredient": ing,
                "dishes_restored": 0,
                "dish_names": [],
                "revenue_proxy": 0.0,
            })
            row["dishes_restored"] += 1
            row["dish_names"].append(dish)
            row["revenue_proxy"] += price
    return sorted(
        rows.values(),
        key=lambda r: (-int(r["dishes_restored"]), -float(r["revenue_proxy"]),
                       str(r["ingredient"])),
    )


def parse_supply_advice(data: dict[str, Any], obs: Observation,
                        params: Params) -> SupplyAdvice:
    raw = data.get("supply_advice") if isinstance(data.get("supply_advice"), dict) else data
    policy = str(raw.get("policy_choice") or raw.get("recommended_policy")
                 or select_supply_policy(obs, BeliefState(), params))
    if policy not in SUPPLY_POLICIES:
        policy = select_supply_policy(obs, BeliefState(), params)

    overrides = dict(policy_param_overrides(policy, params))
    bounds = Params.bounds()
    if isinstance(raw.get("param_overrides"), dict):
        for key, value in raw["param_overrides"].items():
            if key not in SUPPLY_PARAM_KEYS or key not in bounds:
                continue
            lo, hi = bounds[key]
            overrides[key] = max(lo, min(hi, float(value)))

    adjustments = []
    for item in raw.get("order_adjustments", []) or []:
        clean = _clean_order_adjustment(item, obs)
        if clean:
            adjustments.append(clean)

    return SupplyAdvice(
        policy_choice=policy,
        summary=str(raw.get("summary", ""))[:800],
        param_overrides=overrides,
        order_adjustments=adjustments[:12],
        menu_recovery_priority=[
            str(x)[:80] for x in (raw.get("menu_recovery_priority", []) or [])[:10]
        ],
        risks=[str(x)[:300] for x in (raw.get("risks", []) or [])[:8]],
        confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0))),
    )


def supply_advice_to_actions(advice: SupplyAdvice) -> list[Action]:
    actions = []
    for item in advice.order_adjustments:
        if item.get("action") in ("increase", "replace_supplier", "set_quantity"):
            qty = float(item.get("quantity_kg", 0) or 0)
            if qty > 0:
                actions.append(Action("place_order", {
                    "supplier": item["supplier"],
                    "ingredient": item["ingredient"],
                    "quantity_kg": round(qty, 2),
                }))
    return actions


def _clean_order_adjustment(item: Any, obs: Observation) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    action = str(item.get("action", "set_quantity"))
    if action not in ("increase", "decrease", "drop", "replace_supplier", "set_quantity"):
        return None
    supplier = str(item.get("supplier", ""))
    ingredient = str(item.get("ingredient", ""))
    valid = any(
        s.get("name") == supplier and ingredient in (s.get("ingredients", {}) or {})
        for s in obs.supplier_catalog
    )
    if action != "drop" and not valid:
        return None
    qty = float(item.get("quantity_kg", 0) or 0)
    if action != "drop" and qty <= 0:
        return None
    clean = {
        "action": action,
        "supplier": supplier,
        "ingredient": ingredient,
        "quantity_kg": round(qty, 2),
        "reason": str(item.get("reason", ""))[:240],
    }
    if item.get("replace_supplier_from"):
        clean["replace_supplier_from"] = str(item.get("replace_supplier_from"))[:120]
    return clean


def _supplier_options(obs: Observation, belief: BeliefState,
                      ingredient: str) -> list[dict[str, Any]]:
    options = []
    for supplier in supplier_for(obs, ingredient):
        eta = earliest_delivery_day(
            obs.day,
            int(supplier.get("lead_time_days", 1)),
            supplier.get("delivery_days", []),
        )
        reliability = belief.supplier(supplier["name"]).reliability
        options.append({
            "supplier": supplier["name"],
            "eta_day": eta,
            "unit_price": float(supplier.get("_price", 0) or 0),
            "min_order_kg": float(supplier.get("min_order_kg", 0) or 0),
            "reliability": round(reliability, 3),
            "delivery_days": supplier.get("delivery_days", []),
        })
    return sorted(options, key=lambda row: (
        row["eta_day"] if row["eta_day"] is not None else 999,
        row["unit_price"] / max(0.05, row["reliability"]),
    ))


def _earliest_supplier_eta(obs: Observation, ingredient: str) -> int | None:
    etas = [
        earliest_delivery_day(
            obs.day, int(s.get("lead_time_days", 1)), s.get("delivery_days", []))
        for s in supplier_for(obs, ingredient)
    ]
    etas = [eta for eta in etas if eta is not None]
    return min(etas) if etas else None


def _expiry_risk(inv: dict, daily_usage: float) -> str:
    expiring = 0.0
    for batch in inv.get("batches", []) or []:
        if int(batch.get("expires_in_days", 999) or 999) <= 2:
            expiring += float(batch.get("quantity_kg", 0) or 0)
    if expiring <= 0:
        return "low"
    if expiring > max(1.0, 2.0 * daily_usage):
        return "high"
    return "medium"


def _usage_by_ingredient(obs: Observation,
                         belief: BeliefState) -> dict[str, float]:
    usage = dict(belief.ingredient_daily_usage)
    if usage:
        return usage
    active = [d for d in obs.active_menu if obs.recipe(d)]
    if not active:
        return {}
    per_dish = 100.0 / max(1, len(active))
    for dish in active:
        rec = obs.recipe(dish)
        for ing in rec.get("ingredients", []) or []:
            name = ing["ingredient"]
            usage[name] = usage.get(name, 0.0) + \
                float(ing.get("quantity_kg", 0) or 0) * per_dish
    return usage


def _renovation_signal(obs: Observation, belief: BeliefState) -> bool:
    alerts = " ".join(obs.alerts).lower()
    return renovation_prep_or_recovery(obs, belief) or "renovation" in alerts
