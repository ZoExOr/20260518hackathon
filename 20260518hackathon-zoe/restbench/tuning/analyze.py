"""Offline replay analysis — the "meta-research" that costs zero live games.

OWNER: research/tuning workstream.

Reads the JSONL replays produced by ReplayStore and extracts structure that
the controllers can bake in as priors:

  * price_elasticity(dish): regress log(units/covers) on log(price/base)
  * promo_lift(): estimate covers/revenue deltas after marketing/happy hour
  * dish_mix(): sales and revenue contribution by dish
  * supplier_reliability_table(): empirical fill-ratio per supplier
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from statistics import fmean


def load_replays(pattern: str = "replays/*.jsonl") -> list[dict]:
    rows: list[dict] = []
    for path in glob.glob(pattern):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _actions(row: dict) -> list[dict]:
    return list(row.get("actions", []) or [])


def _action_amount(row: dict, tool: str, default: float = 0.0) -> float:
    for a in _actions(row):
        if a.get("tool") == tool:
            args = a.get("args", {}) or {}
            if "amount" in args:
                return float(args["amount"])
            return 1.0
    return default


def _has_action(row: dict, tool: str) -> bool:
    return any(a.get("tool") == tool for a in _actions(row))


def _linreg_slope(xy: list[tuple[float, float]]) -> float | None:
    if len(xy) < 3:
        return None
    n = len(xy)
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    cov = sum((x - mx) * (y - my) for x, y in xy)
    var = sum((x - mx) ** 2 for x, _ in xy)
    if var <= 1e-9:
        return None
    return cov / var


def price_elasticity(rows: list[dict]) -> dict[str, float]:
    """Log-log slope of dish share vs relative price.

    Uses units/covers instead of raw units and price/base_price instead of raw
    price. That does not fully control for weekday/weather/reputation, but it
    removes the largest scale confounders and is good enough as a prior.
    """
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        ss = (r.get("observation", {}) or {}).get("service_summary", {}) or {}
        sold = ss.get("dishes_sold", {}) or {}
        covers = float(ss.get("total_covers", 0) or 0)
        if covers <= 0:
            continue
        for m in (r["observation"].get("menu_book", []) or []):
            d = m["name"]
            price = m.get("current_price")
            base = m.get("base_price")
            u = sold.get(d)
            if price and base and u and u > 0:
                rel_price = float(price) / float(base)
                share = float(u) / covers
                if rel_price > 0 and share > 0:
                    pts[d].append((math.log(rel_price), math.log(share)))
    out: dict[str, float] = {}
    for d, xy in pts.items():
        slope = _linreg_slope(xy)
        if slope is not None:
            out[d] = slope
    return out


def promo_lift(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate covers/revenue lift on days with marketing or happy hour.

    Replay rows log actions submitted before the service day and day_result
    after service, so we can compare promoted rows against non-promoted rows.
    """
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        dr = r.get("day_result") or {}
        covers = float(dr.get("total_covers", 0) or 0)
        revenue = float(dr.get("total_revenue", 0) or 0)
        if covers <= 0 and revenue <= 0:
            continue
        marketing = _action_amount(r, "set_marketing_spend", 0.0)
        happy = _has_action(r, "run_happy_hour")
        if marketing > 0:
            groups["marketing"].append((covers, revenue))
        if happy:
            groups["happy_hour"].append((covers, revenue))
        if marketing <= 0 and not happy:
            groups["baseline"].append((covers, revenue))

    base = groups.get("baseline", [])
    base_covers = fmean(c for c, _ in base) if base else 0.0
    base_revenue = fmean(r for _, r in base) if base else 0.0
    out: dict[str, dict[str, float]] = {}
    for name in ("marketing", "happy_hour"):
        vals = groups.get(name, [])
        if not vals:
            continue
        avg_covers = fmean(c for c, _ in vals)
        avg_revenue = fmean(r for _, r in vals)
        out[name] = {
            "days": float(len(vals)),
            "avg_covers": avg_covers,
            "avg_revenue": avg_revenue,
            "covers_lift": avg_covers - base_covers,
            "revenue_lift": avg_revenue - base_revenue,
        }
    return out


def dish_mix(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate dish units and estimated revenue contribution."""
    units: dict[str, float] = defaultdict(float)
    revenue: dict[str, float] = defaultdict(float)
    for r in rows:
        obs = r.get("observation", {}) or {}
        ss = obs.get("service_summary", {}) or {}
        sold = ss.get("dishes_sold", {}) or {}
        prices = {
            m["name"]: float(m.get("current_price", m.get("base_price", 0)))
            for m in obs.get("menu_book", []) or []
        }
        for dish, qty in sold.items():
            q = float(qty)
            units[dish] += q
            revenue[dish] += q * prices.get(dish, 0.0)
    total_units = sum(units.values()) or 1.0
    total_revenue = sum(revenue.values()) or 1.0
    return {
        d: {
            "units": units[d],
            "unit_share": units[d] / total_units,
            "revenue": revenue[d],
            "revenue_share": revenue[d] / total_revenue,
        }
        for d in sorted(units)
    }


def supplier_reliability_table(rows: list[dict]) -> dict[str, float]:
    agg: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for d in (r["observation"].get("delivery_history", []) or []):
            o = max(1e-6, float(d.get("ordered_kg", 0)))
            agg[d["supplier"]].append(float(d.get("delivered_kg", 0)) / o)
    return {s: (sum(v) / len(v)) for s, v in agg.items() if v}


if __name__ == "__main__":
    rows = load_replays()
    print("replay turns:", len(rows))
    print("elasticity:", price_elasticity(rows))
    print("promo lift:", promo_lift(rows))
    print("dish mix:", dish_mix(rows))
    print("reliability:", supplier_reliability_table(rows))
