"""Offline replay analysis — the "meta-research" that costs zero live games.

OWNER: research/tuning workstream.

Reads the JSONL replays produced by ReplayStore and extracts structure that
the controllers can bake in as priors:

  * price_elasticity(dish): regress log(units) on log(price) across days
  * scenario_signature(): classify which alerts/delivery patterns precede
    demand or supply shocks (feeds RegimeSupervisor thresholds)
  * supplier_reliability_table(): empirical fill-ratio per supplier

These are deliberately stubs with the right *contract*; this is a top
candidate task once the live agent survives 30 days.
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict


def load_replays(pattern: str = "replays/*.jsonl") -> list[dict]:
    rows: list[dict] = []
    for path in glob.glob(pattern):
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def price_elasticity(rows: list[dict]) -> dict[str, float]:
    """Crude log-log slope of units vs price per dish across all replays.

    TODO(research): control for weather/weekday/reputation before trusting
    these — raw slope conflates demand drivers. Treat as a prior, not truth.
    """
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        ss = (r.get("observation", {}) or {}).get("service_summary", {}) or {}
        sold = ss.get("dishes_sold", {}) or {}
        for m in (r["observation"].get("menu_book", []) or []):
            d, price = m["name"], m.get("current_price")
            u = sold.get(d)
            if price and u and u > 0:
                pts[d].append((math.log(price), math.log(u)))
    out: dict[str, float] = {}
    for d, xy in pts.items():
        if len(xy) < 5:
            continue
        n = len(xy)
        mx = sum(x for x, _ in xy) / n
        my = sum(y for _, y in xy) / n
        cov = sum((x - mx) * (y - my) for x, y in xy)
        var = sum((x - mx) ** 2 for x, _ in xy)
        if var > 1e-9:
            out[d] = cov / var  # elasticity ~ d log(u) / d log(p)
    return out


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
    print("reliability:", supplier_reliability_table(rows))
