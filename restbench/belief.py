"""BeliefEstimator — the "online research" half of the system.

This costs ZERO extra API calls: it is pure inference on the observation
stream you already receive. This is where most of the score is won.

OWNER: belief / research workstream.

Three jobs:
  1. Uncensor demand. Observed covers/sales are censored by capacity and by
     stockouts (`service_summary.dishes_unavailable_at`). We reconstruct an
     estimate of *true* demand so the supply controller orders for what
     customers wanted, not just what they got.
  2. Track supplier reliability as a Beta posterior over fill ratio
     (delivered_kg / ordered_kg) from `delivery_history`.
  3. Reconstruct a continuous reputation estimate from the lagged review
     stream + band, and a trajectory ("Rising"/"Stable"/"Falling").

Everything is a heuristic skeleton with the right *contract*. Improving the
internals is the single highest-value task — see TODO(belief) markers.
"""
from __future__ import annotations

from .types import (Observation, BeliefState, SERVICE_HOURS, weekday_of_day)

_BAND_TO_SCALAR = {  # rough midpoints; tune against review stream
    "Poor": 1.5, "Fair": 2.5, "Good": 3.5, "Very Good": 4.2, "Excellent": 4.8,
}
_TREND_FACTOR = {"Declining": 0.9, "Stable": 1.0, "Growing": 1.1}
_EMA = 0.35  # smoothing for online estimates


class BeliefEstimator:
    def __init__(self) -> None:
        self.belief = BeliefState()

    # -----------------------------------------------------------------------
    def update(self, obs: Observation) -> BeliefState:
        """Fold one observation into the carried belief and return it."""
        b = self.belief
        b.day = obs.day
        self._update_demand(obs, b)
        self._update_suppliers(obs, b)
        self._update_reputation(obs, b)
        return b

    # --- demand ------------------------------------------------------------
    def _uncensor_covers(self, obs: Observation) -> float:
        """Estimate true covers from yesterday's (censored) service.

        Censoring sources:
          * a dish ran out partway through service -> lost the rest of its day
          * table capacity saturated + walkouts -> demand exceeded seats
        TODO(belief): replace the flat uplifts with a fitted model.
        """
        ss = obs.service_summary
        covers = float(ss.get("total_covers", 0) or 0)
        if covers <= 0:
            return covers

        # 1) stockout uplift: fraction of service hours lost to the first
        #    dish that ran out (conservative lower bound on lost demand).
        unavailable = ss.get("dishes_unavailable_at", {}) or {}
        if unavailable:
            first_out = min(int(h) for h in unavailable.values())
            total_h = len(SERVICE_HOURS)
            lost_h = max(0, (SERVICE_HOURS[-1] - first_out))
            covers *= 1.0 + 0.5 * (lost_h / total_h)  # 0.5 = damping

        # 2) capacity uplift if peak utilisation saturated with walkouts.
        util = float(ss.get("table_utilization_peak", 0) or 0)
        walk = ss.get("walkout_band", "None")
        if util >= 0.95 and walk in ("Some", "Many"):
            covers *= 1.15
        return covers

    def _update_demand(self, obs: Observation, b: BeliefState) -> None:
        wd = obs.day_of_week
        true_covers = self._uncensor_covers(obs)
        if true_covers > 0:
            prev = b.weekday_covers.get(wd, true_covers)
            b.weekday_covers[wd] = (1 - _EMA) * prev + _EMA * true_covers

        # weather effect: ratio of today's covers to this weekday's mean.
        w = obs.weather_today
        if true_covers > 0 and wd in b.weekday_covers:
            ratio = true_covers / max(1.0, b.weekday_covers[wd])
            prev = b.weather_factor.get(w, 1.0)
            b.weather_factor[w] = (1 - _EMA) * prev + _EMA * ratio

        # per-ingredient daily kg usage, scaled by the censoring uplift.
        ss = obs.service_summary
        sold = ss.get("dishes_sold", {}) or {}
        raw_covers = max(1.0, float(ss.get("total_covers", 1) or 1))
        scale = true_covers / raw_covers if raw_covers else 1.0
        usage: dict[str, float] = {}
        for dish, qty in sold.items():
            rec = obs.recipe(dish)
            if not rec:
                continue
            for ing in rec.get("ingredients", []):
                usage[ing["ingredient"]] = usage.get(ing["ingredient"], 0.0) + \
                    ing["quantity_kg"] * qty * scale
        for ing, kg in usage.items():
            prev = b.ingredient_daily_usage.get(ing, kg)
            b.ingredient_daily_usage[ing] = (1 - _EMA) * prev + _EMA * kg

    def forecast_covers(self, obs: Observation, day_offset: int = 1) -> float:
        """Predict covers `day_offset` days ahead (weekday x weather x trend)."""
        b = self.belief
        target_day = obs.day + day_offset
        wd = weekday_of_day(target_day)
        base = b.weekday_covers.get(wd) or (
            sum(b.weekday_covers.values()) / len(b.weekday_covers)
            if b.weekday_covers else 90.0)
        forecast = obs.weather_forecast
        w = forecast[day_offset - 1] if 0 <= day_offset - 1 < len(forecast) else ""
        wf = b.weather_factor.get(w, 1.0)
        tf = _TREND_FACTOR.get(obs.customer_trend, 1.0)
        return base * wf * tf

    # --- suppliers ---------------------------------------------------------
    def _update_suppliers(self, obs: Observation, b: BeliefState) -> None:
        # Only fold each delivery record once (track by a stable key).
        seen: set = b.memory.setdefault("_seen_deliveries", set())
        for d in obs.delivery_history:
            key = (d["supplier"], d["ingredient"], d["order_day"],
                   d["delivery_day"])
            if key in seen:
                continue
            seen.add(key)
            sb = b.supplier(d["supplier"])
            ordered = max(1e-6, float(d.get("ordered_kg", 0)))
            fill = float(d.get("delivered_kg", 0)) / ordered
            sb.alpha += max(0.0, min(1.0, fill))
            sb.beta += max(0.0, 1.0 - min(1.0, fill))

    # --- reputation --------------------------------------------------------
    def _update_reputation(self, obs: Observation, b: BeliefState) -> None:
        anchor = _BAND_TO_SCALAR.get(obs.reputation_band, b.reputation_est)
        reviews = obs.recent_reviews or []
        if reviews:
            mean_stars = sum(r.get("stars", anchor) for r in reviews) / len(reviews)
            target = 0.5 * anchor + 0.5 * mean_stars
        else:
            target = anchor
        prev = b.reputation_est
        b.reputation_est = (1 - _EMA) * prev + _EMA * target
        if b.reputation_est > prev + 0.05:
            b.reputation_trajectory = "Rising"
        elif b.reputation_est < prev - 0.05:
            b.reputation_trajectory = "Falling"
        else:
            b.reputation_trajectory = "Stable"
