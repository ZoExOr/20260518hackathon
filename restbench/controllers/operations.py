"""OperationsController — staffing.

OWNER: operations workstream.

Staffing is a newsvendor-on-labour problem: too few staff -> slow kitchen,
long waits, walkouts (reputation damage, which compounds); too many ->
wasted 120 EUR/person/day. We scale staff to forecast covers and react to
observed wait/walkout signals.

Working baseline: staff = clip(forecast_covers / covers_per_staff) with a
weekend bump and a reactive nudge from yesterday's wait/walkout band.
TODO(ops): fit covers_per_staff from (staff, peak_wait, walkout) history;
the relationship is explicitly non-linear (STRATEGY_GUIDE).
"""
from __future__ import annotations

from .base import Controller
from ..types import (Observation, BeliefState, ProposedAction, Action,
                      weekday_of_day)
from ..params import Params

_WEEKEND = {"Friday", "Saturday", "Sunday"}


class OperationsController:
    name = "operations"

    def __init__(self, forecaster=None):
        # forecaster: callable(obs, offset) -> covers. Injected by the agent
        # so ops and supply share ONE demand model.
        self.forecaster = forecaster

    def propose(self, obs: Observation, belief: BeliefState,
                params: Params) -> list[ProposedAction]:
        if self.forecaster is not None:
            covers = self.forecaster(obs, 1)
        else:
            covers = belief.weekday_covers.get(
                weekday_of_day(obs.day + 1), 90.0)

        level = covers / max(1.0, params.covers_per_staff)
        if weekday_of_day(obs.day + 1) in _WEEKEND:
            level += params.weekend_staff_bonus

        # reactive nudge from yesterday's service
        ss = obs.service_summary
        walk = ss.get("walkout_band", "None")
        peak_wait = float(ss.get("peak_wait_minutes", 0) or 0)
        if walk in ("Some", "Many") or peak_wait > 20:
            level += 1
        elif walk == "None" and peak_wait < 5 and obs.staff_level > params.staff_min:
            level -= 0.5  # gently trim idle capacity

        lvl = int(round(max(params.staff_min, min(params.staff_max, level))))
        if lvl == obs.staff_level:
            return []
        return [ProposedAction(
            Action("set_staff_level", {"level": lvl}),
            priority=50,
            rationale=f"covers~{covers:.0f}, walk={walk}, wait={peak_wait:.0f}")]
