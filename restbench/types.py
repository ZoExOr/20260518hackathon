"""Core typed data model.

This module is the *contract* every other module depends on. Touch it only
with the whole team's agreement — a change here ripples everywhere.

Design rule: there is exactly ONE source of truth per turn, the `BeliefState`.
Controllers are pure functions of (Observation, BeliefState, Params) and emit
`ProposedAction`s. They never mutate shared state and never call the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Day 1 is Monday (see AGENT_CONTRACT.md "Key Numbers").
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
SERVICE_HOURS = list(range(11, 23))  # 11:00..22:00 inclusive of 22 -> 12 slots


def weekday_of_day(day: int) -> str:
    """Map a 1-indexed simulation day to its weekday name."""
    return WEEKDAYS[(day - 1) % 7]


# --- Actions -----------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A single tool call, ready to POST to /games/{id}/action."""
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}


@dataclass(frozen=True)
class ProposedAction:
    """A controller's suggestion, before composition + safety gating.

    `priority` lets the composer resolve conflicts deterministically
    (higher wins). `rationale` is for debugging/replay only.
    """
    action: Action
    priority: int = 0
    rationale: str = ""


# --- Regime ------------------------------------------------------------------

class Mode(str, Enum):
    """High-level operating mode set by the RegimeSupervisor."""
    NORMAL = "normal"
    SUPPLY_DEFENSIVE = "supply_defensive"   # supplier outage / unreliable
    DEMAND_SURGE = "demand_surge"           # tourist season etc.
    DEMAND_SLUMP = "demand_slump"
    ENDGAME = "endgame"                     # last few days: protect final rep


# --- Observation (typed view over the raw JSON) ------------------------------

@dataclass
class Observation:
    """Typed, defensively-parsed view of the raw observation JSON.

    Keep this a thin, lossless wrapper. Derived/estimated quantities belong
    in BeliefState, NOT here.
    """
    raw: dict[str, Any]

    @property
    def day(self) -> int: return int(self.raw["day"])
    @property
    def day_of_week(self) -> str: return self.raw["day_of_week"]
    @property
    def days_remaining(self) -> int: return int(self.raw["days_remaining"])
    @property
    def cash(self) -> float: return float(self.raw["cash"])
    @property
    def staff_level(self) -> int: return int(self.raw["staff_level"])
    @property
    def reputation_band(self) -> str: return self.raw.get("reputation_band", "")
    @property
    def customer_trend(self) -> str: return self.raw.get("customer_trend", "Stable")
    @property
    def weather_today(self) -> str: return self.raw.get("weather_today", "")
    @property
    def weather_forecast(self) -> list[str]: return self.raw.get("weather_forecast", [])
    @property
    def alerts(self) -> list[str]: return self.raw.get("alerts", [])
    @property
    def active_menu(self) -> list[str]: return list(self.raw.get("active_menu", []))
    @property
    def inventory(self) -> list[dict]: return self.raw.get("inventory", [])
    @property
    def supplier_catalog(self) -> list[dict]: return self.raw.get("supplier_catalog", [])
    @property
    def pending_orders(self) -> list[dict]: return self.raw.get("pending_orders", [])
    @property
    def delivery_history(self) -> list[dict]: return self.raw.get("delivery_history", [])
    @property
    def menu_book(self) -> list[dict]: return self.raw.get("menu_book", [])
    @property
    def recent_reviews(self) -> list[dict]: return self.raw.get("recent_reviews", [])
    @property
    def service_summary(self) -> dict: return self.raw.get("service_summary", {}) or {}
    @property
    def notes(self) -> str: return self.raw.get("notes", "") or ""

    def recipe(self, dish: str) -> dict | None:
        for m in self.menu_book:
            if m.get("name") == dish:
                return m
        return None


# --- BeliefState (the single source of truth) --------------------------------

@dataclass
class SupplierBelief:
    """Posterior reliability of one supplier (Beta over fill-ratio)."""
    name: str
    alpha: float = 2.0   # pseudo-successes (delivered as ordered)
    beta: float = 1.0    # pseudo-failures (short / failed deliveries)

    @property
    def reliability(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class BeliefState:
    """Everything the controllers are allowed to read, derived from the
    observation stream. Built fresh-but-carried each turn by BeliefEstimator.

    OWNER: belief workstream. Controllers READ this; only BeliefEstimator
    WRITES it.
    """
    day: int = 0
    mode: Mode = Mode.NORMAL

    # Demand model -----------------------------------------------------------
    # Estimated *true* (uncensored) covers, keyed by weekday name.
    weekday_covers: dict[str, float] = field(default_factory=dict)
    # Estimated true daily kg usage per ingredient (post-deconvolution).
    ingredient_daily_usage: dict[str, float] = field(default_factory=dict)
    # Multiplicative weather effect on demand, keyed by weather string.
    weather_factor: dict[str, float] = field(default_factory=dict)

    # Supply -----------------------------------------------------------------
    suppliers: dict[str, SupplierBelief] = field(default_factory=dict)

    # Reputation -------------------------------------------------------------
    # Continuous reconstruction (0..5-ish) of the hidden reputation scalar.
    reputation_est: float = 4.0
    reputation_trajectory: str = "Stable"   # "Rising"/"Stable"/"Falling"

    # Free-form scratch carried across turns (Python-side memory; the API
    # `notes` field is only needed for LLM-context continuity).
    memory: dict[str, Any] = field(default_factory=dict)

    def supplier(self, name: str) -> SupplierBelief:
        return self.suppliers.setdefault(name, SupplierBelief(name=name))
