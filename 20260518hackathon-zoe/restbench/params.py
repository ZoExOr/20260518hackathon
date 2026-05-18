"""The tunable parameter vector.

This is deliberately SMALL. Every field here is a knob the offline tuner
(BO / CMA-ES) is allowed to search. Keep the dimensionality low (~12-20):
the interaction budget is a few hundred episodes total, so a 40-dim search
is infeasible. If you want to add a knob, ask whether a sane constant would
do instead.

OWNER: tuning workstream owns the search; each controller owner owns the
*meaning* and sane default of their own fields.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields


@dataclass
class Params:
    # --- Safety (survival) -------------------------------------------------
    cash_reserve_floor: float = 2500.0     # never let projected cash drop below this
    max_order_cash_frac: float = 0.35      # cap a single turn's ordering spend

    # --- Supply / (s,S) inventory -----------------------------------------
    reorder_days: float = 3.0              # s: reorder when cover < this many days
    target_days: float = 7.0              # S: top up to this many days of cover
    safety_days: float = 2.0              # extra buffer added to S
    reliability_inflation: float = 1.0     # extra stock per unit (1/reliability-1)
    waste_aversion: float = 1.0            # >1 shrinks orders near shelf-life limits
    delivery_guard_days: float = 1.0       # reorder earlier when next feasible delivery is far away
    bootstrap_full_menu_days: int = 4      # early uncertainty: provision for the likely widened menu
    cold_start_covers: float = 120.0       # conservative prior covers before belief warms up

    # --- Operations / staffing --------------------------------------------
    covers_per_staff: float = 16.0         # heuristic kitchen throughput per staff
    staff_min: int = 4
    staff_max: int = 11
    weekend_staff_bonus: int = 1

    # --- Pricing / promos --------------------------------------------------
    base_price_mult: float = 1.0           # global price multiplier (0.8..1.2)
    price_mult_good_demand: float = 1.08   # healthy demand/service -> margin
    price_mult_bad_service: float = 1.0    # protect service/reputation
    marketing_default: float = 0.0
    marketing_slump: float = 200.0         # marketing when trend is Declining
    marketing_recovery: float = 180.0      # extra push when demand is weak
    happy_hour_on_weak_days: bool = True
    happy_hour_max_peak_wait: float = 14.0 # avoid promos when ops are strained

    # --- Regime ------------------------------------------------------------
    llm_every_k_days: int = 5              # 0 disables the LLM supervisor
    endgame_window: int = 4               # last N days -> ENDGAME mode

    # --- Vector <-> dataclass for the optimizer ---------------------------
    # Only float/int fields are exposed to the continuous optimizer; bools
    # and the few structural ints stay fixed unless you add encoding for them.
    TUNABLE = (
        "cash_reserve_floor", "max_order_cash_frac", "reorder_days",
        "target_days", "safety_days", "reliability_inflation",
        "waste_aversion", "covers_per_staff", "base_price_mult",
        "price_mult_good_demand", "price_mult_bad_service",
        "marketing_slump", "marketing_recovery",
    )

    def to_vector(self) -> list[float]:
        return [float(getattr(self, k)) for k in self.TUNABLE]

    @classmethod
    def from_vector(cls, vec: list[float], base: "Params | None" = None) -> "Params":
        p = base or cls()
        d = asdict(p)
        for k, v in zip(cls.TUNABLE, vec):
            d[k] = v
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name != "TUNABLE"})

    @staticmethod
    def bounds() -> dict[str, tuple[float, float]]:
        """Search bounds for the tunable fields (used by BO/CMA-ES)."""
        return {
            "cash_reserve_floor": (1000.0, 5000.0),
            "max_order_cash_frac": (0.15, 0.6),
            "reorder_days": (1.5, 6.0),
            "target_days": (4.0, 12.0),
            "safety_days": (0.5, 4.0),
            "reliability_inflation": (0.0, 3.0),
            "waste_aversion": (0.5, 2.5),
            "covers_per_staff": (10.0, 24.0),
            "base_price_mult": (0.85, 1.15),
            "price_mult_good_demand": (1.0, 1.15),
            "price_mult_bad_service": (0.9, 1.02),
            "marketing_slump": (0.0, 500.0),
            "marketing_recovery": (0.0, 350.0),
        }
