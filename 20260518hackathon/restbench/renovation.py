"""Shared renovation scenario helpers."""
from __future__ import annotations

from .types import BeliefState, Observation

RENOVATION_RECOVERY_DAYS = 4
RENOVATION_PREP_DAYS = 2


def renovation_until(belief: BeliefState) -> int:
    return int(belief.memory.get("capacity_reduced_until", 0) or 0)


def renovation_active(obs: Observation, belief: BeliefState) -> bool:
    return renovation_until(belief) >= obs.day


def renovation_recovery(obs: Observation, belief: BeliefState) -> bool:
    until = renovation_until(belief)
    return until > 0 and until < obs.day <= until + RENOVATION_RECOVERY_DAYS


def renovation_prep_or_recovery(obs: Observation, belief: BeliefState) -> bool:
    until = renovation_until(belief)
    return until > 0 and until - RENOVATION_PREP_DAYS <= obs.day <= \
        until + RENOVATION_RECOVERY_DAYS
