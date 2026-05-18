"""Thin HTTP client over the RestBench REST API.

The ONLY module that talks to the network. Everything else is offline-testable.

OWNER: infra/runner workstream.

Rate limits (per team): 10 concurrent games, 60 games/hour. The tuning
harness must respect this — see tuning/harness.py.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from .types import Action, Observation

DEFAULT_URL = os.environ.get("RESTBENCH_URL", "http://52.48.183.209:8001")


class RestBenchError(RuntimeError):
    pass


class RestBenchClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0,
                 max_retries: int = 3):
        self.base = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    # --- low level ---------------------------------------------------------
    def _req(self, method: str, path: str, json: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.request(method, url, json=json,
                                         timeout=self.timeout)
                if r.status_code == 429:                 # rate limited
                    time.sleep(2 ** attempt * 5)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:        # network/HTTP
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RestBenchError(f"{method} {path} failed: {last}")

    # --- game lifecycle ----------------------------------------------------
    def create_game(self, team_name: str, scenario: str = "baseline",
                     seed: int = 42) -> tuple[str, Observation, str]:
        d = self._req("POST", "/games", {
            "team_name": team_name, "scenario": scenario, "seed": seed})
        return d["game_id"], Observation(d["observation"]), d["status"]

    def submit_action(self, game_id: str, action: Action) -> dict:
        """Returns {"status": "accepted"} or {"status": "rejected", ...}.

        We expect the SafetyGate to make rejections rare; a rejection in
        production is a bug worth logging loudly.
        """
        return self._req("POST", f"/games/{game_id}/action",
                          action.to_payload())

    def end_turn(self, game_id: str) -> dict:
        """Returns {observation, day, status, day_result}."""
        return self._req("POST", f"/games/{game_id}/end-turn")

    def score(self, game_id: str) -> dict:
        return self._req("GET", f"/games/{game_id}/score")

    def observe(self, game_id: str) -> Observation:
        return Observation(self._req("GET", f"/games/{game_id}/observe"))
