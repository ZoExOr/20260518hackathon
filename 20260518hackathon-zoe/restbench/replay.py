"""ReplayStore — append-only JSONL log of full trajectories.

Determinism is our biggest asset: (seed, scenario, actions) reproduces
exactly. Logging every turn lets the offline analysis (tuning/analyze.py)
estimate elasticity, scenario signatures, etc. WITHOUT spending live games.

OWNER: tuning workstream.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ReplayStore:
    def __init__(self, run_dir: str = "replays"):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"run_{int(time.time())}.jsonl"
        self._fh = self.path.open("w")

    def log(self, scenario: str, seed: int, day: int,
            observation: dict[str, Any], actions: list[dict],
            day_result: dict | None) -> None:
        self._fh.write(json.dumps({
            "scenario": scenario, "seed": seed, "day": day,
            "observation": observation, "actions": actions,
            "day_result": day_result}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
