"""Reusable agent runner — plays a full game via the RestBench HTTP API.

Usage:
    from agents.runner import run_game
    from agents.naive_rule import strategy

    result = run_game(strategy, base_url="http://52.48.183.209:8001", team_name="naive", seed=42)
    print(result)

A strategy is a callable: (observation: dict, day: int) -> list[dict]
Each dict in the list is a tool call: {"tool": "place_order", "args": {...}}
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

import httpx

Strategy = Callable[[dict, int], list[dict]]

DEFAULT_URL = os.getenv("RESTBENCH_URL", "http://52.48.183.209:8001")


def _request_json(client: httpx.Client, method: str, path: str,
                  *, json_body: dict | None = None,
                  max_retries: int = 5) -> dict:
    """HTTP helper with 429 backoff.

    `agents.evaluate` maps any exception to -100000, so transient server
    throttling must be handled here instead of looking like a strategy failure.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = client.request(method, path, json=json_body)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    delay = min(60.0, max(1.0, float(retry_after)))
                else:
                    delay = min(60.0, 3.0 * (2 ** attempt))
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(min(15.0, 1.5 * (attempt + 1)))
    if last_error is not None:
        url = str(client.base_url).rstrip("/") + "/" + path.lstrip("/")
        raise RuntimeError(
            f"{method} {url} failed after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error
    raise RuntimeError(f"{method} {path} failed after retries")


def run_game(
    strategy: Strategy,
    *,
    base_url: str = DEFAULT_URL,
    team_name: str = "agent",
    scenario: str = "baseline",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(base_url=base_url, timeout=60.0, transport=transport) as client:
        data = _request_json(client, "POST", "/games", json_body={
            "team_name": team_name,
            "scenario": scenario,
            "seed": seed,
        })
        game_id = data["game_id"]
        observation = data["observation"]
        day = data["day"]

        if verbose:
            print(f"Game {game_id} created — Day {day}, Cash: {observation['cash']}")

        for turn in range(30):
            tool_calls = strategy(observation, day)

            accepted = 0
            rejected = 0
            for tc in tool_calls:
                result = _request_json(
                    client, "POST", f"/games/{game_id}/action", json_body=tc)
                if result["status"] == "accepted":
                    accepted += 1
                else:
                    rejected += 1
                    if verbose:
                        print(f"  Day {day}: REJECTED {tc['tool']}: {result['reason']}")

            turn_data = _request_json(client, "POST", f"/games/{game_id}/end-turn")

            observation = turn_data["observation"]
            day = turn_data["day"]
            status = turn_data["status"]
            dr = turn_data["day_result"]

            if verbose:
                print(
                    f"  Day {day-1}: covers={dr['total_covers']}, "
                    f"revenue={dr['total_revenue']}, "
                    f"cash={observation['cash']:.0f}, "
                    f"actions={accepted}ok/{rejected}rej"
                )

            if status != "in_progress":
                if verbose:
                    print(f"Game ended: {status}")
                break

        score_data = _request_json(client, "GET", f"/games/{game_id}/score")

        if verbose:
            s = score_data['score']
            print(f"\nFinal score: {s['total_score']}")
            print(f"  Net profit: {s['net_profit']}")
            print(f"  Satisfaction penalty: {s['satisfaction_penalty']}")
            print(f"  Reputation penalty: {s['reputation_penalty']}")
            print(f"  Walkout penalty: {s['walkout_penalty']}")
            print(f"  Waste penalty: {s['waste_penalty']}")
            print(f"  Days survived: {score_data['days_survived']}")
            print(f"  Final cash: {score_data['final_cash']}")

        return score_data


if __name__ == "__main__":
    print("Use: python -m agents.do_nothing / agents.naive_rule / agents.starter_template")
