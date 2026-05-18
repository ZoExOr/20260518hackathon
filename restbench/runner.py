"""Runner — plays one full game.

OWNER: infra/runner workstream.

    create_game -> loop(decide -> submit each action -> end_turn) -> score

Submits each action individually (the API takes one tool call per POST).
A rejected action is logged loudly: with the SafetyGate in front, rejections
should be ~impossible, so one means a real bug.
"""
from __future__ import annotations

import sys

from .api import RestBenchClient
from .agent import Agent
from .params import Params
from .replay import ReplayStore
from .types import Observation


def play_game(team_name: str, scenario: str = "baseline", seed: int = 42,
              params: Params | None = None, regime=None,
              client: RestBenchClient | None = None,
              replay: ReplayStore | None = None, verbose: bool = True
              ) -> dict:
    client = client or RestBenchClient()
    agent = Agent(params=params, regime=regime)

    game_id, obs, status = client.create_game(team_name, scenario, seed)
    while status == "in_progress":
        actions = agent.decide(obs)
        submitted = []
        for a in actions:
            resp = client.submit_action(game_id, a)
            submitted.append(a.to_payload())
            if resp.get("status") == "rejected" and verbose:
                print(f"[REJECTED] day {obs.day} {a.tool} "
                      f"{a.args} -> {resp.get('reason')}", file=sys.stderr)

        turn = client.end_turn(game_id)
        if replay:
            replay.log(scenario, seed, obs.day, obs.raw, submitted,
                       turn.get("day_result"))
        status = turn["status"]
        if verbose:
            print(f"day {obs.day:>2} cash={obs.cash:>9.0f} "
                  f"rep={obs.reputation_band:<10} status={status}")
        if status != "in_progress":
            break
        obs = Observation(turn["observation"])

    final = client.score(game_id)
    if verbose:
        print(f"FINAL [{scenario}/{seed}] score={final.get('total_score')}")
    return final
