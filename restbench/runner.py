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
              replay: ReplayStore | None = None, verbose: bool = True,
              use_advisors: bool = False
              ) -> dict:
    client = client or RestBenchClient()
    agent = Agent(params=params, regime=regime, use_advisors=use_advisors)

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
<<<<<<< Updated upstream
            print(f"day {obs.day:>2} cash={obs.cash:>9.0f} "
                  f"rep={obs.reputation_band:<10} status={status}")
=======
            # Print AFTER end_turn using the NEW observation so we see
            # the result of THIS day's actions. cost_breakdown and
            # service_summary describe what JUST happened.
            new = turn.get("observation", {}) or {}
            cb = new.get("cost_breakdown", {}) or {}
            ss = new.get("service_summary", {}) or {}
            stockouts = (ss.get("dishes_unavailable_at") or {})
            menu = new.get("active_menu", []) or []
            stockout_str = (",".join(stockouts.keys())[:30] or "-")
            print(
                f"day {obs.day:>2} -> "
                f"cash={new.get('cash', obs.cash):>8.0f} "
                f"rev={new.get('yesterday_revenue', 0):>5.0f} "
                f"waste={cb.get('waste', 0):>5.1f} "
                f"mkt={cb.get('marketing', 0):>4.0f} "
                f"staff={cb.get('staff', 0):>5.0f} "
                f"walk={ss.get('walkout_band', '?'):<5} "
                f"rep={new.get('reputation_band', '?'):<10} "
                f"menu={len(menu)} stockout=[{stockout_str}] "
                f"status={status}"
            )
>>>>>>> Stashed changes
        if status != "in_progress":
            break
        obs = Observation(turn["observation"])

    final = client.score(game_id)
    if verbose:
<<<<<<< Updated upstream
        print(f"FINAL [{scenario}/{seed}] score={final.get('total_score')}")
    return final
=======
        s = final.get("score", {})
        print(f"\nFINAL [{scenario}/{seed}] score={s.get('total_score', final.get('total_score'))}")
        if s:
            print(f"  Net profit:           {s.get('net_profit')}")
            print(f"  Satisfaction penalty: {s.get('satisfaction_penalty')}")
            print(f"  Reputation penalty:   {s.get('reputation_penalty')}")
            print(f"  Walkout penalty:      {s.get('walkout_penalty')}")
            print(f"  Waste penalty:        {s.get('waste_penalty')}")
            print(f"  Days survived:        {final.get('days_survived')}")
            print(f"  Final cash:           {final.get('final_cash')}")
    return final
>>>>>>> Stashed changes
