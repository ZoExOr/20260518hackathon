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


def _extract_total_score(final: dict) -> object:
    """Support both nested and flat score payloads."""
    if "score" in final and isinstance(final["score"], dict):
        return final["score"].get("total_score")
    return final.get("total_score")


def _format_day_debug(day_result: dict | None, submitted: list[dict]) -> str:
    """Compact per-day diagnostics for debugging collapses."""
    dr = day_result or {}
    costs = dr.get("cost_breakdown", {}) or {}
    revenue = dr.get("total_revenue", dr.get("revenue", 0.0))
    covers = dr.get("total_covers", 0)
    walk = dr.get("walkout_band", "n/a")
    avg_wait = dr.get("avg_wait_minutes", 0.0)
    peak_wait = dr.get("peak_wait_minutes", 0.0)
    stockouts = dr.get("dishes_unavailable_at", {}) or {}
    sold_out = ",".join(sorted(stockouts)) if stockouts else "-"
    total_cost = dr.get("total_costs", dr.get("yesterday_total_costs", 0.0))
    staff_cost = costs.get("staff", 0.0)
    waste_cost = costs.get("waste", 0.0)
    return (
        f"covers={covers:>3} rev={revenue:>7.0f} cost={total_cost:>7.0f} "
        f"staff_cost={staff_cost:>5.0f} waste={waste_cost:>4.0f} "
        f"walk={walk:<4} wait={avg_wait:>4.1f}/{peak_wait:>4.1f} "
        f"stockout={sold_out} actions={len(submitted):>2}"
    )


def _format_actions(submitted: list[dict]) -> str:
    if not submitted:
        return "[]"
    parts: list[str] = []
    for a in submitted:
        tool = a.get("tool", "?")
        args = a.get("args", {}) or {}
        if tool == "place_order":
            parts.append(
                f"place_order({args.get('ingredient')} {args.get('quantity_kg')}kg "
                f"from {args.get('supplier')})"
            )
        elif tool == "set_menu":
            dishes = args.get("dishes", []) or []
            preview = ", ".join(dishes[:5])
            if len(dishes) > 5:
                preview += ", ..."
            parts.append(f"set_menu({len(dishes)} dishes: {preview})")
        elif tool == "set_price":
            parts.append(f"set_price({args.get('dish')}={args.get('price')})")
        elif tool == "set_staff_level":
            parts.append(f"set_staff_level({args.get('level')})")
        elif tool == "set_marketing_spend":
            parts.append(f"set_marketing_spend({args.get('amount')})")
        elif tool == "offer_daily_special":
            parts.append(f"offer_daily_special({args.get('dish')})")
        elif tool == "save_notes":
            parts.append(f"save_notes({str(args.get('text', ''))[:60]})")
        else:
            parts.append(f"{tool}({args})")
    return "[" + "; ".join(parts) + "]"


def _format_pending(obs: Observation) -> str:
    if not obs.pending_orders:
        return "[]"
    parts = []
    for po in obs.pending_orders:
        parts.append(
            f"{po.get('ingredient')}:{po.get('quantity_kg')}kg"
            f"@d{po.get('delivery_day')} via {po.get('supplier')}"
        )
    return "[" + "; ".join(parts) + "]"


def _format_inventory(obs: Observation) -> str:
    if not obs.inventory:
        return "[]"
    rows = sorted(
        [
            (
                float(i.get("total_kg", 0) or 0),
                i.get("ingredient", "?"),
                min(
                    [int(b.get("expires_in_days", 9999)) for b in i.get("batches", [])]
                    or [9999]
                ),
            )
            for i in obs.inventory
        ],
        key=lambda x: (x[0], x[2]),
    )
    parts = [f"{name}:{kg:.1f}kg(exp{exp}d)" for kg, name, exp in rows[:10]]
    return "[" + "; ".join(parts) + "]"


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
            print(
                f"day {obs.day:>2} cash={obs.cash:>9.0f} "
                f"rep={obs.reputation_band:<10} status={status} "
                f"{_format_day_debug(turn.get('day_result'), submitted)}"
            )
            print(f"  submitted={_format_actions(submitted)}")
        if status != "in_progress":
            break
        obs = Observation(turn["observation"])
        if verbose:
            menu_preview = ", ".join(obs.active_menu[:8])
            if len(obs.active_menu) > 8:
                menu_preview += ", ..."
            print(
                f"  next_obs menu={len(obs.active_menu)} [{menu_preview}] "
                f"staff={obs.staff_level} trend={obs.customer_trend} "
                f"alerts={obs.alerts if obs.alerts else '[]'}"
            )
            print(f"  next_obs pending={_format_pending(obs)}")
            print(f"  next_obs inventory={_format_inventory(obs)}")

    final = client.score(game_id)
    if verbose:
        print(f"FINAL [{scenario}/{seed}] score={_extract_total_score(final)}")
    return final
