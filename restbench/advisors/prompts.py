"""All advisor prompts in one place. EDIT HERE.

Each prompt follows the same discipline:
  * COMPACT input (no full observation dump — the LLM doesn't need it)
  * LABELLED output (never raw numbers; the deterministic code chooses
    the numbers consistent with the label)
  * EXAMPLE in the system message so the LLM has a few-shot anchor
  * FALLBACK is documented in the docstring of the calling advisor

These prompts are deliberately short. The 30-second per-turn API budget
means we can't afford long chats. Keep additions to the prompts
proportional — every extra line slows every turn.
"""
from __future__ import annotations

# ============================================================================
# Demand Advisor — chooses today's demand posture (replaces the simple
# `weak` flag in pricing.py).
# ============================================================================

DEMAND_SYSTEM = """\
You set today's demand posture for an Italian restaurant.

Output ONLY a JSON object: {"posture": "<label>", "why": "<≤12 words>"}.

Labels and when each applies:
  - "push"      : healthy cash + healthy reputation + (weekend OR sunny OR
                  customer_trend=Growing). Boost prices/specials/marketing.
  - "hold"      : DEFAULT. Use whenever in doubt. Use ESPECIALLY when:
                  cash < 4000, reputation in {Poor, Fair}, or any alert.
  - "discount"  : slow weekday + Very Good/Excellent reputation + low
                  covers yesterday. Drive traffic with happy hour + special.

Hard rules:
  * NEVER "push" when cash < 5000 OR reputation in {Poor, Fair}.
  * NEVER "discount" when reputation in {Poor, Fair}.

Example input:
  day=12/30 (Friday), cash=8200, rep=Very Good, trend=Stable,
  weather=sunny, forecast=[sunny,cloudy], walkout_band=None, alerts=[]
Example output:
  {"posture": "push", "why": "healthy rep, sunny Friday"}"""


def demand_user(obs, belief) -> str:
    ss = obs.service_summary or {}
    return (
        f"day={obs.day}/30 ({obs.day_of_week}), cash={obs.cash:.0f}, "
        f"rep={obs.reputation_band}, trend={obs.customer_trend}, "
        f"weather={obs.weather_today}, forecast={obs.weather_forecast}, "
        f"walkout_band={ss.get('walkout_band', 'None')}, "
        f"yesterday_covers={ss.get('total_covers', 0)}, "
        f"alerts={obs.alerts[:3]}"
    )


# ============================================================================
# Critic — final pass over the FULL proposed action list. Catches
# strategically-bad combinations the math checker and safety gate can't see.
# ============================================================================

CRITIC_SYSTEM = """\
You review proposed restaurant actions before submission.

You'll see today's state and a numbered list of proposed actions.

Output ONLY a JSON object:
{"veto": [<action_indices to drop>], "reason": "<≤25 words>"}.

Veto when ANY of these hold:
  * staff_level dropped below 6 on Friday/Saturday/Sunday
  * staff_level dropped below 5 when reputation is Poor or Fair
  * marketing_spend > 100 AND cash < 3000
  * place_order for an ingredient that already has cover_days > 7
  * set_menu reducing the menu below 6 dishes
  * set_price > base_price when reputation is Poor or Fair
  * run_happy_hour on three consecutive days (check the narrative)

When unsure: empty veto list, brief reason. Default to KEEP.

Example input:
  state: day=8, cash=4200, rep=Fair, day_of_week=Saturday
  actions:
    0: set_staff_level level=4
    1: set_marketing_spend amount=300
    2: place_order Fresh Farms NL Chicken 12kg (current cover=2.1d)
  narrative: "running happy hour 3 days in a row"
Example output:
  {"veto": [0, 1], "reason": "staff too low for Saturday; marketing too high with thin cash"}"""


def critic_user(obs, belief, actions, narrative: str) -> str:
    state = (
        f"day={obs.day}/30, cash={obs.cash:.0f}, rep={obs.reputation_band}, "
        f"day_of_week={obs.day_of_week}, trend={obs.customer_trend}"
    )
    lines = [f"state: {state}", f"narrative: {narrative}", "actions:"]
    for i, a in enumerate(actions):
        args = ", ".join(f"{k}={v}" for k, v in a.args.items())
        lines.append(f"  {i}: {a.tool} {args}")
    return "\n".join(lines)


# ============================================================================
# Research Advisor — maintains a short narrative for the save_notes field.
# The Regime and Critic both see this on subsequent turns.
# ============================================================================

RESEARCH_SYSTEM = """\
You maintain a strategic notebook for a restaurant management agent.

Each turn you see: yesterday's results + a few rolling signals.
Output ONLY a JSON object:
{
  "today_focus": "<one sentence: the single most important thing today>",
  "risks":       ["<at most 3 risks for tomorrow>"],
  "streak":      {"happy_hour_days": <int>, "marketing_days": <int>}
}

Be terse. The whole output should fit in ~250 characters.

Example input:
  day=6, rev_yesterday=1830, walk=Few, rep=Good, trend=Stable,
  stockouts=["Salmon at hour 19"], waste_yesterday=80,
  prior_narrative="day 5: pushing margin"
Example output:
  {
    "today_focus": "reorder salmon before Friday peak; keep margin",
    "risks": ["salmon stockout Fri", "waste creeping up"],
    "streak": {"happy_hour_days": 0, "marketing_days": 1}
  }"""


def research_user(obs, belief, prior_narrative: str) -> str:
    ss = obs.service_summary or {}
    stockouts = list((ss.get("dishes_unavailable_at") or {}).keys())
    waste = (obs.raw.get("cost_breakdown") or {}).get("waste", 0)
    return (
        f"day={obs.day}, rev_yesterday={obs.raw.get('yesterday_revenue', 0):.0f}, "
        f"walk={ss.get('walkout_band', 'None')}, rep={obs.reputation_band}, "
        f"trend={obs.customer_trend}, stockouts={stockouts}, "
        f"waste_yesterday={waste:.0f}, prior_narrative={prior_narrative!r}"
    )