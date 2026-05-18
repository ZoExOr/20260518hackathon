"""Optional LLM multi-agent advice layer.

The core agent remains deterministic by default. This module adds a bounded
LLM layer that can suggest actions/overrides, while the existing coordinator
and SafetyGate keep final authority.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .llm_client import DEFAULT_LLM_MODEL, chat_completion
from .params import Params
from .types import Action, BeliefState, Observation, ProposedAction
from .controllers.base import serviceability_risk

ALLOWED_TOOLS = {
    "place_order", "set_staff_level", "set_menu", "set_price",
    "set_marketing_spend", "run_happy_hour", "offer_daily_special",
    "save_notes",
}

class LLMClient(Protocol):
    def complete_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class OpenAIProxyClient:
    def __init__(self, model: str | None = None, timeout: float = 30.0):
        self.model = model or DEFAULT_LLM_MODEL
        self.timeout = timeout

    def complete_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ],
            model=self.model,
            temperature=0.2,
            max_tokens=1200,
            timeout=self.timeout,
        )
        content = response.choices[0].message.content
        return parse_json_object(content)


@dataclass
class AgentAdvice:
    role: str
    summary: str = ""
    proposed_actions: list[Action] = field(default_factory=list)
    param_overrides: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["proposed_actions"] = [a.to_payload() for a in self.proposed_actions]
        return d


@dataclass
class RiskReview:
    approved_actions: list[Action] = field(default_factory=list)
    dropped_actions: list[dict[str, Any]] = field(default_factory=list)
    adjusted_actions: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_actions": [a.to_payload() for a in self.approved_actions],
            "dropped_actions": self.dropped_actions,
            "adjusted_actions": self.adjusted_actions,
            "risk_score": self.risk_score,
            "rationale": self.rationale,
        }


@dataclass
class MultiAgentTrace:
    advice: list[AgentAdvice] = field(default_factory=list)
    risk_review: RiskReview | None = None
    deterministic_risk: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "advice": [a.to_dict() for a in self.advice],
            "risk_review": self.risk_review.to_dict() if self.risk_review else None,
            "deterministic_risk": list(self.deterministic_risk),
            "errors": list(self.errors),
        }


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a single JSON object from raw LLM text."""
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data


def _action_from_payload(payload: Any, obs: Observation) -> Action | None:
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    args = payload.get("args", {})
    if tool not in ALLOWED_TOOLS or not isinstance(args, dict):
        return None

    if tool == "place_order":
        supplier = str(args.get("supplier", ""))
        ingredient = str(args.get("ingredient", ""))
        valid_supplier = any(
            s.get("name") == supplier and ingredient in s.get("ingredients", {})
            for s in obs.supplier_catalog
        )
        if not valid_supplier:
            return None
        return Action(tool, {
            "supplier": supplier,
            "ingredient": ingredient,
            "quantity_kg": float(args.get("quantity_kg", 0)),
        })
    if tool == "set_staff_level":
        return Action(tool, {"level": int(round(float(args.get("level", obs.staff_level))))})
    if tool == "set_menu":
        dishes = [str(d) for d in args.get("dishes", []) if obs.recipe(str(d))]
        return Action(tool, {"dishes": dishes}) if dishes else None
    if tool == "set_price":
        dish = str(args.get("dish", ""))
        if not obs.recipe(dish):
            return None
        return Action(tool, {"dish": dish, "price": float(args.get("price", 0))})
    if tool == "set_marketing_spend":
        return Action(tool, {"amount": float(args.get("amount", 0))})
    if tool == "offer_daily_special":
        dish = str(args.get("dish", ""))
        return Action(tool, {"dish": dish}) if obs.recipe(dish) else None
    if tool == "save_notes":
        return Action(tool, {"text": str(args.get("text", ""))})
    return Action(tool, {})


def parse_agent_advice(role: str, data: dict[str, Any], obs: Observation) -> AgentAdvice:
    actions = []
    for raw in data.get("proposed_actions", []) or []:
        action = _action_from_payload(raw, obs)
        if action:
            actions.append(action)

    overrides = {}
    if isinstance(data.get("param_overrides"), dict):
        for key, value in data["param_overrides"].items():
            if key in Params.TUNABLE:
                overrides[key] = value

    return AgentAdvice(
        role=str(data.get("role") or role),
        summary=str(data.get("summary", ""))[:1000],
        proposed_actions=actions,
        param_overrides=overrides,
        constraints=[str(x)[:300] for x in (data.get("constraints", []) or [])[:8]],
        risks=[str(x)[:300] for x in (data.get("risks", []) or [])[:8]],
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
    )


def parse_risk_review(data: dict[str, Any], obs: Observation,
                      fallback: list[ProposedAction]) -> RiskReview:
    approved = []
    raw_actions = data.get("approved_actions")
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            action = _action_from_payload(raw, obs)
            if action:
                approved.append(action)
    else:
        approved = [p.action for p in fallback]

    return RiskReview(
        approved_actions=approved,
        dropped_actions=list(data.get("dropped_actions", []) or [])[:20],
        adjusted_actions=list(data.get("adjusted_actions", []) or [])[:20],
        risk_score=max(0.0, min(1.0, float(data.get("risk_score", 0.0) or 0.0))),
        rationale=str(data.get("rationale", ""))[:1200],
    )


def compact_context(obs: Observation, belief: BeliefState,
                    params: Params) -> dict[str, Any]:
    inventory = []
    for inv in obs.inventory:
        batches = inv.get("batches", []) or []
        earliest_exp = min([int(b.get("expires_in_days", 9999)) for b in batches] or [9999])
        inventory.append({
            "ingredient": inv.get("ingredient"),
            "total_kg": inv.get("total_kg", 0),
            "earliest_expires_in_days": earliest_exp,
        })
    return {
        "day": obs.day,
        "day_of_week": obs.day_of_week,
        "days_remaining": obs.days_remaining,
        "cash": round(obs.cash, 2),
        "staff_level": obs.staff_level,
        "reputation_band": obs.reputation_band,
        "customer_trend": obs.customer_trend,
        "weather_today": obs.weather_today,
        "weather_forecast": obs.weather_forecast,
        "alerts": obs.alerts,
        "active_menu": obs.active_menu,
        "inventory": inventory,
        "pending_orders": obs.pending_orders,
        "delivery_history_tail": obs.delivery_history[-10:],
        "supplier_catalog": obs.supplier_catalog,
        "service_summary": obs.service_summary,
        "recent_reviews": obs.recent_reviews[-10:],
        "belief": {
            "mode": belief.mode.value,
            "weekday_covers": belief.weekday_covers,
            "ingredient_daily_usage": belief.ingredient_daily_usage,
            "reputation_est": round(belief.reputation_est, 2),
            "reputation_trajectory": belief.reputation_trajectory,
            "supplier_reliability": {
                name: round(sb.reliability, 3)
                for name, sb in belief.suppliers.items()
            },
        },
        "params": {k: getattr(params, k) for k in Params.TUNABLE},
        "supply_risk": serviceability_risk(obs, belief, params).to_dict(),
    }


def actions_payload(proposals: list[ProposedAction]) -> list[dict[str, Any]]:
    return [
        {
            "tool": p.action.tool,
            "args": p.action.args,
            "priority": p.priority,
            "rationale": p.rationale,
        }
        for p in proposals
    ]


class RoleLLMAgent:
    def __init__(self, role: str, system_prompt: str,
                 client: LLMClient | None = None):
        self.role = role
        self.system_prompt = system_prompt
        self.client = client or OpenAIProxyClient()

    def advise(self, obs: Observation, belief: BeliefState, params: Params,
               baseline: list[ProposedAction]) -> AgentAdvice:
        payload = {
            "role": self.role,
            "context": compact_context(obs, belief, params),
            "baseline_proposals": actions_payload(baseline),
            "output_schema": {
                "role": self.role,
                "summary": "short string",
                "proposed_actions": [{"tool": "set_staff_level", "args": {"level": 6}}],
                "param_overrides": {"target_days": 8.0},
                "constraints": ["short string"],
                "risks": ["short string"],
                "confidence": 0.0,
            },
        }
        data = self.client.complete_json(self.system_prompt, payload)
        return parse_agent_advice(self.role, data, obs)


class RiskLLMAgent:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or OpenAIProxyClient()
        self.system_prompt = (
            "You are the final risk reviewer for a restaurant simulation agent. "
            "Return ONLY JSON. You may drop or adjust obviously risky actions, "
            "but do not invent new strategies. Preserve safe baseline actions. "
            "The deterministic SafetyGate will run after you."
        )

    def review(self, obs: Observation, belief: BeliefState, params: Params,
               composed: list[ProposedAction]) -> RiskReview:
        payload = {
            "context": compact_context(obs, belief, params),
            "candidate_actions": actions_payload(composed),
            "checks": [
                "duplicate pending orders",
                "cash/overhead/order spend risk",
                "menu too narrow or invalid dish names",
                "stockout and waste pressure",
                "pricing/staffing conflicts",
            ],
            "output_schema": {
                "approved_actions": [{"tool": "set_staff_level", "args": {"level": 6}}],
                "dropped_actions": [{"tool": "place_order", "reason": "duplicate"}],
                "adjusted_actions": [{"from": {}, "to": {}, "reason": ""}],
                "risk_score": 0.0,
                "rationale": "short string",
            },
        }
        data = self.client.complete_json(self.system_prompt, payload)
        return parse_risk_review(data, obs, composed)


SUPPLY_PROMPT = (
    "You are SupplyLLMAgent. Focus only on inventory, expiry, pending orders, "
    "supplier reliability, lead time and delivery days. Return ONLY JSON. "
    "Read context.supply_risk first. Prefer conservative suggestions that "
    "restore at least five serviceable dishes and reduce stockout/waste risk."
)
DEMAND_PROMPT = (
    "You are DemandLLMAgent. Focus only on weather, weekday demand, prices, "
    "marketing, happy hour, daily specials, reviews and demand trend. "
    "Return ONLY JSON. Read context.supply_risk first. Do not stimulate demand "
    "when supply is constrained; use small reversible suggestions."
)
OPERATIONS_PROMPT = (
    "You are OperationsLLMAgent. Focus only on staff, waits, walkouts, table "
    "utilization and kitchen bottlenecks. Return ONLY JSON. Balance service "
    "quality against staff cost. Many walkouts require protective staffing "
    "unless supply risk is critical."
)


class MultiAgentAdvisor:
    def __init__(self, client: LLMClient | None = None,
                 *, allow_fallback: bool = True):
        self.allow_fallback = allow_fallback
        self.supply = RoleLLMAgent("supply", SUPPLY_PROMPT, client)
        self.demand = RoleLLMAgent("demand", DEMAND_PROMPT, client)
        self.operations = RoleLLMAgent("operations", OPERATIONS_PROMPT, client)
        self.risk = RiskLLMAgent(client)

    def collect_advice(self, obs: Observation, belief: BeliefState,
                       params: Params,
                       baseline: dict[str, list[ProposedAction]]
                       ) -> tuple[list[AgentAdvice], list[str]]:
        out: list[AgentAdvice] = []
        errors: list[str] = []
        for role, agent in (
            ("supply", self.supply),
            ("demand", self.demand),
            ("operations", self.operations),
        ):
            try:
                baseline_key = "pricing" if role == "demand" else role
                advice = agent.advise(
                    obs, belief, params, baseline.get(baseline_key, []))
                out.append(advice)
            except Exception as exc:
                if not self.allow_fallback:
                    raise
                errors.append(f"{role}: {exc}")
        return out, errors

    def review(self, obs: Observation, belief: BeliefState, params: Params,
               composed: list[ProposedAction]) -> tuple[RiskReview | None, list[str]]:
        try:
            return self.risk.review(obs, belief, params, composed), []
        except Exception as exc:
            if not self.allow_fallback:
                raise
            return None, [f"risk: {exc}"]


def advice_to_proposals(advice: AgentAdvice) -> list[ProposedAction]:
    priority = 35 if advice.confidence >= 0.7 else 25
    return [
        ProposedAction(
            action,
            priority=priority,
            rationale=f"llm:{advice.role}: {advice.summary}"[:300],
        )
        for action in advice.proposed_actions
    ]
