"""LLMClient — thin LiteLLM wrapper. JSON-in, JSON-out, always falls back.

The advisors never block the agent: any failure (no API key, timeout,
parse error, missing key) returns the provided fallback dict. The agent
proceeds with deterministic behaviour.

Set AGENT_MODEL (default: anthropic/claude-haiku-4-5) and one of
ANTHROPIC_API_KEY / OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover
    litellm = None


_DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "anthropic/claude-haiku-4-5")
_DEFAULT_TIMEOUT = 12.0  # seconds per call; the API turn budget is 30s


def _strip_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in markdown fences despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # last-resort: grab the outermost {...}
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return text.strip()


class LLMClient:
    """Stateless. One instance per advisor or shared — doesn't matter."""

    def __init__(self, model: str | None = None, timeout: float = _DEFAULT_TIMEOUT):
        self.model = model or _DEFAULT_MODEL
        self.timeout = timeout

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        fallback: dict,
        max_tokens: int = 250,
        temperature: float = 0.2,
    ) -> dict:
        """Send a chat turn expecting a JSON object. Always returns a dict."""
        if litellm is None:
            return dict(fallback)
        has_key = (os.environ.get("ANTHROPIC_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
        if not has_key:
            return dict(fallback)
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            text = resp["choices"][0]["message"]["content"]
            data = json.loads(_strip_fences(text))
            if not isinstance(data, dict):
                return dict(fallback)
            # backfill any missing required keys from the fallback
            for k, v in fallback.items():
                data.setdefault(k, v)
            return data
        except Exception as e:                          # pragma: no cover
            # print, don't raise — the agent must always make a decision
            print(f"[advisor] {self.model} error: {e}; using fallback")
            return dict(fallback)