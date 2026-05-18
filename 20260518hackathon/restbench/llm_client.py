"""OpenAI-compatible client for the team's LiteLLM proxy."""
from __future__ import annotations

import os
from typing import Any

DEFAULT_LLM_BASE_URL = (
    "http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com"
)
DEFAULT_LLM_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4.1-mini")


def llm_api_key() -> str:
    """Read the LLM API key without hard-coding secrets in the repo."""
    key = os.environ.get("RESTBENCH_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set RESTBENCH_LLM_API_KEY or OPENAI_API_KEY before using an LLM agent."
        )
    return key


def make_openai_client(base_url: str | None = None) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional LLM dependency first: pip install openai"
        ) from exc
    return openai.OpenAI(
        api_key=llm_api_key(),
        base_url=base_url or os.environ.get("RESTBENCH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    )


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    timeout: float | None = 30.0,
) -> Any:
    client = make_openai_client()
    return client.chat.completions.create(
        model=model or DEFAULT_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
