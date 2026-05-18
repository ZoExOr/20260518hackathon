"""LiteLLM logging defaults for command-line evaluations."""
from __future__ import annotations

import logging
import os


def silence_litellm() -> None:
    """Suppress LiteLLM provider hints and optional backend warnings."""
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "1")
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
