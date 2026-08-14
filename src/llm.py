"""Tiered LLM access.

Every node used to construct its own ``ChatOpenAI`` on each invocation, and
every node used the same top-tier model regardless of how hard its job was.
This module fixes both: clients are cached per (tier, temperature), and the
model behind each tier is configurable so cheap work can run on a cheap model.

Tiers:
    FAST       — mechanical, high-volume work (extraction, classification).
    REASONING  — research agents that must plan and interpret evidence.
    SYNTHESIS  — the final report, where quality matters most.

Override any tier without touching code::

    DD_MODEL_FAST=gpt-4o-mini
    DD_MODEL_REASONING=gpt-4o
    DD_MODEL_SYNTHESIS=gpt-4o
"""

import os
from enum import Enum
from functools import lru_cache

from langchain_openai import ChatOpenAI

REQUEST_TIMEOUT_SECONDS = float(os.getenv("DD_LLM_TIMEOUT", "90"))
MAX_RETRIES = int(os.getenv("DD_LLM_MAX_RETRIES", "2"))


class Tier(str, Enum):
    FAST = "fast"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


_DEFAULT_MODELS: dict[Tier, str] = {
    Tier.FAST: "gpt-4o-mini",
    Tier.REASONING: "gpt-4o",
    Tier.SYNTHESIS: "gpt-4o",
}

_ENV_OVERRIDES: dict[Tier, str] = {
    Tier.FAST: "DD_MODEL_FAST",
    Tier.REASONING: "DD_MODEL_REASONING",
    Tier.SYNTHESIS: "DD_MODEL_SYNTHESIS",
}


def model_name(tier: Tier) -> str:
    """Resolve the model id for *tier*, honouring env overrides."""
    return os.getenv(_ENV_OVERRIDES[tier], _DEFAULT_MODELS[tier])


@lru_cache(maxsize=None)
def get_llm(tier: Tier = Tier.REASONING, temperature: float = 0.0) -> ChatOpenAI:
    """Return a cached ``ChatOpenAI`` for *tier*.

    Cached on (tier, temperature), so repeated calls reuse one client and its
    underlying connection pool. ChatOpenAI is safe to share across tasks.
    """
    return ChatOpenAI(
        model=model_name(tier),
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
