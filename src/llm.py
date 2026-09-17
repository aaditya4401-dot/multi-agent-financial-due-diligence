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

    The tier is stamped onto the client as tags and metadata, which is what
    makes cost and latency groupable by tier in a trace. It belongs *here*
    rather than at the seven call sites for a reason worth stating: tier is a
    property of the client, not of the run. It is part of this cache key, so a
    given client always has exactly one tier, and stamping it once covers every
    call site — including the ReAct loop in ``agents/base.py``, whose nested
    model calls inherit it without that node having to know tracing exists.

    The same argument runs the other way for anything run-scoped. Per-run
    callbacks must *never* be attached here: this cache outlives the run, so
    one run's callbacks would fire for every later run. Those go through
    ``config`` at invoke time instead. See ``src/observability.py``.
    """
    return ChatOpenAI(
        model=model_name(tier),
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        # "routing_tier", not "tier": SourceTier in src/claims/models.py is an
        # unrelated evidence-quality grade, and one ambiguous "tier" column in
        # the trace UI would be worse than none.
        tags=[f"tier:{tier.value}"],
        metadata={"routing_tier": tier.value, "routing_model": model_name(tier)},
    )
