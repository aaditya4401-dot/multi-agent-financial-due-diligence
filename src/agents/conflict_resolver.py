"""Qualitative tension detection — the LLM's remaining share of conflict work.

Numeric disagreements are found deterministically in
:mod:`src.claims.contradictions`. What survives for a model to judge is the
genuinely qualitative case: two claims that are both true but sit in tension,
like "growing fast" against "burning cash". There is no arithmetic for that.

Scope is deliberately narrow. The old resolver was handed every finding from
every agent and asked to find three different kinds of conflict at once. This
sees only qualitative claims, ranked by confidence, and looks for one thing.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.claims.models import Claim
from src.llm import Tier, get_llm
from src.state import Conflict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You review sourced claims from independent due-diligence agents and find "
    "**complementary tensions**: pairs of claims that are both plausibly true "
    "but together paint a more nuanced picture than either alone. For example "
    "'revenue growing 40% YoY' alongside 'burn rate increased 60%', or "
    "'compliance issues flagged' alongside 'media coverage remains neutral'.\n\n"
    "Rules:\n"
    "- Do NOT report simple numeric disagreements. Those are detected elsewhere.\n"
    "- Do NOT invent tension between unrelated claims.\n"
    "- Only pair claims that are genuinely in tension with each other.\n"
    "- Quote each claim faithfully; never restate it as a stronger assertion.\n"
    "- `agent_a` and `agent_b` must be the bare agent name shown as "
    "`agent=...` on each claim, and nothing else.\n"
    "- If nothing is genuinely in tension, return an empty list. That is a "
    "valid and common answer."
)


AgentName = Literal["financial", "market", "risk", "sentiment"]


class Tension(BaseModel):
    agent_a: AgentName = Field(description="Agent that made claim_a.")
    agent_b: AgentName = Field(description="Agent that made claim_b.")
    claim_a: str
    claim_b: str
    resolution: str = Field(
        description="Why both can be true, and what the combination implies."
    )
    resolved_confidence: float = Field(ge=0.0, le=1.0)
    type: Literal["complementary_tension"] = "complementary_tension"


class TensionList(BaseModel):
    tensions: list[Tension] = Field(default_factory=list)


def _format_claims(claims: list[Claim]) -> str:
    # `agent=` is spelled out so the model copies the bare agent name into
    # agent_a/agent_b rather than the whole bracketed label.
    return "\n".join(
        f"{i}. (agent={c.extracted_by}; source={c.source_tier.value}; "
        f"confidence={c.confidence:.2f}) {c.assertion[:300]}"
        for i, c in enumerate(claims, 1)
    )


async def detect_tensions(subject: str, claims: list[Claim]) -> list[Conflict]:
    """Find complementary tensions among qualitative claims.

    Returns an empty list on any failure — tension detection is additive
    colour, never a reason to fail a run.
    """
    if len(claims) < 2:
        return []

    # Tensions are only interesting between different agents' perspectives.
    if len({c.extracted_by for c in claims}) < 2:
        return []

    try:
        structured = get_llm(Tier.REASONING).with_structured_output(TensionList)
        result: TensionList = await structured.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claims about {subject}:\n\n{_format_claims(claims)}\n\n"
                    "Identify complementary tensions between them."
                ),
            },
        ])
    except Exception:
        logger.exception("Tension detection failed for %r", subject)
        return []

    conflicts = [
        Conflict(
            type="complementary_tension",
            agent_a=t.agent_a,
            agent_b=t.agent_b,
            claim_a=t.claim_a,
            claim_b=t.claim_b,
            resolution=t.resolution,
            resolved_confidence=t.resolved_confidence,
        )
        for t in result.tensions
    ]
    logger.info("Detected %d complementary tension(s)", len(conflicts))
    return conflicts
