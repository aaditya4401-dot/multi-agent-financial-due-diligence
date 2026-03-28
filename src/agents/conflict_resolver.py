import json
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.state import DueDiligenceState, AgentFindings, Conflict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an analyst reviewing findings from 4 independent research agents. "
    "Compare their findings and identify:\n"
    "- Factual contradictions (agents report different numbers for the same metric)\n"
    "- Complementary tensions (findings that are both true but create a nuanced "
    "picture, e.g. 'growing fast' vs 'burning cash')\n"
    "- Stale data conflicts (one agent has newer data than another)\n\n"
    "For each conflict, explain the resolution and assign a confidence score."
)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class ConflictItem(BaseModel):
    type: Literal["factual_contradiction", "complementary_tension", "stale_data"]
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    resolution: str
    resolved_confidence: float = Field(ge=0.0, le=1.0)


class ConflictList(BaseModel):
    conflicts: list[ConflictItem] = Field(
        default_factory=list,
        description="List of detected conflicts. Empty list if no conflicts found.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_agent_findings(findings: AgentFindings | None) -> str:
    """Format one agent's findings into a readable block for the LLM prompt."""
    if not findings:
        return "(no findings)"

    lines = [f"Agent: {findings['agent_name']}"]
    lines.append(f"Summary: {findings['summary'][:500]}")
    for i, f in enumerate(findings["findings"], 1):
        lines.append(
            f"  {i}. [{f['source_quality']}, confidence={f['confidence']}, "
            f"date={f['date_of_data']}] {f['claim'][:300]}"
        )
    return "\n".join(lines)


def _build_user_prompt(state: DueDiligenceState) -> str:
    sections = [
        ("FINANCIAL FINDINGS", state.get("financial_findings")),
        ("MARKET FINDINGS", state.get("market_findings")),
        ("RISK FINDINGS", state.get("risk_findings")),
        ("SENTIMENT FINDINGS", state.get("sentiment_findings")),
    ]
    parts = []
    for label, findings in sections:
        parts.append(f"=== {label} ===\n{_format_agent_findings(findings)}")

    return (
        "Below are findings from 4 independent due-diligence agents analyzing "
        f"{state['company']}. Identify all conflicts between them.\n\n"
        + "\n\n".join(parts)
        + "\n\nReturn the list of conflicts. If there are no conflicts, return an empty list."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def conflict_resolver_node(state: DueDiligenceState) -> dict:
    """LangGraph node: detect and classify conflicts across agent findings."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(ConflictList)

        result: ConflictList = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state)},
        ])

        conflicts: list[Conflict] = [
            Conflict(
                type=c.type,
                agent_a=c.agent_a,
                agent_b=c.agent_b,
                claim_a=c.claim_a,
                claim_b=c.claim_b,
                resolution=c.resolution,
                resolved_confidence=c.resolved_confidence,
            )
            for c in result.conflicts
        ]

    except Exception as exc:
        logger.error("Conflict resolver failed: %s", exc)
        conflicts = []

    return {"conflicts": conflicts}
