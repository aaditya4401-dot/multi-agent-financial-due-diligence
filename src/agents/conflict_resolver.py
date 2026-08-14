import logging

from pydantic import BaseModel, Field

from src.llm import Tier, get_llm
from src.models.schemas import ConflictModel
from src.state import AgentFindings, Conflict, DueDiligenceState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an analyst reviewing findings from independent research agents. "
    "Compare their findings and identify:\n"
    "- Factual contradictions (agents report different numbers for the same metric)\n"
    "- Complementary tensions (findings that are both true but create a nuanced "
    "picture, e.g. 'growing fast' vs 'burning cash')\n"
    "- Stale data conflicts (one agent has newer data than another)\n\n"
    "Only compare agents that actually produced findings. "
    "For each conflict, explain the resolution and assign a confidence score."
)

SECTIONS = [
    ("FINANCIAL", "financial_findings"),
    ("MARKET", "market_findings"),
    ("RISK", "risk_findings"),
    ("SENTIMENT", "sentiment_findings"),
]


class ConflictList(BaseModel):
    conflicts: list[ConflictModel] = Field(
        default_factory=list,
        description="List of detected conflicts. Empty list if no conflicts found.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_agent_findings(findings: AgentFindings) -> str:
    lines = [f"Agent: {findings['agent_name']}", f"Summary: {findings['summary'][:500]}"]
    for i, f in enumerate(findings["findings"], 1):
        lines.append(
            f"  {i}. [{f['source_quality']}, confidence={f['confidence']}, "
            f"date={f['date_of_data']}] {f['claim'][:300]}"
        )
    return "\n".join(lines)


def _usable_sections(state: DueDiligenceState) -> list[tuple[str, AgentFindings]]:
    """Sections from agents that ran successfully and produced findings."""
    usable = []
    for label, key in SECTIONS:
        findings = state.get(key)
        if findings and findings.get("ok", True) and findings.get("findings"):
            usable.append((label, findings))
    return usable


def _build_user_prompt(company: str, sections: list[tuple[str, AgentFindings]]) -> str:
    parts = [f"=== {label} FINDINGS ===\n{_format_agent_findings(f)}" for label, f in sections]
    return (
        f"Below are findings from independent due-diligence agents analyzing "
        f"{company}. Identify all conflicts between them.\n\n"
        + "\n\n".join(parts)
        + "\n\nReturn the list of conflicts. If there are no conflicts, return an empty list."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def conflict_resolver_node(state: DueDiligenceState) -> dict:
    """LangGraph node: detect and classify conflicts across agent findings."""
    sections = _usable_sections(state)

    # Conflicts require at least two sources to disagree — skip the call otherwise.
    if len(sections) < 2:
        logger.info(
            "Only %d usable agent result(s) — skipping conflict detection", len(sections)
        )
        return {"conflicts": []}

    try:
        structured_llm = get_llm(Tier.REASONING).with_structured_output(ConflictList)
        result: ConflictList = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state["company"], sections)},
        ])

        conflicts: list[Conflict] = [Conflict(**c.model_dump()) for c in result.conflicts]
        logger.info("Detected %d conflict(s)", len(conflicts))

    except Exception:
        logger.exception("Conflict resolver failed")
        conflicts = []

    return {"conflicts": conflicts}
