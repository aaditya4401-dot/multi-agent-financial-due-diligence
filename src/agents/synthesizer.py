import logging
from datetime import datetime

from src.llm import Tier, get_llm
from src.models.schemas import DueDiligenceReport, inconclusive_report
from src.state import AgentFindings, Conflict, DueDiligenceState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior investment analyst writing a due diligence report. "
    "Synthesize findings from financial, market, risk, and sentiment analyses. "
    "Account for detected conflicts and their resolutions. "
    "Assign an overall score (0-100), verdict, and risk level. "
    "Be concise but thorough.\n\n"
    "Some agents may have failed to run. Their sections are marked "
    "UNAVAILABLE. Treat an unavailable section as unknown, never as a "
    "negative finding: set its `available` field to false, leave its findings "
    "empty, list the agent in `unavailable_sections`, and lower "
    "`overall_confidence` to reflect the missing evidence. Do not lower "
    "`overall_score` because a section is missing. If no section ran "
    "successfully, return the verdict 'Inconclusive' and risk level 'Unknown'."
)

SECTIONS = [
    ("FINANCIAL", "financial_findings"),
    ("MARKET", "market_findings"),
    ("RISK", "risk_findings"),
    ("SENTIMENT", "sentiment_findings"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_findings(findings: AgentFindings | None) -> str:
    if not findings:
        return "UNAVAILABLE — agent did not run."
    if not findings.get("ok", True):
        return f"UNAVAILABLE — {findings.get('summary', 'agent failed')}"

    lines = [f"Agent: {findings['agent_name']}", f"Summary: {findings['summary'][:500]}"]
    for i, f in enumerate(findings["findings"], 1):
        lines.append(
            f"  {i}. [{f['source_quality']}, confidence={f['confidence']}, "
            f"date={f['date_of_data']}] {f['claim'][:300]}"
        )
    if not findings["findings"]:
        lines.append("  (ran successfully but surfaced no specific findings)")
    return "\n".join(lines)


def _format_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "(no conflicts detected)"
    return "\n".join(
        f"  {i}. [{c['type']}] {c['agent_a']} vs {c['agent_b']}: "
        f"'{c['claim_a'][:150]}' vs '{c['claim_b'][:150]}' — "
        f"Resolution: {c['resolution'][:200]} (confidence={c['resolved_confidence']})"
        for i, c in enumerate(conflicts, 1)
    )


def failed_agents(state: DueDiligenceState) -> list[str]:
    """Names of agents that errored out, for the report's unavailable list."""
    failed = []
    for label, key in SECTIONS:
        findings = state.get(key)
        if not findings or not findings.get("ok", True):
            failed.append(label.lower())
    return failed


def _build_user_prompt(state: DueDiligenceState) -> str:
    parts = [
        f"=== {label} FINDINGS ===\n{_format_findings(state.get(key))}"
        for label, key in SECTIONS
    ]
    parts.append(f"=== CONFLICTS DETECTED ===\n{_format_conflicts(state.get('conflicts', []))}")

    today = datetime.now().strftime("%Y-%m-%d")
    unavailable = failed_agents(state)
    note = (
        f"\n\nNOTE: these agents failed and their sections are unknown: "
        f"{', '.join(unavailable)}."
        if unavailable else ""
    )

    return (
        f"Produce a due diligence report for {state['company']} (report date: {today}).\n\n"
        + "\n\n".join(parts)
        + note
        + "\n\nSynthesize everything into a structured investment memo."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def synthesizer_node(state: DueDiligenceState) -> dict:
    """LangGraph node: synthesize all findings into a final DueDiligenceReport."""
    company = state["company"]
    today = datetime.now().strftime("%Y-%m-%d")
    unavailable = failed_agents(state)

    if len(unavailable) == len(SECTIONS):
        logger.error("All agents failed for %r — reporting inconclusive", company)
        report = inconclusive_report(
            company, today, "every research agent failed to produce findings"
        )
        return {"final_report": report.model_dump()}

    try:
        structured_llm = get_llm(Tier.SYNTHESIS).with_structured_output(DueDiligenceReport)
        result: DueDiligenceReport = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state)},
        ])

        # Trust our own failure tracking over the model's self-report.
        result.unavailable_sections = unavailable
        report = result

    except Exception as exc:
        logger.exception("Synthesizer failed for %r", company)
        report = inconclusive_report(company, today, f"report generation failed ({exc})")

    return {"final_report": report.model_dump()}
