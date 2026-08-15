import logging
from datetime import datetime

from src.claims.confidence import aggregate_confidence
from src.claims.models import Claim
from src.llm import Tier, get_llm
from src.models.schemas import DueDiligenceReport, inconclusive_report
from src.state import AgentFindings, Conflict, DueDiligenceState

logger = logging.getLogger(__name__)

#: Claims per section handed to the synthesizer, best evidence first. The old
#: prompt inlined every raw finding twice over; a ranked digest of typed claims
#: says more in a fraction of the context.
MAX_CLAIMS_PER_SECTION = 15

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


def claims_by_agent(state: DueDiligenceState) -> dict[str, list[Claim]]:
    """Group scored claims by the agent that produced them, best evidence first."""
    grouped: dict[str, list[Claim]] = {}
    for claim in state.get("claims") or []:
        grouped.setdefault(claim.extracted_by, []).append(claim)
    for claims in grouped.values():
        claims.sort(key=lambda c: c.confidence, reverse=True)
    return grouped


def section_confidences(state: DueDiligenceState) -> dict[str, float]:
    """Confidence per section, derived from its claims rather than guessed."""
    return {
        agent: aggregate_confidence(claims)
        for agent, claims in claims_by_agent(state).items()
    }


def _format_claims(claims: list[Claim]) -> str:
    lines = [
        f"  {i}. {c.describe()}\n     {c.assertion[:250]}"
        for i, c in enumerate(claims[:MAX_CLAIMS_PER_SECTION], 1)
    ]
    if len(claims) > MAX_CLAIMS_PER_SECTION:
        lines.append(f"  … and {len(claims) - MAX_CLAIMS_PER_SECTION} lower-confidence claim(s)")
    return "\n".join(lines)


def _build_user_prompt(state: DueDiligenceState) -> str:
    grouped = claims_by_agent(state)

    parts = []
    for label, key in SECTIONS:
        findings = state.get(key)
        agent = label.lower()

        if not findings or not findings.get("ok", True):
            parts.append(f"=== {label} ===\n{_format_findings(findings)}")
        elif grouped.get(agent):
            parts.append(
                f"=== {label} CLAIMS "
                f"(confidence {aggregate_confidence(grouped[agent]):.0%}) ===\n"
                f"{_format_claims(grouped[agent])}"
            )
        else:
            # No claims extracted — fall back to the raw findings.
            parts.append(f"=== {label} FINDINGS ===\n{_format_findings(findings)}")
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


def _apply_computed_confidence(
    report: DueDiligenceReport,
    state: DueDiligenceState,
    unavailable: list[str],
) -> None:
    """Replace the model's guessed confidences with values derived from evidence.

    Section confidence is the aggregate of that section's scored claims, and
    overall confidence is the mean across available sections. A model asked to
    rate its own certainty will produce a plausible number; this produces a
    traceable one.
    """
    computed = section_confidences(state)
    if not computed:
        return

    available: list[float] = []
    for agent in ("financial", "market", "risk", "sentiment"):
        section = getattr(report, f"{agent}_section", None)
        if section is None:
            continue

        if agent in unavailable:
            section.available = False
            section.section_confidence = 0.0
            continue

        if agent in computed:
            section.section_confidence = computed[agent]
            available.append(computed[agent])

    if available:
        report.overall_confidence = round(sum(available) / len(available), 4)


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

        # Trust our own bookkeeping over the model's self-report.
        result.unavailable_sections = unavailable
        _apply_computed_confidence(result, state, unavailable)
        report = result

    except Exception as exc:
        logger.exception("Synthesizer failed for %r", company)
        report = inconclusive_report(company, today, f"report generation failed ({exc})")

    return {"final_report": report.model_dump()}
