import logging
from datetime import datetime

from src.claims.confidence import aggregate_confidence
from src.claims.models import Claim
from src.llm import Tier, get_llm
from src.models.schemas import DueDiligenceReport, FinalReport, inconclusive_report
from src.models.thesis import InvestmentThesis
from src.planning.plan import ALL_AGENTS
from src.state import AgentFindings, Conflict, DueDiligenceState, findings_for

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
    "Some sections may be missing, for two different reasons, and they are not "
    "interchangeable:\n"
    "- UNAVAILABLE means the agent ran and failed. Treat it as unknown, never "
    "as a negative finding: set `available` to false, leave its findings "
    "empty, list the agent in `unavailable_sections`, and lower "
    "`overall_confidence` to reflect the missing evidence.\n"
    "- NOT RESEARCHED means the section was deliberately left out of the "
    "research plan because it was not relevant to this company. List the agent "
    "in `skipped_sections`. Do NOT lower `overall_confidence` for it — nothing "
    "is missing that we wanted.\n"
    "Never lower `overall_score` because a section is absent for either "
    "reason. If no section produced findings, return the verdict "
    "'Inconclusive' and risk level 'Unknown'.\n\n"
    "Where a section carries a CAVEAT, honour it in your prose: do not "
    "describe an estimate as a reported figure.\n\n"
    "An investment thesis has already been formed from the evidence and is "
    "shown to you. Your executive summary must be consistent with it — do not "
    "contradict its recommendation or invent drivers it does not contain."
)

SECTIONS = [
    ("FINANCIAL", "financial"),
    ("MARKET", "market"),
    ("RISK", "risk"),
    ("SENTIMENT", "sentiment"),
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


def planned_agents(state: DueDiligenceState) -> set[str]:
    """Agents the plan asked for.

    With no plan in state — a directly-constructed state, or a run from before
    the planner existed — every agent counts as planned, so a missing section
    reads as a failure exactly as it used to.
    """
    plan = state.get("plan")
    return set(plan.agents) if plan else set(ALL_AGENTS)


def agent_status(state: DueDiligenceState) -> tuple[list[str], list[str]]:
    """Split absent sections into ``(failed, skipped)``.

    These must not be conflated. A failed agent is evidence we wanted and did
    not get, so it costs confidence. A skipped agent was never asked for, so it
    costs nothing — penalising it would mean the system scores itself down for
    following its own plan.
    """
    planned = planned_agents(state)
    failed: list[str] = []
    skipped: list[str] = []

    for _, agent in SECTIONS:
        if agent not in planned:
            skipped.append(agent)
            continue
        blocks = findings_for(state, agent)
        if not blocks or not any(b.get("ok", True) for b in blocks):
            failed.append(agent)

    return failed, skipped


def failed_agents(state: DueDiligenceState) -> list[str]:
    """Names of agents that ran and errored, for the report's unavailable list."""
    return agent_status(state)[0]


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


def _caveat_for(state: DueDiligenceState, agent: str) -> str:
    """The planner's caveat for *agent*, preferring what the run recorded."""
    for block in findings_for(state, agent):
        if block.get("caveat"):
            return block["caveat"]
    plan = state.get("plan")
    return plan.caveat_for(agent) if plan else ""


def _format_thesis(thesis: InvestmentThesis | None) -> str:
    """The already-formed view, so the memo is written to match it."""
    if thesis is None:
        return "(no thesis was formed)"

    lines = [f"Recommendation: {thesis.recommendation}", f"Thesis: {thesis.summary}"]
    for driver in thesis.drivers:
        lines.append(
            f"  - [{driver.direction}] {driver.statement} "
            f"(confidence {driver.confidence:.0%})"
        )
    for flag in thesis.deal_breakers:
        lines.append(f"  - DEAL BREAKER: {flag.issue}")
    return "\n".join(lines)


def _build_user_prompt(state: DueDiligenceState) -> str:
    grouped = claims_by_agent(state)
    _, skipped = agent_status(state)

    parts = []
    for label, agent in SECTIONS:
        if agent in skipped:
            plan = state.get("plan")
            why = plan.rationale if plan else ""
            parts.append(
                f"=== {label} ===\nNOT RESEARCHED — deliberately left out of "
                f"the research plan. {why}"
            )
            continue

        blocks = findings_for(state, agent)
        primary = blocks[0] if blocks else None
        caveat = _caveat_for(state, agent)
        header = f"=== {label}"

        if not blocks or not any(b.get("ok", True) for b in blocks):
            parts.append(f"{header} ===\n{_format_findings(primary)}")
            continue

        if grouped.get(agent):
            body = (
                f"{header} CLAIMS "
                f"(confidence {aggregate_confidence(grouped[agent]):.0%}) ===\n"
                f"{_format_claims(grouped[agent])}"
            )
        else:
            # No claims extracted — fall back to the raw findings.
            body = f"{header} FINDINGS ===\n" + "\n".join(
                _format_findings(b) for b in blocks
            )

        if caveat:
            body += f"\n  CAVEAT: {caveat}"
        parts.append(body)

    parts.append(f"=== CONFLICTS DETECTED ===\n{_format_conflicts(state.get('conflicts', []))}")
    parts.append(f"=== INVESTMENT THESIS (already formed) ===\n"
                 f"{_format_thesis(state.get('thesis'))}")

    today = datetime.now().strftime("%Y-%m-%d")
    unavailable = failed_agents(state)
    note = (
        f"\n\nNOTE: these agents failed and their sections are unknown: "
        f"{', '.join(unavailable)}."
        if unavailable else ""
    )
    if skipped:
        note += (
            f"\n\nNOTE: these sections were not researched by design, and are "
            f"not evidence gaps: {', '.join(skipped)}."
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
    skipped: list[str],
) -> None:
    """Replace the model's guessed confidences with values derived from evidence.

    Section confidence is the aggregate of that section's scored claims, and
    overall confidence is the mean across sections that actually ran. A model
    asked to rate its own certainty will produce a plausible number; this
    produces a traceable one.

    Skipped sections are excluded from the mean rather than counted as zero.
    Averaging in a section nobody asked for would punish the plan for being
    selective.
    """
    computed = section_confidences(state)
    if not computed:
        return

    available: list[float] = []
    for agent in ("financial", "market", "risk", "sentiment"):
        section = getattr(report, f"{agent}_section", None)
        if section is None:
            continue

        if agent in skipped or agent in unavailable:
            section.available = False
            section.section_confidence = 0.0
            continue

        if agent in computed:
            section.section_confidence = computed[agent]
            available.append(computed[agent])

    if available:
        report.overall_confidence = round(sum(available) / len(available), 4)


def _finalize(report: DueDiligenceReport, state: DueDiligenceState) -> FinalReport:
    """Attach the thesis the model was never asked to author."""
    return FinalReport(**report.model_dump(), thesis=state.get("thesis"))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def synthesizer_node(state: DueDiligenceState) -> dict:
    """LangGraph node: synthesize all findings into a final DueDiligenceReport."""
    company = state["company"]
    today = datetime.now().strftime("%Y-%m-%d")
    unavailable, skipped = agent_status(state)

    if len(unavailable) + len(skipped) == len(SECTIONS):
        logger.error("No section produced findings for %r — reporting inconclusive", company)
        report = inconclusive_report(
            company, today, "every research agent failed to produce findings"
        )
        report.skipped_sections = skipped
        report.unavailable_sections = unavailable
        return {"final_report": _finalize(report, state).model_dump()}

    try:
        structured_llm = get_llm(Tier.SYNTHESIS).with_structured_output(DueDiligenceReport)
        result: DueDiligenceReport = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state)},
        ])

        # Trust our own bookkeeping over the model's self-report.
        result.unavailable_sections = unavailable
        result.skipped_sections = skipped
        _apply_computed_confidence(result, state, unavailable, skipped)
        report = result

    except Exception as exc:
        logger.exception("Synthesizer failed for %r", company)
        report = inconclusive_report(company, today, f"report generation failed ({exc})")

    return {"final_report": _finalize(report, state).model_dump()}
