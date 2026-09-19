"""Forming an investment view, and refusing to let it float free of evidence.

Runs after the evidence is settled and before the memo is written, so the
executive summary can reflect a view rather than the other way round.

The interesting work is not the prompt — it is what happens to the answer.
A model asked for a thesis will produce a fluent one whether or not the
research supports it, and it will happily cite claim ids that do not exist.
So every citation is checked against the actual claim graph, and a driver whose
citations all turn out to be invented is dropped. If that leaves nothing, the
recommendation is forced to "Insufficient evidence".

This check is only possible because claims are typed and have stable ids. It is
the concrete payoff for the claim graph: groundedness becomes a set membership
test rather than a matter of opinion.
"""

import logging

from src.claims.models import Claim
from src.eval.models import CitationAudit
from src.gaps.models import Gap
from src.llm import Tier, get_llm
from src.models.thesis import InvestmentThesis, insufficient_evidence_thesis
from src.state import Conflict, DueDiligenceState

logger = logging.getLogger(__name__)

#: Claims shown to the thesis prompt, best evidence first.
MAX_CLAIMS = 45

#: Open questions carried into the memo.
MAX_OPEN_QUESTIONS = 8

SYSTEM_PROMPT = (
    "You are an investment analyst forming a due diligence thesis.\n\n"
    "You are given sourced claims, each with an id in square brackets, plus "
    "any detected conflicts and unresolved gaps.\n\n"
    "Produce:\n"
    "- `summary`: the thesis in one sentence.\n"
    "- `drivers`: 3-5 propositions the case actually rests on. Each must be "
    "specific and checkable — 'revenue is growing' is an observation, not a "
    "driver. Each must cite `evidence_claim_ids` using ONLY ids shown to you, "
    "and state `what_would_falsify_this`.\n"
    "- `bull_case`, `base_case`, `bear_case`: how this plausibly goes, with "
    "the assumptions each depends on.\n"
    "- `red_flags`: graded `deal_breaker` (ends the conversation), `material` "
    "(changes terms), or `monitor`. Cite ids where you have them.\n"
    "- `recommendation` and `recommendation_rationale`: the rationale must "
    "trace from drivers to verdict.\n\n"
    "Rules:\n"
    "- NEVER invent a claim id. Cite only ids listed below. An uncited driver "
    "is rejected outright.\n"
    "- Do not treat missing evidence as bad news. If the evidence is too thin "
    "to support a view, say so and recommend 'Insufficient evidence'.\n"
    "- Where a section is marked as estimates rather than reported figures, "
    "do not write about it as though it were audited."
)


def _format_claims(claims: list[Claim]) -> str:
    """Claim digest with ids, which is what makes citation possible at all."""
    ranked = sorted(claims, key=lambda c: c.confidence, reverse=True)[:MAX_CLAIMS]
    lines = []
    for claim in ranked:
        lines.append(f"  [{claim.id}] {claim.describe()}")
        lines.append(f"        {claim.assertion[:220]}")
    if len(claims) > MAX_CLAIMS:
        lines.append(f"  … and {len(claims) - MAX_CLAIMS} weaker claim(s), omitted")
    return "\n".join(lines)


def _format_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "  (none detected)"
    return "\n".join(
        f"  - [{c['type']}] {c['agent_a']} vs {c['agent_b']}: "
        f"{c['claim_a'][:120]} / {c['claim_b'][:120]} — {c['resolution'][:140]}"
        for c in conflicts[:15]
    )


def _format_gaps(gaps: list[Gap]) -> str:
    if not gaps:
        return "  (none — the evidence covered what the verdict needs)"
    return "\n".join(f"  - {g.detail}" for g in gaps[:MAX_OPEN_QUESTIONS])


def open_questions_from_gaps(gaps: list[Gap]) -> list[str]:
    """The memo's 'further diligence required', taken from what actually failed.

    Generated rather than invented. The gap analyzer already knows precisely
    what it could not establish, so asking a model to guess would be strictly
    worse information.
    """
    return [
        g.detail[0].upper() + g.detail[1:] if g.detail else g.kind.value
        for g in sorted(gaps, key=lambda g: g.value_of_information, reverse=True)[
            :MAX_OPEN_QUESTIONS
        ]
    ]


def audit_citations(thesis: InvestmentThesis, claims: list[Claim]) -> CitationAudit:
    """Count what the model proposed, before validation throws any of it away.

    Must run *before* :func:`validate_citations`: once dropped drivers are gone
    they are unrecoverable, and "how much of this thesis was invented" is
    exactly what the evaluation layer needs to report.
    """
    known = {claim.id for claim in claims}
    invented = sum(
        1
        for driver in thesis.drivers
        for cid in driver.evidence_claim_ids
        if cid not in known
    )
    kept = sum(
        1 for driver in thesis.drivers
        if any(cid in known for cid in driver.evidence_claim_ids)
    )
    return CitationAudit(
        drivers_proposed=len(thesis.drivers),
        drivers_kept=kept,
        invented_citations=invented,
    )


def validate_citations(
    thesis: InvestmentThesis, claims: list[Claim]
) -> tuple[InvestmentThesis, list[str]]:
    """Strip invented citations; drop drivers left with none.

    Returns the cleaned thesis and a list of human-readable problems found.
    A driver citing only ids that do not exist was not derived from the
    evidence, whatever it says about itself.
    """
    known = {claim.id for claim in claims}
    problems: list[str] = []

    kept = []
    for driver in thesis.drivers:
        valid = [cid for cid in driver.evidence_claim_ids if cid in known]
        invented = set(driver.evidence_claim_ids) - known

        if invented:
            problems.append(
                f"driver {driver.statement[:60]!r} cited "
                f"{len(invented)} unknown claim id(s)"
            )
        if not valid:
            problems.append(
                f"dropped driver {driver.statement[:60]!r}: no valid citations"
            )
            continue
        kept.append(driver.model_copy(update={"evidence_claim_ids": valid}))

    flags = [
        flag.model_copy(update={
            "evidence_claim_ids": [c for c in flag.evidence_claim_ids if c in known]
        })
        for flag in thesis.red_flags
    ]

    cleaned = thesis.model_copy(update={"drivers": kept, "red_flags": flags})

    if not kept:
        problems.append("no driver survived citation checking")
        cleaned = cleaned.model_copy(update={
            "recommendation": "Insufficient evidence",
            "recommendation_rationale": (
                "No thesis driver could be traced to a sourced claim, so no "
                "recommendation is supportable."
            ),
        })

    return cleaned, problems


def _build_user_prompt(state: DueDiligenceState, claims: list[Claim]) -> str:
    company = state["company"]
    plan = state.get("plan")
    context = ""
    if plan is not None:
        context = f"\n{company} is classified as {plan.company_type.value}. {plan.rationale}\n"

    return (
        f"Form an investment thesis for {company}.\n{context}\n"
        f"=== SOURCED CLAIMS (cite these ids, and only these) ===\n"
        f"{_format_claims(claims)}\n\n"
        f"=== CONFLICTS ===\n{_format_conflicts(state.get('conflicts') or [])}\n\n"
        f"=== UNRESOLVED GAPS ===\n{_format_gaps(state.get('gaps') or [])}\n"
    )


async def thesis_node(state: DueDiligenceState) -> dict:
    """LangGraph node: form the investment thesis, grounded in the claim graph."""
    company = state["company"]
    claims = state.get("claims") or []
    gaps = state.get("gaps") or []

    if not claims:
        logger.warning("No claims for %r — no thesis can be formed", company)
        thesis = insufficient_evidence_thesis(
            "no sourced claims were gathered for this company"
        )
        thesis.open_questions = open_questions_from_gaps(gaps)
        return {"thesis": thesis, "thesis_audit": CitationAudit()}

    try:
        model = get_llm(Tier.SYNTHESIS).with_structured_output(InvestmentThesis)
        result: InvestmentThesis = await model.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state, claims)},
        ])

        # Audited before validation, which is destructive by design.
        audit = audit_citations(result, claims)
        thesis, problems = validate_citations(result, claims)
        for problem in problems:
            logger.warning("Thesis grounding: %s", problem)

        # Generated, not invented — see open_questions_from_gaps.
        thesis.open_questions = open_questions_from_gaps(gaps)

        logger.info(
            "Thesis for %s: %s — %d driver(s), %d red flag(s), %d deal-breaker(s)",
            company, thesis.recommendation, len(thesis.drivers),
            len(thesis.red_flags), len(thesis.deal_breakers),
        )

    except Exception as exc:
        logger.exception("Thesis formation failed for %r", company)
        thesis = insufficient_evidence_thesis(f"thesis generation failed ({exc})")
        thesis.open_questions = open_questions_from_gaps(gaps)
        audit = CitationAudit()

    return {"thesis": thesis, "thesis_audit": audit}
