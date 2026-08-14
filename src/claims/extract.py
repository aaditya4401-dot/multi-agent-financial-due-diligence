"""Turning agent output into typed claims.

The split of labour here is deliberate:

* the **LLM** does semantic work — which metric is this, what period, what did
  the source literally say the number was;
* **Python** does the arithmetic — parsing "$1.9T" into 1.9e12, normalising
  periods, and assigning source tiers from URLs.

Models are unreliable at unit conversion and will happily emit a plausible
wrong number, so the model is never asked to compute anything. It reports the
value *as written*; :func:`parse_value` decides what that means.

Extraction runs on the FAST tier: it is high-volume, mechanical, and needs no
deep reasoning.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.claims.models import Claim, tier_for_source
from src.claims.ontology import (
    PREDICATE_UNITS,
    Predicate,
    Unit,
    is_plausible,
    normalize_period,
    parse_value,
    resolve_predicate,
)
from src.llm import Tier, get_llm

if TYPE_CHECKING:
    # Deferred: src.state imports Claim from this package, so a runtime import
    # here would be circular. These are only ever used as annotations.
    from src.state import AgentFindings, Finding

logger = logging.getLogger(__name__)

_KNOWN_METRICS = ", ".join(sorted(p.value for p in Predicate))

SYSTEM_PROMPT = (
    "You extract structured claims from due-diligence research notes.\n\n"
    "For each distinct assertion, emit one claim:\n"
    "- `assertion`: the claim in one clear sentence.\n"
    "- `metric`: if the claim states a specific number about the company, name "
    f"the metric. Prefer one of these known metrics: {_KNOWN_METRICS}. "
    "Use the source's own wording if none fit. Leave empty for claims that are "
    "not about a specific number.\n"
    "- `raw_value`: the number EXACTLY as the source wrote it, including any "
    "currency symbol and magnitude word — e.g. '$1.9T', '34%', '3.4/5'. "
    "Never convert, round, or compute. Leave empty if there is no number.\n"
    "- `period`: the time period the number refers to, as written "
    "(e.g. 'FY2025', 'Q3 2025', '2024'). Leave empty if unstated.\n"
    "- `source_url`: the URL this came from, if given.\n\n"
    "Extract only what the notes support. Do not infer, combine, or estimate "
    "numbers. A qualitative observation with no number is still a valid claim."
)


class ExtractedClaim(BaseModel):
    assertion: str
    metric: str = ""
    raw_value: str = ""
    period: str = ""
    source_url: str = ""


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def to_claim(
    extracted: ExtractedClaim,
    subject: str,
    agent_name: str,
    observed_at: date,
    legacy_quality: str | None = None,
) -> Claim:
    """Convert one extracted record into a Claim, doing all arithmetic here."""
    predicate: Predicate | None = None
    unit = Unit.NONE
    value: float | None = None

    if extracted.metric:
        predicate = resolve_predicate(extracted.metric)

    if predicate is not None:
        unit = PREDICATE_UNITS.get(predicate, Unit.NONE)
        value = parse_value(extracted.raw_value, unit)

        if value is None:
            # A named metric with an unparseable number is not a usable
            # quantitative claim — keep the text, drop the false precision.
            predicate, unit, value = None, Unit.NONE, None
        elif not is_plausible(predicate, value):
            # Extraction sometimes grabs the wrong number from a sentence — a
            # share price where a valuation belongs. Keep the assertion, but
            # never let an implausible magnitude into the numeric graph, where
            # it would manufacture contradictions and skew confidence.
            logger.warning(
                "Implausible %s value %r for %s — demoting to qualitative",
                predicate.value, value, subject,
            )
            predicate, unit, value = None, Unit.NONE, None

    return Claim(
        subject=subject,
        assertion=extracted.assertion.strip(),
        source_url=extracted.source_url.strip(),
        source_tier=tier_for_source(extracted.source_url, legacy_quality),
        observed_at=observed_at,
        extracted_by=agent_name,
        predicate=predicate,
        period=normalize_period(extracted.period) if predicate else "UNKNOWN",
        value=value,
        unit=unit,
    )


def finding_to_claim(finding: "Finding", subject: str, agent_name: str) -> Claim:
    """Fallback conversion with no LLM — preserves a finding as a qualitative claim.

    Used when extraction is unavailable or fails, so a degraded run loses
    structure but never loses evidence.
    """
    try:
        observed_at = date.fromisoformat(finding.get("date_of_data", ""))
    except (ValueError, TypeError):
        observed_at = date.today()

    return Claim(
        subject=subject,
        assertion=finding.get("claim", "")[:2000],
        source_url=finding.get("source", ""),
        source_tier=tier_for_source(
            finding.get("source", ""), finding.get("source_quality")
        ),
        observed_at=observed_at,
        extracted_by=agent_name,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _build_prompt(subject: str, findings: list["Finding"], summary: str) -> str:
    lines = [f"Company under analysis: {subject}", ""]
    if summary:
        lines += ["Analyst summary:", summary[:2000], ""]

    lines.append("Research notes:")
    for i, f in enumerate(findings, 1):
        lines.append(
            f"{i}. [{f.get('date_of_data', 'undated')}] "
            f"{f.get('claim', '')[:600]}\n   source: {f.get('source', 'unknown')}"
        )
    return "\n".join(lines)


async def extract_claims(agent_findings: "AgentFindings", subject: str) -> list[Claim]:
    """Extract typed claims from one agent's findings.

    Falls back to qualitative claims rather than raising, so a failed
    extraction degrades the structure without losing the evidence.
    """
    if not agent_findings or not agent_findings.get("ok", True):
        return []

    findings = agent_findings.get("findings", [])
    summary = agent_findings.get("summary", "")
    agent_name = agent_findings.get("agent_name", "unknown")

    if not findings and not summary:
        return []

    try:
        structured = get_llm(Tier.FAST).with_structured_output(ExtractionResult)
        result: ExtractionResult = await structured.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(subject, findings, summary)},
        ])
    except Exception:
        logger.exception(
            "Claim extraction failed for %s — falling back to qualitative claims",
            agent_name,
        )
        return [finding_to_claim(f, subject, agent_name) for f in findings]

    today = date.today()
    claims = [
        to_claim(item, subject, agent_name, today)
        for item in result.claims
        if item.assertion.strip()
    ]

    quantitative = sum(1 for c in claims if c.is_quantitative)
    logger.info(
        "%s agent: extracted %d claim(s), %d quantitative",
        agent_name, len(claims), quantitative,
    )
    return claims
