"""Does the report only assert what the system actually found?

Deterministic checks over a finished report and the claim graph behind it. No
model runs here, which is the point: an evaluation that itself hallucinates is
worth nothing, and "is this figure in our evidence?" is a lookup, not a
judgement.

The load-bearing check is :func:`unsupported_figures`. A due diligence memo is
mostly numbers, and a number in the prose that matches no claim is the single
most damaging thing this system could emit — it reads as researched fact and is
not. Finding those is a parse and a comparison, both of which already exist for
the claim graph.

**Only figures carrying a magnitude marker are checked** — a currency symbol, a
scale word, or a percent sign. Bare integers are too ambiguous to test: years,
counts, list positions and ordinals would all trip a naive check and bury the
real findings in noise. A tighter rule with no false positives is worth more
than a broad one nobody trusts.
"""

import logging
import re

from src.claims.models import Claim
from src.claims.ontology import DEFAULT_TOLERANCE, Unit, parse_value
from src.eval.models import CitationAudit, EvaluationReport, GroundednessCheck

logger = logging.getLogger(__name__)

#: Figures worth checking: `$1.4T`, `1.9 billion`, `34%`. A bare `2024` is not.
_FIGURE_RE = re.compile(
    r"""(
        \$\s?\d[\d,]*(?:\.\d+)?\s*
            (?:k|mm|m|bn|b|tn|t|thousand|million|billion|trillion)?s?\b
      | \d[\d,]*(?:\.\d+)?\s*
            (?:k|mm|bn|tn|thousand|million|billion|trillion)s?\b
      | \d[\d,]*(?:\.\d+)?\s*%
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: Percent figures only compare against percent claims, and magnitudes only
#: against magnitude claims — otherwise "34%" would match a claim of 34 units.
_MAGNITUDE_UNITS = {Unit.USD, Unit.COUNT, Unit.MONTHS}


def extract_figures(text: str) -> list[str]:
    """Every magnitude-bearing figure asserted in *text*."""
    if not text:
        return []
    return [m.group(0).strip() for m in _FIGURE_RE.finditer(text)]


def _claim_values(claims: list[Claim], percent: bool) -> list[float]:
    wanted = {Unit.PERCENT, Unit.RATING} if percent else _MAGNITUDE_UNITS
    return [
        c.value for c in claims
        if c.value is not None and c.unit in wanted
    ]


def is_supported(figure: str, claims: list[Claim]) -> bool:
    """Does any gathered claim carry this number, within tolerance?"""
    percent = "%" in figure
    value = parse_value(figure, Unit.PERCENT if percent else Unit.USD)
    if value is None:
        return True  # unparseable: not evidence of a problem

    for candidate in _claim_values(claims, percent):
        scale = max(abs(candidate), abs(value), 1e-9)
        if abs(candidate - value) / scale <= DEFAULT_TOLERANCE:
            return True
    return False


def report_prose(report: dict) -> str:
    """Every piece of the report a human would read as an assertion of fact."""
    parts: list[str] = [report.get("executive_summary") or ""]

    for key in ("financial_section", "market_section", "risk_section",
                "sentiment_section"):
        section = report.get(key) or {}
        for finding in section.get("findings") or []:
            parts.append(finding.get("claim") or "")

    thesis = report.get("thesis") or {}
    parts.append(thesis.get("summary") or "")
    parts.append(thesis.get("recommendation_rationale") or "")
    for driver in thesis.get("drivers") or []:
        parts.append(driver.get("statement") or "")
        parts.append(driver.get("what_would_falsify_this") or "")
    for case_key in ("bull_case", "base_case", "bear_case"):
        case = thesis.get(case_key) or {}
        parts.append(case.get("narrative") or "")
        parts.extend(case.get("key_assumptions") or [])
    for flag in thesis.get("red_flags") or []:
        parts.append(flag.get("issue") or "")

    return "\n".join(p for p in parts if p)


def unsupported_figures(report: dict, claims: list[Claim]) -> list[str]:
    """Figures asserted in the report that match no gathered claim."""
    seen: set[str] = set()
    unsupported: list[str] = []
    for figure in extract_figures(report_prose(report)):
        key = figure.lower().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        if not is_supported(figure, claims):
            unsupported.append(figure)
    return unsupported


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_claim_sourcing(claims: list[Claim]) -> GroundednessCheck:
    if not claims:
        return GroundednessCheck(
            name="claims_have_sources", passed=False, score=0.0,
            detail="no claims were gathered")
    sourced = sum(1 for c in claims if c.source_url.strip())
    rate = sourced / len(claims)
    return GroundednessCheck(
        name="claims_have_sources", passed=rate >= 0.95, score=round(rate, 4),
        detail=f"{sourced}/{len(claims)} claims carry a source URL")


def _check_figures(report: dict, claims: list[Claim]) -> tuple[GroundednessCheck, list[str]]:
    figures = extract_figures(report_prose(report))
    if not figures:
        return GroundednessCheck(
            name="figures_traceable", passed=True, score=1.0,
            detail="report asserts no magnitude figures"), []

    unsupported = unsupported_figures(report, claims)
    rate = 1.0 - (len(unsupported) / len(set(f.lower() for f in figures)))
    return GroundednessCheck(
        name="figures_traceable", passed=not unsupported,
        score=round(max(rate, 0.0), 4),
        detail=(f"{len(unsupported)} unsupported of "
                f"{len(set(f.lower() for f in figures))} distinct figure(s)"
                + (f": {', '.join(unsupported[:5])}" if unsupported else ""))
    ), unsupported


def _check_thesis(audit: CitationAudit | None) -> GroundednessCheck:
    if audit is None or audit.drivers_proposed == 0:
        return GroundednessCheck(
            name="thesis_drivers_grounded", passed=False, score=0.0,
            detail="no thesis drivers were proposed")
    rate = audit.grounding_rate
    return GroundednessCheck(
        name="thesis_drivers_grounded", passed=rate >= 0.99, score=rate,
        detail=(f"{audit.drivers_kept}/{audit.drivers_proposed} drivers survived "
                f"citation checking; {audit.invented_citations} invented id(s)"))


def _check_sections(report: dict, claims: list[Claim]) -> GroundednessCheck:
    by_agent = {c.extracted_by for c in claims}
    available, backed = 0, 0
    for agent in ("financial", "market", "risk", "sentiment"):
        section = report.get(f"{agent}_section")
        # A section absent from the report is not an unbacked section — there
        # is nothing there to be backed. Only sections the report actually
        # presents as available are held to needing evidence.
        if not section:
            continue
        if not section.get("available", True):
            continue
        if agent in report.get("skipped_sections", []):
            continue
        available += 1
        if agent in by_agent:
            backed += 1

    if available == 0:
        return GroundednessCheck(
            name="sections_backed_by_claims", passed=False, score=0.0,
            detail="no section was available")
    rate = backed / available
    return GroundednessCheck(
        name="sections_backed_by_claims", passed=rate >= 0.99, score=round(rate, 4),
        detail=f"{backed}/{available} available sections rest on extracted claims")


def evaluate(
    report: dict,
    claims: list[Claim],
    audit: CitationAudit | None = None,
) -> EvaluationReport:
    """Score a finished report against the evidence behind it."""
    figures_check, unsupported = _check_figures(report, claims)
    checks = [
        _check_claim_sourcing(claims),
        figures_check,
        _check_thesis(audit),
        _check_sections(report, claims),
    ]
    score = round(sum(c.score for c in checks) / len(checks), 4)

    result = EvaluationReport(
        score=score, checks=checks,
        unsupported_figures=unsupported, citation_audit=audit,
    )
    if result.failed_checks:
        logger.warning(
            "Groundedness %s: %s", result.describe(),
            "; ".join(f"{c.name} ({c.detail})" for c in result.failed_checks),
        )
    return result
