"""Finding gaps in the evidence — deterministically, from the claim graph.

No LLM runs here. Everything this module reports is arithmetic over claims that
already exist: which core metrics have no claim, which sections rest on weak
sources, which numbers are contradicted without a clear winner, which are old.

That matters for the refinement loop's credibility. If a model decided when the
evidence was thin, "should we spend another round of research" would itself be
a hallucination risk. Here it is a groupby and a threshold.

**Value of information.** Each gap is scored by how much resolving it would move
the verdict::

    voi = kind_weight x importance x residual_uncertainty

so a contradicted revenue figure outranks a missing Glassdoor rating, and a
metric we already know confidently scores near zero even if it is technically
improvable. Prioritising this way — rather than "is anything unresolved" —
is what stops round two spending real money on a detail nobody's decision
turns on.
"""

import logging
from datetime import date

from src.claims.confidence import recency_factor
from src.claims.contradictions import detect
from src.claims.models import Claim
from src.claims.ontology import Predicate
from src.gaps.models import Gap, GapKind
from src.planning.models import CompanyType

logger = logging.getLogger(__name__)

#: How much each kind of gap should worry us, before the specifics.
#:
#: A contradiction outranks a hole deliberately: a hole means the report omits
#: something, but a contradiction means the report will *state* one of two
#: incompatible numbers. Being confidently wrong is worse than being silent.
KIND_WEIGHT: dict[GapKind, float] = {
    GapKind.UNRESOLVED_CONTRADICTION: 1.00,
    GapKind.MISSING_METRIC: 0.90,
    GapKind.LOW_CONFIDENCE: 0.60,
    GapKind.STALE_EVIDENCE: 0.40,
}

#: The metrics a verdict actually leans on, and how heavily, per company type.
#: A private company has no market cap to find, so its absence is not a gap.
CORE_PREDICATES: dict[CompanyType, dict[Predicate, float]] = {
    CompanyType.PUBLIC: {
        Predicate.REVENUE: 1.00,
        Predicate.REVENUE_GROWTH_YOY: 0.90,
        Predicate.GROSS_MARGIN: 0.80,
        Predicate.OPERATING_MARGIN: 0.65,
        Predicate.NET_INCOME: 0.60,
        Predicate.MARKET_CAP: 0.45,
        Predicate.MARKET_SHARE: 0.40,
    },
    CompanyType.PRIVATE: {
        Predicate.REVENUE: 1.00,
        Predicate.REVENUE_GROWTH_YOY: 0.90,
        Predicate.VALUATION: 0.80,
        Predicate.FUNDING_TOTAL: 0.65,
        Predicate.BURN_RATE: 0.55,
        Predicate.MARKET_SHARE: 0.40,
    },
    CompanyType.UNKNOWN: {
        Predicate.REVENUE: 1.00,
        Predicate.REVENUE_GROWTH_YOY: 0.90,
        Predicate.MARKET_SHARE: 0.40,
    },
}

#: Which agent is equipped to chase which metric.
_MARKET_PREDICATES = {
    Predicate.TAM, Predicate.SAM, Predicate.SOM,
    Predicate.MARKET_SHARE, Predicate.CUSTOMER_COUNT,
}
_SENTIMENT_PREDICATES = {Predicate.GLASSDOOR_RATING}

#: A section below this is resting on weak sources.
LOW_CONFIDENCE_THRESHOLD = 0.45

#: A claim whose recency factor has decayed below this is worth refreshing.
STALE_THRESHOLD = 0.50

#: Contradictions where the winner leads by less than this are not settled.
THIN_MARGIN = 0.15

RESEARCH_AGENTS = ("financial", "market", "risk", "sentiment")


def owner_for(predicate: Predicate) -> str:
    """Which agent should chase *predicate*."""
    if predicate in _MARKET_PREDICATES:
        return "market"
    if predicate in _SENTIMENT_PREDICATES:
        return "sentiment"
    return "financial"


def _by_agent(claims: list[Claim]) -> dict[str, list[Claim]]:
    grouped: dict[str, list[Claim]] = {}
    for claim in claims:
        grouped.setdefault(claim.extracted_by, []).append(claim)
    return grouped


def _missing_metrics(
    claims: list[Claim], subject: str, company_type: CompanyType
) -> list[Gap]:
    """Core metrics with no quantitative claim at all."""
    present = {c.predicate for c in claims if c.is_quantitative}
    gaps = []

    for predicate, importance in CORE_PREDICATES[company_type].items():
        if predicate in present:
            continue
        # Nothing known at all, so residual uncertainty is total.
        gaps.append(Gap(
            kind=GapKind.MISSING_METRIC,
            agent=owner_for(predicate),
            subject=subject,
            predicate=predicate.value,
            detail=f"no {predicate.value} figure was found for {subject}",
            question=(
                f"Find {subject}'s {predicate.value.replace('_', ' ')} for the most "
                f"recent reported period. Report the figure exactly as the source "
                f"states it, the period it covers, and the source URL. If no "
                f"reliable figure exists, say so explicitly rather than estimating."
            ),
            value_of_information=round(
                KIND_WEIGHT[GapKind.MISSING_METRIC] * importance, 4),
        ))
    return gaps


def _unresolved_contradictions(claims: list[Claim], subject: str) -> list[Gap]:
    """Numeric disagreements where no source clearly wins.

    A contradiction with a decisive winner is not a gap — the scorer already
    demoted the loser. What needs another round is the close call.
    """
    gaps = []
    for contradiction in detect(claims).contradictions:
        winner = contradiction.winner
        losers = contradiction.losers()
        if winner is None or not losers:
            continue

        best_loser = max(losers, key=lambda c: c.confidence)
        margin = winner.confidence - best_loser.confidence
        if margin >= THIN_MARGIN:
            continue  # settled

        predicate = winner.predicate.value if winner.predicate else None
        importance = 0.8
        if winner.predicate:
            for table in CORE_PREDICATES.values():
                importance = max(importance, table.get(winner.predicate, 0.0))

        gaps.append(Gap(
            kind=GapKind.UNRESOLVED_CONTRADICTION,
            agent=winner.extracted_by or "financial",
            subject=subject,
            predicate=predicate,
            detail=(
                f"sources disagree on {predicate or 'a figure'}: "
                f"{winner.value} vs {best_loser.value} "
                f"(margin {margin:.2f}, unsettled)"
            ),
            question=(
                f"Sources disagree on {subject}'s "
                f"{(predicate or 'figure').replace('_', ' ')} for "
                f"{winner.period}: one reports {winner.value:,.4g}, another "
                f"{best_loser.value:,.4g}. Find a primary source — a filing, an "
                f"earnings release, or the company's own disclosure — that "
                f"settles which is correct. Quote the figure and give the URL."
            ),
            # Residual uncertainty is highest when the two sides are level.
            value_of_information=round(
                KIND_WEIGHT[GapKind.UNRESOLVED_CONTRADICTION]
                * importance
                * (1.0 - margin / THIN_MARGIN), 4),
        ))
    return gaps


def _low_confidence_sections(
    claims: list[Claim], subject: str, planned: tuple[str, ...]
) -> list[Gap]:
    """Sections that produced claims, but only weak ones."""
    from src.claims.confidence import aggregate_confidence

    gaps = []
    grouped = _by_agent(claims)
    for agent in planned:
        section = grouped.get(agent, [])
        if not section:
            continue  # absent entirely — that is a failure, not a weak section
        confidence = aggregate_confidence(section)
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            continue
        gaps.append(Gap(
            kind=GapKind.LOW_CONFIDENCE,
            agent=agent,
            subject=subject,
            detail=f"{agent} section rests on weak sources (confidence {confidence:.0%})",
            question=(
                f"The {agent} findings for {subject} currently rest on "
                f"low-quality sources. Find corroboration from stronger ones — "
                f"regulatory filings, earnings materials, or tier-1 press — for "
                f"the main claims, and give URLs."
            ),
            value_of_information=round(
                KIND_WEIGHT[GapKind.LOW_CONFIDENCE] * (1.0 - confidence), 4),
        ))
    return gaps


def _stale_evidence(
    claims: list[Claim], subject: str, company_type: CompanyType,
    today: date | None = None,
) -> list[Gap]:
    """Core metrics whose best claim has decayed past usefulness."""
    gaps = []
    core = CORE_PREDICATES[company_type]

    best: dict[Predicate, Claim] = {}
    for claim in claims:
        if not claim.is_quantitative or claim.predicate not in core:
            continue
        current = best.get(claim.predicate)
        if current is None or claim.confidence > current.confidence:
            best[claim.predicate] = claim

    for predicate, claim in best.items():
        freshness = recency_factor(claim, today)
        if freshness >= STALE_THRESHOLD:
            continue
        gaps.append(Gap(
            kind=GapKind.STALE_EVIDENCE,
            agent=owner_for(predicate),
            subject=subject,
            predicate=predicate.value,
            detail=(
                f"best {predicate.value} figure is from {claim.observed_at} "
                f"(freshness {freshness:.2f})"
            ),
            question=(
                f"The most recent {predicate.value.replace('_', ' ')} figure "
                f"found for {subject} dates from {claim.observed_at}. Find a "
                f"more recent figure and give its period and source URL."
            ),
            value_of_information=round(
                KIND_WEIGHT[GapKind.STALE_EVIDENCE]
                * core[predicate] * (1.0 - freshness), 4),
        ))
    return gaps


def find_gaps(
    claims: list[Claim],
    subject: str,
    company_type: CompanyType = CompanyType.UNKNOWN,
    planned: tuple[str, ...] = RESEARCH_AGENTS,
    today: date | None = None,
) -> list[Gap]:
    """Every gap in the current evidence, best value of information first."""
    gaps = [
        *_unresolved_contradictions(claims, subject),
        *_missing_metrics(claims, subject, company_type),
        *_low_confidence_sections(claims, subject, planned),
        *_stale_evidence(claims, subject, company_type, today),
    ]
    gaps.sort(key=lambda g: g.value_of_information, reverse=True)
    logger.debug("Found %d gap(s) for %r", len(gaps), subject)
    return gaps
