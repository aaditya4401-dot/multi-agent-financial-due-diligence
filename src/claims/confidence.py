"""Confidence propagation.

Confidence used to be a constant chosen at the call site — 0.85 for anything
yfinance returned, 0.5 for anything a web search returned — and the final
report confidence was invented by the LLM. Neither reacted to evidence.

Here a claim's confidence is derived from four things it can actually be held
accountable to::

    confidence = prior x recency x corroboration_boost x contradiction_penalty

* **prior** — how much the source tier is worth (a filing beats a forum).
* **recency** — exponential decay on the predicate's half-life, floored so an
  old 10-K is worth less than a new one but never worthless.
* **corroboration** — noisy-OR over *independent domains*, discounted and
  capped so a pile of syndicated blogspam cannot outweigh a primary source.
* **contradiction** — when claims conflict, they split confidence in
  proportion to their standing, so the loser is visibly demoted.
"""

import logging
from datetime import date

from src.claims.contradictions import DetectionResult
from src.claims.models import Claim
from src.claims.ontology import DEFAULT_HALF_LIFE_DAYS, half_life_days

logger = logging.getLogger(__name__)

#: Confidence retained by an infinitely old claim. Age discounts evidence;
#: it does not delete it.
RECENCY_FLOOR = 0.30

#: Corroborating sources are rarely fully independent, so their contribution
#: is discounted before being combined.
INDEPENDENCE_DISCOUNT = 0.60

#: Beyond this many corroborators the marginal evidence is mostly syndication.
MAX_CORROBORATORS = 4


def recency_factor(claim: Claim, today: date | None = None) -> float:
    """Exponential decay on the predicate's half-life, floored at RECENCY_FLOOR."""
    today = today or date.today()
    age_days = max((today - claim.observed_at).days, 0)
    half_life = (
        half_life_days(claim.predicate) if claim.predicate else DEFAULT_HALF_LIFE_DAYS
    )
    decay = 0.5 ** (age_days / half_life)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decay


def base_score(claim: Claim, today: date | None = None) -> float:
    """Source quality tempered by age, before any cross-claim evidence."""
    return claim.prior() * recency_factor(claim, today)


def score_claims(
    claims: list[Claim],
    detection: DetectionResult | None = None,
    today: date | None = None,
) -> list[Claim]:
    """Assign confidence to every claim, in place, and return them.

    Corroboration uses each supporter's *base* score rather than its final
    confidence, which keeps this a single deterministic pass with no circular
    dependency between mutually-supporting claims.
    """
    today = today or date.today()
    by_id = {c.id: c for c in claims}
    bases = {c.id: base_score(c, today) for c in claims}

    corroborations = detection.corroborations if detection else {}
    contradictions = detection.contradictions if detection else []

    # Which claims lost an adjudication, and to whom.
    demoted: dict[str, float] = {}
    for contradiction in contradictions:
        winner = contradiction.winner
        if winner is None:
            continue
        winner_base = bases.get(winner.id, 0.0)
        for loser in contradiction.losers():
            demoted[loser.id] = max(demoted.get(loser.id, 0.0), winner_base)
            loser.contradicted_by = sorted(
                set(loser.contradicted_by) | {winner.id}
            )
        winner.contradicted_by = sorted(
            set(winner.contradicted_by) | {c.id for c in contradiction.losers()}
        )

    for claim in claims:
        score = bases[claim.id]

        # --- corroboration: noisy-OR over independent domains --------------
        supporter_ids = corroborations.get(claim.id, [])
        if supporter_ids:
            claim.corroborated_by = sorted(set(supporter_ids))
            supporters = sorted(
                (by_id[i] for i in set(supporter_ids) if i in by_id),
                key=lambda c: bases[c.id],
                reverse=True,
            )[:MAX_CORROBORATORS]
            for supporter in supporters:
                contribution = bases[supporter.id] * INDEPENDENCE_DISCOUNT
                score = 1.0 - (1.0 - score) * (1.0 - contribution)

        # --- contradiction: split standing with the winner ------------------
        opposing = demoted.get(claim.id)
        if opposing is not None:
            total = score + opposing
            score *= (score / total) if total > 0 else 0.0

        claim.confidence = round(min(max(score, 0.0), 1.0), 4)

    logger.debug("Scored %d claim(s)", len(claims))
    return claims


def aggregate_confidence(claims: list[Claim]) -> float:
    """Section-level confidence: the evidence-weighted mean of its claims.

    Weighting by confidence itself means a section carried by one strong claim
    scores better than one padded with weak ones.
    """
    scored = [c for c in claims if c.confidence > 0]
    if not scored:
        return 0.0
    weights = [c.confidence for c in scored]
    return round(sum(w * w for w in weights) / sum(weights), 4)
