"""Deterministic contradiction detection.

The old resolver handed four blobs of prose to an LLM and asked it to spot that
"$1.9T" in one and "$1.4T" in another referred to the same metric. That is a
parsing job dressed up as a reasoning job, and it failed quietly.

Once claims are typed, the same question is a groupby:

* **Factual contradiction** — same subject, predicate and period, values that
  differ by more than the predicate's tolerance.
* **Stale data** — same subject and predicate across time, where an older
  observation materially disagrees with a newer one.
* **Corroboration** — agreement within tolerance from *independent* domains.

Complementary tension ("growing fast" vs "burning cash") is deliberately absent:
it is a genuinely qualitative judgement and stays with the LLM, which now only
has to weigh a handful of flagged pairs instead of scanning everything.
"""

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from src.claims.models import Claim
from src.claims.ontology import Predicate, half_life_days, tolerance

logger = logging.getLogger(__name__)

ContradictionKind = Literal["factual_contradiction", "stale_data"]

#: Divergence above which undated claims are treated as conflicting rather
#: than as snapshots of different moments.
UNDATED_MIN_SPREAD = 0.50


@dataclass
class Contradiction:
    """Two or more claims that cannot all be right."""

    kind: ContradictionKind
    subject: str
    predicate: Predicate
    period: str
    claims: list[Claim]
    spread: float                       # relative disagreement, 0.0–1.0+
    winner: Claim | None = None         # adjudicated on tier, then recency
    rationale: str = ""
    #: Claims grouped by mutually-agreeing value.
    clusters: list[list[Claim]] = field(default_factory=list)

    def winning_cluster(self) -> list[Claim]:
        """Every claim that agrees with the winner, not just the winner itself."""
        if self.winner is None:
            return []
        for cluster in self.clusters:
            if any(c.id == self.winner.id for c in cluster):
                return cluster
        return [self.winner]

    def losers(self) -> list[Claim]:
        """Claims that disagree with the adjudicated value.

        Membership is by cluster, not identity: a source that corroborates the
        winner has not lost anything and must not be demoted alongside the
        source that actually contradicted it.
        """
        winning_ids = {c.id for c in self.winning_cluster()}
        return [c for c in self.claims if c.id not in winning_ids]


@dataclass
class DetectionResult:
    contradictions: list[Contradiction] = field(default_factory=list)
    #: claim id → ids of independent claims agreeing with it
    corroborations: dict[str, list[str]] = field(default_factory=dict)

    def contradicted_ids(self) -> set[str]:
        return {c.id for con in self.contradictions for c in con.claims}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_spread(values: list[float]) -> float:
    """Disagreement as a fraction of magnitude.

    Uses the largest absolute value as the denominator so the result is
    symmetric and stable when values straddle zero.
    """
    if len(values) < 2:
        return 0.0
    lo, hi = min(values), max(values)
    scale = max(abs(lo), abs(hi))
    if scale == 0:
        return 0.0
    return (hi - lo) / scale


def _cluster_values(claims: list[Claim], tol: float) -> list[list[Claim]]:
    """Split claims into runs of mutually-agreeing values.

    Greedy single-pass clustering over sorted values: a claim joins the current
    cluster if adding it keeps the cluster's spread within tolerance, otherwise
    it starts a new one. Returns one cluster when everything agrees.
    """
    ordered = sorted(claims, key=lambda c: c.value)
    clusters: list[list[Claim]] = [[ordered[0]]]

    for candidate in ordered[1:]:
        current = clusters[-1]
        values = [c.value for c in current] + [candidate.value]
        if _relative_spread(values) <= tol:
            current.append(candidate)
        else:
            clusters.append([candidate])

    return clusters


def _recency_rank(claim: Claim) -> tuple:
    """Sort key for adjudication: source quality first, then recency."""
    return (claim.prior(), claim.observed_at)


def _adjudicate(claims: list[Claim], kind: ContradictionKind) -> tuple[Claim, str]:
    """Pick the claim to believe, and say why.

    Better source tier wins; ties break on recency. For stale-data conflicts
    recency leads instead, because the disagreement *is* about time.
    """
    if kind == "stale_data":
        winner = max(claims, key=lambda c: (c.observed_at, c.prior()))
        reason = (
            f"accepted the more recent figure ({winner.observed_at}) from "
            f"{winner.source_domain or 'unknown source'}"
        )
    else:
        winner = max(claims, key=_recency_rank)
        reason = (
            f"preferred the higher-quality source "
            f"({winner.source_tier.value}, {winner.observed_at})"
        )
    return winner, reason


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(claims: list[Claim]) -> DetectionResult:
    """Find contradictions and corroborations across a set of claims."""
    quantitative = [c for c in claims if c.is_quantitative]
    result = DetectionResult()

    if not quantitative:
        return result

    flagged_groups: set[tuple[str, str, str]] = set()

    # --- Pass A: same period, divergent values -----------------------------
    by_group: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
    for claim in quantitative:
        by_group[claim.group_key].append(claim)

    for group_key, group in by_group.items():
        if len(group) < 2:
            continue

        subject, predicate_name, period = group_key
        predicate = Predicate(predicate_name)
        tol = tolerance(predicate)

        # Split the group into clusters of mutually-agreeing values. Sources
        # inside a cluster corroborate each other even when the group as a
        # whole is in conflict — "eight sources say X, one says Y" is very
        # different evidence from "nine sources all disagree".
        clusters = _cluster_values(group, tol)
        for cluster in clusters:
            _record_corroboration(result, cluster)

        if len(clusters) < 2:
            continue

        spread = _relative_spread([c.value for c in group])

        # Undated claims may simply describe different moments, so ordinary
        # divergence between them is not evidence of a conflict. Extreme
        # divergence is a different matter — that is usually a bad extraction
        # or a genuine disagreement, and staying silent hides both.
        if period == "UNKNOWN" and spread < UNDATED_MIN_SPREAD:
            continue
        winner, reason = _adjudicate(group, "factual_contradiction")
        flagged_groups.add(group_key)
        sizes = "/".join(str(len(c)) for c in clusters)
        result.contradictions.append(Contradiction(
            kind="factual_contradiction",
            subject=subject,
            predicate=predicate,
            period=period,
            claims=group,
            spread=spread,
            winner=winner,
            clusters=clusters,
            rationale=(
                f"{len(group)} sources split {sizes} across {len(clusters)} "
                f"incompatible values for {predicate.value} in {period} "
                f"(spread {spread:.0%}); {reason}"
            ),
        ))

    # --- Pass B: same metric across time, materially divergent -------------
    by_metric: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim in quantitative:
        by_metric[(claim.subject, claim.predicate.value)].append(claim)

    for (subject, predicate_name), group in by_metric.items():
        if len(group) < 2:
            continue

        predicate = Predicate(predicate_name)
        oldest = min(group, key=lambda c: c.observed_at)
        newest = max(group, key=lambda c: c.observed_at)

        age_gap_days = (newest.observed_at - oldest.observed_at).days
        if age_gap_days < half_life_days(predicate):
            continue

        spread = _relative_spread([oldest.value, newest.value])
        if spread <= tolerance(predicate):
            continue

        pair = [oldest, newest]
        # Skip if Pass A already flagged these two together.
        if any(k in flagged_groups for k in {oldest.group_key, newest.group_key} if
               oldest.group_key == newest.group_key):
            continue

        winner, reason = _adjudicate(pair, "stale_data")
        result.contradictions.append(Contradiction(
            kind="stale_data",
            subject=subject,
            predicate=predicate,
            period=f"{oldest.period}→{newest.period}",
            claims=pair,
            spread=spread,
            winner=winner,
            clusters=[[oldest], [newest]],
            rationale=(
                f"{predicate.value} differs {spread:.0%} across a {age_gap_days}-day "
                f"gap, longer than its {half_life_days(predicate):.0f}-day half-life; {reason}"
            ),
        ))

    logger.info(
        "Detected %d contradiction(s) across %d quantitative claim(s)",
        len(result.contradictions), len(quantitative),
    )
    return result


def _record_corroboration(result: DetectionResult, group: list[Claim]) -> None:
    """Record mutual support between agreeing claims from independent domains."""
    for a, b in itertools.combinations(group, 2):
        # Same publisher agreeing with itself is not independent evidence.
        if a.source_domain and a.source_domain == b.source_domain:
            continue
        result.corroborations.setdefault(a.id, []).append(b.id)
        result.corroborations.setdefault(b.id, []).append(a.id)
