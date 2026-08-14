"""The Claim — the unit of evidence the system reasons over.

Replaces the old ``Finding``, which was a 500-character slice of raw web text
with a hardcoded confidence. A Claim is either:

* **quantitative** — a number pinned to (subject, predicate, period, unit),
  which makes it mechanically comparable to every other claim about the same
  thing; or
* **qualitative** — a sourced assertion that carries no number. These still
  earn confidence and still reach the report, but they never participate in
  groupby contradiction detection, because you cannot subtract two sentences.

Being explicit about that split is the point. Pretending prose is comparable is
what made the old conflict resolver unreliable.
"""

import hashlib
from datetime import date
from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.claims.ontology import Predicate, Unit


class SourceTier(str, Enum):
    """How much a source is worth before any other evidence is considered."""

    FILING = "official_filing"
    EARNINGS = "earnings_call"
    TIER1_NEWS = "tier1_news"
    COMPANY = "company_source"
    NEWS = "news_article"
    FORUM = "forum"
    SOCIAL = "social_media"
    UNKNOWN = "unknown"


#: Prior probability that a claim from this tier is accurate, before recency
#: decay and corroboration are applied.
TIER_PRIOR: dict[SourceTier, float] = {
    SourceTier.FILING: 0.95,
    SourceTier.EARNINGS: 0.90,
    SourceTier.TIER1_NEWS: 0.75,
    SourceTier.COMPANY: 0.70,
    SourceTier.NEWS: 0.55,
    SourceTier.FORUM: 0.35,
    SourceTier.SOCIAL: 0.30,
    SourceTier.UNKNOWN: 0.30,
}

#: Domain → tier. Keeps tiering out of the LLM's hands where a URL settles it.
_DOMAIN_TIERS: dict[str, SourceTier] = {
    "sec.gov": SourceTier.FILING,
    "investor.gov": SourceTier.FILING,
    "ft.com": SourceTier.TIER1_NEWS,
    "wsj.com": SourceTier.TIER1_NEWS,
    "reuters.com": SourceTier.TIER1_NEWS,
    "bloomberg.com": SourceTier.TIER1_NEWS,
    "economist.com": SourceTier.TIER1_NEWS,
    "nytimes.com": SourceTier.TIER1_NEWS,
    "reddit.com": SourceTier.FORUM,
    "news.ycombinator.com": SourceTier.FORUM,
    "quora.com": SourceTier.FORUM,
    "x.com": SourceTier.SOCIAL,
    "twitter.com": SourceTier.SOCIAL,
    "facebook.com": SourceTier.SOCIAL,
    "linkedin.com": SourceTier.SOCIAL,
}

#: Legacy ``Finding.source_quality`` strings → tiers, so pre-existing data and
#: the yfinance tool path keep working.
_LEGACY_QUALITY: dict[str, SourceTier] = {
    "official_filing": SourceTier.FILING,
    "earnings_call": SourceTier.EARNINGS,
    "news_article": SourceTier.NEWS,
    "social_media": SourceTier.SOCIAL,
    "forum": SourceTier.FORUM,
}


def registrable_domain(url: str) -> str:
    """Best-effort registrable domain, used to judge source independence.

    Five syndicated copies of one wire story are one source, not five, so
    corroboration counts distinct domains rather than distinct URLs.
    """
    if not url:
        return ""
    parsed = urlparse(url if "//" in url else f"//{url}")
    host = (parsed.netloc or parsed.path).lower().strip("/")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Handle co.uk / com.au style suffixes.
    if parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def tier_for_source(url: str, legacy_quality: str | None = None) -> SourceTier:
    """Infer a source tier from a URL, falling back to a legacy quality string."""
    domain = registrable_domain(url)
    if domain in _DOMAIN_TIERS:
        return _DOMAIN_TIERS[domain]
    if legacy_quality and legacy_quality in _LEGACY_QUALITY:
        return _LEGACY_QUALITY[legacy_quality]
    if domain:
        return SourceTier.NEWS
    return SourceTier.UNKNOWN


class Claim(BaseModel):
    """One sourced assertion about a company."""

    subject: str                                   # normalised entity, e.g. "stripe"
    assertion: str                                 # human-readable statement
    source_url: str = ""
    source_tier: SourceTier = SourceTier.UNKNOWN
    observed_at: date
    extracted_by: str = ""                         # which agent produced it

    # Quantitative payload — all None/NONE for qualitative claims.
    predicate: Predicate | None = None
    period: str = "UNKNOWN"
    value: float | None = None
    unit: Unit = Unit.NONE

    # Populated by the confidence pass.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    corroborated_by: list[str] = Field(default_factory=list)   # claim ids
    contradicted_by: list[str] = Field(default_factory=list)   # claim ids

    model_config = {"frozen": False}

    @property
    def is_quantitative(self) -> bool:
        return self.predicate is not None and self.value is not None

    @property
    def group_key(self) -> tuple[str, str, str]:
        """What contradiction detection groups on.

        Claims sharing a group key are talking about the same number, so any
        disagreement between them is a real conflict rather than a topic change.
        """
        predicate = self.predicate.value if self.predicate else "qualitative"
        return (self.subject, predicate, self.period)

    @property
    def source_domain(self) -> str:
        return registrable_domain(self.source_url)

    @property
    def id(self) -> str:
        """Stable identity for *this* claim.

        Deliberately distinct from :attr:`group_key`: two claims may share a
        group key and disagree — that is exactly the case we need to detect,
        so identity must include the value and the source.
        """
        parts = [*self.group_key, str(self.value), self.source_url, self.assertion[:120]]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def prior(self) -> float:
        return TIER_PRIOR.get(self.source_tier, TIER_PRIOR[SourceTier.UNKNOWN])

    def describe(self) -> str:
        """Compact one-line rendering for prompts and logs."""
        if self.is_quantitative:
            return (
                f"{self.subject} {self.predicate.value} [{self.period}] = "
                f"{self.value:,.4g} {self.unit.value} "
                f"({self.source_tier.value}, conf={self.confidence:.2f})"
            )
        return f"{self.subject}: {self.assertion[:160]} ({self.source_tier.value})"
