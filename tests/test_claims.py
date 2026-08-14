"""Tests for the claims layer — normalization, detection, confidence.

All deterministic and offline. This is the layer that replaces LLM guesswork
with arithmetic, so it earns real tests rather than smoke tests.
"""

from datetime import date, timedelta

import pytest

from src.claims.confidence import (
    RECENCY_FLOOR,
    aggregate_confidence,
    base_score,
    recency_factor,
    score_claims,
)
from src.claims.contradictions import detect
from src.claims.extract import ExtractedClaim, finding_to_claim, to_claim
from src.claims.models import Claim, SourceTier, registrable_domain, tier_for_source
from src.claims.ontology import (
    Predicate,
    Unit,
    normalize_period,
    parse_value,
    resolve_predicate,
)

TODAY = date(2026, 8, 14)


def claim(
    predicate=Predicate.PAYMENT_VOLUME,
    value=1.9e12,
    period="FY2025",
    tier=SourceTier.NEWS,
    url="https://example.com/a",
    days_old=0,
    subject="stripe",
    unit=Unit.USD,
) -> Claim:
    return Claim(
        subject=subject,
        assertion=f"{predicate.value} was {value}",
        source_url=url,
        source_tier=tier,
        observed_at=TODAY - timedelta(days=days_old),
        predicate=predicate,
        period=period,
        value=value,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Value parsing — the arithmetic the LLM is never trusted with
# ---------------------------------------------------------------------------

class TestParseValue:
    @pytest.mark.parametrize("raw,expected", [
        ("$1.9T", 1.9e12),
        ("1.9 trillion", 1.9e12),
        ("$1,900,000,000,000", 1.9e12),
        ("1.9T", 1.9e12),
        ("$159B", 159e9),
        ("159 billion", 159e9),
        ("$1B", 1e9),
        ("500M", 500e6),
        ("$500 million", 500e6),
        ("50k", 50e3),
    ])
    def test_magnitudes_normalise_to_same_number(self, raw, expected):
        assert parse_value(raw, Unit.USD) == pytest.approx(expected, rel=1e-9)

    def test_differently_written_values_compare_equal(self):
        assert parse_value("$1.9T", Unit.USD) == parse_value("1,900 billion", Unit.USD)

    @pytest.mark.parametrize("raw,expected", [
        ("34%", 34.0),
        ("0.34", 34.0),      # bare ratio for a percent predicate
        ("34.5%", 34.5),
        ("17%", 17.0),
    ])
    def test_percent_handling(self, raw, expected):
        assert parse_value(raw, Unit.PERCENT) == pytest.approx(expected)

    def test_negative_and_parenthesised(self):
        assert parse_value("-$500M", Unit.USD) == pytest.approx(-500e6)
        assert parse_value("($500M)", Unit.USD) == pytest.approx(-500e6)

    @pytest.mark.parametrize("raw", ["", None, "not a number", "N/A"])
    def test_unparseable_returns_none(self, raw):
        assert parse_value(raw, Unit.USD) is None


class TestResolvePredicate:
    @pytest.mark.parametrize("label", [
        "TPV", "tpv", "payment volume", "Total Payment Volume",
        "total payments processed", "processing volume", "GPV",
    ])
    def test_payment_volume_synonyms_collapse(self, label):
        assert resolve_predicate(label) is Predicate.PAYMENT_VOLUME

    def test_alias_inside_longer_phrase(self):
        assert resolve_predicate("total payment volume in 2025") is Predicate.PAYMENT_VOLUME

    def test_unknown_metric_returns_none(self):
        assert resolve_predicate("vibe score") is None
        assert resolve_predicate("") is None


class TestNormalizePeriod:
    @pytest.mark.parametrize("raw", ["FY2025", "fiscal 2025", "2025", "FY 2025"])
    def test_equivalent_years_collapse(self, raw):
        assert normalize_period(raw) == "FY2025"

    def test_quarters(self):
        assert normalize_period("Q3 2025") == "Q3-2025"

    def test_missing_period_is_unknown(self):
        assert normalize_period("") == "UNKNOWN"
        assert normalize_period(None) == "UNKNOWN"


class TestSourceTiering:
    @pytest.mark.parametrize("url,domain", [
        ("https://www.sec.gov/edgar/x", "sec.gov"),
        ("https://old.reddit.com/r/stripe", "reddit.com"),
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
    ])
    def test_registrable_domain(self, url, domain):
        assert registrable_domain(url) == domain

    def test_domain_beats_legacy_quality(self):
        assert tier_for_source("https://sec.gov/f", "news_article") is SourceTier.FILING

    def test_legacy_quality_used_when_domain_unknown(self):
        assert tier_for_source("", "official_filing") is SourceTier.FILING

    def test_filing_outranks_forum(self):
        assert claim(tier=SourceTier.FILING).prior() > claim(tier=SourceTier.FORUM).prior()


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    def test_the_readme_tpv_conflict_is_found_without_an_llm(self):
        """Financial says $1.9T, market says $1.4T, same metric and period."""
        a = claim(value=1.9e12, url="https://ft.com/x", tier=SourceTier.TIER1_NEWS)
        b = claim(value=1.4e12, url="https://blog.example.com/y", tier=SourceTier.NEWS)

        result = detect([a, b])

        assert len(result.contradictions) == 1
        found = result.contradictions[0]
        assert found.kind == "factual_contradiction"
        assert found.predicate is Predicate.PAYMENT_VOLUME
        assert found.spread == pytest.approx((1.9e12 - 1.4e12) / 1.9e12)
        assert found.winner is a, "higher source tier should win"

    def test_close_values_corroborate_instead(self):
        a = claim(value=1.90e12, url="https://ft.com/x")
        b = claim(value=1.92e12, url="https://reuters.com/y")

        result = detect([a, b])

        assert result.contradictions == []
        assert b.id in result.corroborations[a.id]

    def test_same_domain_does_not_self_corroborate(self):
        a = claim(value=1.90e12, url="https://ft.com/one")
        b = claim(value=1.90e12, url="https://ft.com/two")

        result = detect([a, b])

        assert result.corroborations == {}, "one publisher is one source"

    def test_different_periods_are_not_a_contradiction(self):
        a = claim(value=1.9e12, period="FY2025")
        b = claim(value=1.0e12, period="FY2023")

        kinds = {c.kind for c in detect([a, b]).contradictions}
        assert "factual_contradiction" not in kinds

    def test_stale_data_detected_across_a_long_gap(self):
        """$159B valuation now vs $50B from two years ago."""
        old = claim(predicate=Predicate.VALUATION, value=50e9, period="FY2023", days_old=730)
        new = claim(predicate=Predicate.VALUATION, value=159e9, period="FY2025", days_old=10)

        result = detect([old, new])

        stale = [c for c in result.contradictions if c.kind == "stale_data"]
        assert len(stale) == 1
        assert stale[0].winner is new, "the more recent figure should win"

    def test_tolerance_is_predicate_specific(self):
        """TAM estimates vary legitimately; revenue does not."""
        tam_a = claim(predicate=Predicate.TAM, value=1.0e12, url="https://a.com/1")
        tam_b = claim(predicate=Predicate.TAM, value=1.15e12, url="https://b.com/1")
        assert detect([tam_a, tam_b]).contradictions == []

        rev_a = claim(predicate=Predicate.REVENUE, value=1.0e10, url="https://a.com/2")
        rev_b = claim(predicate=Predicate.REVENUE, value=1.15e10, url="https://b.com/2")
        assert len(detect([rev_a, rev_b]).contradictions) == 1

    def test_qualitative_claims_are_never_compared(self):
        a = Claim(subject="stripe", assertion="compliance is challenging",
                  observed_at=TODAY, source_url="https://a.com")
        b = Claim(subject="stripe", assertion="compliance is fine",
                  observed_at=TODAY, source_url="https://b.com")

        assert detect([a, b]).contradictions == []

    def test_single_claim_produces_nothing(self):
        assert detect([claim()]).contradictions == []

    def test_empty_input(self):
        result = detect([])
        assert result.contradictions == [] and result.corroborations == {}


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_recency_decays_but_never_to_zero(self):
        fresh = recency_factor(claim(days_old=0), TODAY)
        old = recency_factor(claim(days_old=3650), TODAY)
        assert fresh > old >= RECENCY_FLOOR
        assert fresh == pytest.approx(1.0)

    def test_half_life_halves_the_decayable_part(self):
        # payment_volume half-life is 180 days
        factor = recency_factor(claim(days_old=180), TODAY)
        assert factor == pytest.approx(RECENCY_FLOOR + (1 - RECENCY_FLOOR) * 0.5)

    def test_filing_beats_forum_all_else_equal(self):
        filing = base_score(claim(tier=SourceTier.FILING), TODAY)
        forum = base_score(claim(tier=SourceTier.FORUM), TODAY)
        assert filing > forum

    def test_corroboration_raises_confidence(self):
        a = claim(value=1.90e12, url="https://a.com/x", tier=SourceTier.NEWS)
        b = claim(value=1.91e12, url="https://b.com/y", tier=SourceTier.NEWS)

        solo = claim(value=1.90e12, url="https://a.com/x", tier=SourceTier.NEWS)
        score_claims([solo], detect([solo]), TODAY)
        score_claims([a, b], detect([a, b]), TODAY)

        assert a.confidence > solo.confidence
        assert a.corroborated_by == [b.id]

    def test_forum_pile_cannot_outweigh_a_filing(self):
        filing = claim(value=1.0e12, tier=SourceTier.FILING, url="https://sec.gov/f")
        forums = [
            claim(value=2.0e12, tier=SourceTier.FORUM, url=f"https://forum{i}.com/p")
            for i in range(8)
        ]
        claims = [filing, *forums]
        score_claims(claims, detect(claims), TODAY)

        assert filing.confidence > max(f.confidence for f in forums)

    def test_agreeing_sources_are_not_demoted_with_the_dissenter(self):
        """Regression: only claims outside the winning cluster lose confidence.

        Two outlets agreeing with each other and a third dissenting is one
        conflict with a 2-claim winning side, not three losers.
        """
        ft = claim(value=1.90e12, tier=SourceTier.TIER1_NEWS, url="https://ft.com/x")
        reuters = claim(value=1.91e12, tier=SourceTier.TIER1_NEWS, url="https://reuters.com/x")
        blog = claim(value=1.40e12, tier=SourceTier.NEWS, url="https://blog.example.com/y")

        claims = [ft, reuters, blog]
        result = detect(claims)
        score_claims(claims, result, TODAY)

        contradiction = result.contradictions[0]
        loser_ids = {c.id for c in contradiction.losers()}
        assert loser_ids == {blog.id}, "corroborating sources must not be demoted"

        assert ft.confidence == pytest.approx(reuters.confidence, abs=0.02), (
            "two sources that agree, from equal tiers on the same date, "
            "should end up with near-identical confidence"
        )
        assert blog.confidence < ft.confidence

    def test_contradiction_demotes_the_loser(self):
        strong = claim(value=1.9e12, tier=SourceTier.FILING, url="https://sec.gov/a")
        weak = claim(value=1.4e12, tier=SourceTier.FORUM, url="https://reddit.com/b")

        claims = [strong, weak]
        score_claims(claims, detect(claims), TODAY)

        assert strong.confidence > weak.confidence
        assert strong.id in weak.contradicted_by

    def test_confidence_stays_in_range(self):
        claims = [
            claim(value=1.9e12, url=f"https://s{i}.com/x", tier=SourceTier.FILING)
            for i in range(10)
        ]
        score_claims(claims, detect(claims), TODAY)
        assert all(0.0 <= c.confidence <= 1.0 for c in claims)

    def test_aggregate_confidence_of_nothing_is_zero(self):
        assert aggregate_confidence([]) == 0.0

    def test_aggregate_favours_strong_evidence_over_padding(self):
        strong = [claim(tier=SourceTier.FILING, url="https://sec.gov/a")]
        score_claims(strong, None, TODAY)

        padded = list(strong) + [
            claim(tier=SourceTier.SOCIAL, url=f"https://x.com/{i}", value=1e9 * i)
            for i in range(1, 6)
        ]
        score_claims(padded, None, TODAY)

        assert aggregate_confidence(strong) > aggregate_confidence(padded)


# ---------------------------------------------------------------------------
# Extraction conversion (no LLM)
# ---------------------------------------------------------------------------

class TestExtractionConversion:
    def test_extracted_record_becomes_quantitative_claim(self):
        c = to_claim(
            ExtractedClaim(
                assertion="Stripe processed $1.9T in payments",
                metric="total payment volume",
                raw_value="$1.9T",
                period="2025",
                source_url="https://ft.com/a",
            ),
            subject="stripe", agent_name="financial", observed_at=TODAY,
        )
        assert c.is_quantitative
        assert c.predicate is Predicate.PAYMENT_VOLUME
        assert c.value == pytest.approx(1.9e12)
        assert c.period == "FY2025"
        assert c.source_tier is SourceTier.TIER1_NEWS

    def test_unparseable_number_degrades_to_qualitative(self):
        c = to_claim(
            ExtractedClaim(
                assertion="Revenue grew a lot",
                metric="revenue",
                raw_value="a lot",
                source_url="https://a.com",
            ),
            subject="stripe", agent_name="financial", observed_at=TODAY,
        )
        assert not c.is_quantitative, "must not invent false precision"
        assert c.assertion == "Revenue grew a lot"

    def test_unknown_metric_becomes_qualitative(self):
        c = to_claim(
            ExtractedClaim(assertion="Culture is strong", metric="vibe", raw_value="9"),
            subject="stripe", agent_name="sentiment", observed_at=TODAY,
        )
        assert not c.is_quantitative

    def test_legacy_finding_fallback_preserves_evidence(self):
        c = finding_to_claim(
            {
                "claim": "Compliance challenges with KYC",
                "source": "https://sec.gov/filing",
                "confidence": 0.9,
                "source_quality": "official_filing",
                "date_of_data": "2026-01-15",
            },
            subject="stripe", agent_name="risk",
        )
        assert c.assertion == "Compliance challenges with KYC"
        assert c.source_tier is SourceTier.FILING
        assert c.observed_at == date(2026, 1, 15)

    def test_bad_date_falls_back_to_today(self):
        c = finding_to_claim(
            {"claim": "x", "source": "https://a.com", "date_of_data": "garbage"},
            subject="stripe", agent_name="risk",
        )
        assert c.observed_at == date.today()


class TestPlausibilityGuard:
    """Regression: a live run extracted 'valuation = 49 usd' from a real page
    and it entered the graph at full confidence."""

    def test_absurd_valuation_demoted_to_qualitative(self):
        c = to_claim(
            ExtractedClaim(
                assertion="Stripe shares changed hands at $49",
                metric="valuation",
                raw_value="$49",
                source_url="https://news.example.com/a",
            ),
            subject="stripe", agent_name="financial", observed_at=TODAY,
        )
        assert not c.is_quantitative
        assert c.assertion.startswith("Stripe shares")

    @pytest.mark.parametrize("predicate,raw,ok", [
        (Predicate.VALUATION, "$159B", True),
        (Predicate.VALUATION, "$49", False),
        (Predicate.GLASSDOOR_RATING, "3.4", True),
        (Predicate.GLASSDOOR_RATING, "47", False),
        (Predicate.MARKET_SHARE, "17%", True),
        (Predicate.MARKET_SHARE, "450%", False),
        (Predicate.EMPLOYEE_COUNT, "8000", True),
        (Predicate.REVENUE_GROWTH_YOY, "-40%", True),
    ])
    def test_bounds(self, predicate, raw, ok):
        c = to_claim(
            ExtractedClaim(assertion="x", metric=predicate.value, raw_value=raw,
                           period="2025", source_url="https://a.com"),
            subject="stripe", agent_name="financial", observed_at=TODAY,
        )
        assert c.is_quantitative is ok

    def test_implausible_value_never_creates_a_conflict(self):
        good = to_claim(
            ExtractedClaim(assertion="valued at $159B", metric="valuation",
                           raw_value="$159B", source_url="https://a.com"),
            subject="stripe", agent_name="financial", observed_at=TODAY,
        )
        junk = to_claim(
            ExtractedClaim(assertion="price was $49", metric="valuation",
                           raw_value="$49", source_url="https://b.com"),
            subject="stripe", agent_name="market", observed_at=TODAY,
        )
        assert detect([good, junk]).contradictions == []


class TestUndatedDivergence:
    def test_mild_divergence_between_undated_claims_is_not_a_conflict(self):
        a = claim(predicate=Predicate.VALUATION, value=100e9, period="UNKNOWN",
                  url="https://a.com/x")
        b = claim(predicate=Predicate.VALUATION, value=120e9, period="UNKNOWN",
                  url="https://b.com/y")
        assert detect([a, b]).contradictions == []

    def test_extreme_divergence_between_undated_claims_is_flagged(self):
        """Regression: an undated group could hold wildly incompatible values
        and never be reported."""
        a = claim(predicate=Predicate.VALUATION, value=159e9, period="UNKNOWN",
                  url="https://a.com/x", tier=SourceTier.FILING)
        b = claim(predicate=Predicate.VALUATION, value=1e9, period="UNKNOWN",
                  url="https://b.com/y", tier=SourceTier.FORUM)

        contradictions = detect([a, b]).contradictions
        assert len(contradictions) == 1
        assert contradictions[0].winner is a


class TestClaimIdentity:
    def test_disagreeing_claims_share_a_group_but_not_an_id(self):
        a = claim(value=1.9e12, url="https://a.com/x")
        b = claim(value=1.4e12, url="https://b.com/y")
        assert a.group_key == b.group_key
        assert a.id != b.id, "identity must include the value, or conflicts collapse"

    def test_id_is_stable(self):
        assert claim().id == claim().id
