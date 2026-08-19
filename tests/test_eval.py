"""Tests for the evaluation layer.

Groundedness is deterministic, so it is testable exactly. The judge is not
tested for its opinions — only that it degrades safely and that its arithmetic
is right.
"""

import asyncio
from datetime import date

from src.agents.evaluation import evaluation_node
from src.agents.thesis import audit_citations
from src.claims.models import Claim, SourceTier
from src.claims.ontology import Predicate, Unit
from src.eval.groundedness import (
    evaluate,
    extract_figures,
    is_supported,
    report_prose,
    unsupported_figures,
)
from src.eval.judge import CriterionScore, JudgeVerdict, judge_report
from src.eval.models import CitationAudit
from src.models.thesis import InvestmentThesis, ScenarioCase, ThesisDriver


def _claim(value, unit=Unit.USD, predicate=Predicate.PAYMENT_VOLUME,
           agent="financial", url="https://ft.com/a"):
    return Claim(
        subject="testco", assertion=f"{predicate.value} {value}", source_url=url,
        source_tier=SourceTier.TIER1_NEWS, observed_at=date.today(),
        extracted_by=agent, predicate=predicate, period="FY2025",
        value=value, unit=unit,
    )


def _report(summary="", thesis=None, **over):
    base = {
        "company_name": "TestCo",
        "executive_summary": summary,
        "financial_section": {"findings": [], "available": True,
                              "section_confidence": 0.6},
        "skipped_sections": [],
        "thesis": thesis,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

class TestFigureExtraction:
    def test_magnitudes_and_percentages_are_found(self):
        figures = extract_figures("Processed $1.4T, up 38%, with 100 million users.")
        assert "$1.4T" in figures
        assert "38%" in figures
        assert "100 million" in figures

    def test_years_are_not_treated_as_figures(self):
        """Otherwise every report date and fiscal year is a false positive,
        and the real findings drown in noise."""
        assert extract_figures("In 2024 revenue rose, and in 2025 it fell.") == []

    def test_bare_counts_are_not_treated_as_figures(self):
        assert extract_figures("The company has 5 offices and 3 products.") == []


class TestFigureSupport:
    def test_exact_match_is_supported(self):
        assert is_supported("$1.4T", [_claim(1.4e12)])

    def test_near_match_within_tolerance_is_supported(self):
        assert is_supported("$1.41T", [_claim(1.4e12)])

    def test_wildly_different_figure_is_unsupported(self):
        assert not is_supported("$9.9T", [_claim(1.4e12)])

    def test_percentages_do_not_match_magnitudes(self):
        """34% and a claim of 34 units are not the same fact."""
        assert not is_supported("34%", [_claim(34.0, unit=Unit.USD)])

    def test_percentage_matches_a_percent_claim(self):
        assert is_supported("34%", [
            _claim(34.0, unit=Unit.PERCENT, predicate=Predicate.GROSS_MARGIN)])

    def test_unparseable_figure_is_not_reported_as_a_problem(self):
        """Absence of a parse is not evidence of a hallucination."""
        assert is_supported("$", [_claim(1.4e12)])


class TestUnsupportedFigures:
    def test_hallucinated_figure_is_caught(self):
        """The failure that matters: a number that reads as researched fact
        and matches nothing the system found."""
        report = _report("TestCo processed $1.4T and holds $88B in cash.")
        assert unsupported_figures(report, [_claim(1.4e12)]) == ["$88B"]

    def test_supported_report_is_clean(self):
        report = _report("TestCo processed $1.4T last year.")
        assert unsupported_figures(report, [_claim(1.4e12)]) == []

    def test_thesis_prose_is_checked_too(self):
        """The thesis is where a model is most tempted to invent numbers."""
        case = ScenarioCase(narrative="Revenue reaches $50B by 2027.")
        thesis = InvestmentThesis(
            summary="s", drivers=[], bull_case=case,
            base_case=case.model_copy(), bear_case=case.model_copy(),
        ).model_dump()
        assert "$50B" in unsupported_figures(_report(thesis=thesis), [_claim(1.4e12)])

    def test_prose_collector_reaches_every_authored_field(self):
        case = ScenarioCase(narrative="bull narrative")
        thesis = InvestmentThesis(
            summary="thesis summary", drivers=[], bull_case=case,
            base_case=case.model_copy(), bear_case=case.model_copy(),
            recommendation_rationale="because reasons",
        ).model_dump()
        prose = report_prose(_report("exec summary", thesis=thesis))
        for expected in ("exec summary", "thesis summary", "bull narrative",
                         "because reasons"):
            assert expected in prose


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_clean_report_scores_well(self):
        claims = [_claim(1.4e12)]
        audit = CitationAudit(drivers_proposed=3, drivers_kept=3)
        result = evaluate(_report("Processed $1.4T."), claims, audit)
        assert result.score > 0.9
        assert result.failed_checks == []

    def test_hallucinated_figure_fails_a_check(self):
        result = evaluate(_report("Processed $1.4T and $88B in cash."),
                          [_claim(1.4e12)],
                          CitationAudit(drivers_proposed=1, drivers_kept=1))
        names = {c.name for c in result.failed_checks}
        assert "figures_traceable" in names
        assert result.unsupported_figures == ["$88B"]

    def test_dropped_drivers_fail_the_thesis_check(self):
        result = evaluate(_report("Processed $1.4T."), [_claim(1.4e12)],
                          CitationAudit(drivers_proposed=4, drivers_kept=2,
                                        invented_citations=3))
        thesis_check = next(c for c in result.checks
                            if c.name == "thesis_drivers_grounded")
        assert not thesis_check.passed
        assert thesis_check.score == 0.5

    def test_unsourced_claims_fail_the_sourcing_check(self):
        result = evaluate(_report(""), [_claim(1.4e12, url="")], None)
        sourcing = next(c for c in result.checks if c.name == "claims_have_sources")
        assert not sourcing.passed

    def test_no_claims_scores_badly_rather_than_crashing(self):
        """A report with nothing behind it scores low, but the empty-figure
        check still legitimately passes — asserting no figures is not a fault."""
        result = evaluate(_report(""), [], None)
        assert result.score < 0.3
        assert {c.name for c in result.failed_checks} >= {
            "claims_have_sources", "thesis_drivers_grounded"}


class TestCitationAudit:
    def test_audit_counts_what_was_proposed_before_validation(self):
        """Must run before validation, which destroys the evidence of what
        the model originally claimed."""
        claim = _claim(1.4e12)
        good = ThesisDriver(statement="a", evidence_claim_ids=[claim.id],
                            confidence=0.7, what_would_falsify_this="x")
        bad = ThesisDriver(statement="b", evidence_claim_ids=["fake000000000000"],
                           confidence=0.9, what_would_falsify_this="y")
        case = ScenarioCase(narrative="n")
        thesis = InvestmentThesis(
            summary="s", drivers=[good, bad], bull_case=case,
            base_case=case.model_copy(), bear_case=case.model_copy())

        audit = audit_citations(thesis, [claim])
        assert audit.drivers_proposed == 2
        assert audit.drivers_kept == 1
        assert audit.invented_citations == 1
        assert audit.grounding_rate == 0.5


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class TestEvaluationNode:
    def test_score_is_attached_to_the_report(self):
        out = asyncio.run(evaluation_node({
            "company": "TestCo",
            "final_report": _report("Processed $1.4T."),
            "claims": [_claim(1.4e12)],
            "thesis_audit": CitationAudit(drivers_proposed=1, drivers_kept=1),
        }))
        assert out["final_report"]["evaluation"]["score"] > 0.9
        assert out["evaluation"].score > 0.9

    def test_evaluation_never_loses_the_report(self):
        """It is the last node — a failure here must not discard finished work."""
        out = asyncio.run(evaluation_node({
            "company": "TestCo", "final_report": None, "claims": None,
            "thesis_audit": None,
        }))
        assert "final_report" not in out or out.get("final_report") is not None


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class TestJudge:
    def test_mean_and_weakest(self):
        verdict = JudgeVerdict(scores=[
            CriterionScore(criterion="thesis_falsifiable", score=5, justification="a"),
            CriterionScore(criterion="conflicts_addressed", score=2, justification="b"),
        ])
        assert verdict.mean == 3.5
        assert verdict.weakest.criterion == "conflicts_addressed"

    def test_empty_verdict_is_zero_not_an_error(self):
        assert JudgeVerdict().mean == 0.0
        assert JudgeVerdict().weakest is None

    def test_judge_failure_degrades(self):
        """No API key is set, so the call raises — the judge must absorb it."""
        verdict = asyncio.run(judge_report(_report("anything")))
        assert verdict.scores == []
        assert "failed" in verdict.summary
