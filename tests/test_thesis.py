"""Tests for the investment thesis and, mostly, its grounding.

The prompt is not the interesting part — a model will write a fluent thesis
whether or not the research supports one. What is testable, and what these
cover, is whether an ungrounded thesis survives contact with the claim graph.
"""

import asyncio
from datetime import date

import pytest
from pydantic import ValidationError

from src.agents.thesis import (
    open_questions_from_gaps,
    thesis_node,
    validate_citations,
)
from src.claims.models import Claim, SourceTier
from src.claims.ontology import Predicate, Unit
from src.gaps.models import Gap, GapKind
from src.models.schemas import FinalReport, inconclusive_report
from src.models.thesis import (
    InvestmentThesis,
    RedFlag,
    ScenarioCase,
    ThesisDriver,
    insufficient_evidence_thesis,
)


def _claim(value=1.0e9, agent="financial"):
    return Claim(
        subject="testco", assertion=f"revenue {value}",
        source_url="https://sec.gov/a", source_tier=SourceTier.FILING,
        observed_at=date.today(), extracted_by=agent,
        predicate=Predicate.REVENUE, period="FY2025", value=value, unit=Unit.USD,
    )


def _driver(ids, statement="Revenue compounds above 25%"):
    return ThesisDriver(
        statement=statement, direction="supports", evidence_claim_ids=ids,
        confidence=0.7, what_would_falsify_this="growth falls below 15%",
    )


def _thesis(drivers, red_flags=None):
    case = ScenarioCase(narrative="n", key_assumptions=[])
    return InvestmentThesis(
        summary="s", drivers=drivers, bull_case=case,
        base_case=case.model_copy(), bear_case=case.model_copy(),
        red_flags=red_flags or [], recommendation="Proceed",
        recommendation_rationale="because",
    )


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------

class TestThesisSchema:
    def test_driver_must_cite_something(self):
        """A driver nothing supports is an opinion, not a driver."""
        with pytest.raises(ValidationError):
            _driver([])

    def test_driver_must_be_falsifiable(self):
        with pytest.raises(ValidationError):
            ThesisDriver(statement="x", evidence_claim_ids=["a"],
                         confidence=0.5, what_would_falsify_this="   ")

    def test_deal_breakers_are_separable_from_noise(self):
        thesis = _thesis([_driver(["a"])], red_flags=[
            RedFlag(issue="pending litigation", severity="deal_breaker"),
            RedFlag(issue="glassdoor dip", severity="monitor"),
        ])
        assert [f.issue for f in thesis.deal_breakers] == ["pending litigation"]

    def test_insufficient_evidence_is_not_a_pass(self):
        """Declining to invest is a conclusion; having no evidence is not."""
        thesis = insufficient_evidence_thesis("nothing was found")
        assert thesis.recommendation == "Insufficient evidence"
        assert thesis.recommendation != "Pass"


# ---------------------------------------------------------------------------
# Grounding — the point of the phase
# ---------------------------------------------------------------------------

class TestCitationValidation:
    def test_valid_citations_survive(self):
        claim = _claim()
        cleaned, problems = validate_citations(_thesis([_driver([claim.id])]), [claim])
        assert len(cleaned.drivers) == 1
        assert problems == []

    def test_invented_ids_are_stripped(self):
        claim = _claim()
        cleaned, problems = validate_citations(
            _thesis([_driver([claim.id, "deadbeefdeadbeef"])]), [claim])
        assert cleaned.drivers[0].evidence_claim_ids == [claim.id]
        assert any("unknown claim id" in p for p in problems)

    def test_driver_citing_only_invented_ids_is_dropped(self):
        """The failure this whole phase exists to catch: a fluent driver that
        was not derived from anything the system actually found."""
        claim = _claim()
        cleaned, problems = validate_citations(
            _thesis([_driver(["nope000000000000"])]), [claim])
        assert cleaned.drivers == []
        assert any("dropped driver" in p for p in problems)

    def test_losing_every_driver_forces_insufficient_evidence(self):
        claim = _claim()
        cleaned, _ = validate_citations(
            _thesis([_driver(["fake111111111111"])]), [claim])
        assert cleaned.recommendation == "Insufficient evidence"
        assert "traced" in cleaned.recommendation_rationale

    def test_red_flag_citations_are_cleaned_but_the_flag_is_kept(self):
        """A red flag is worth surfacing even if its citation was wrong —
        unlike a driver, it does not carry the investment case."""
        claim = _claim()
        cleaned, _ = validate_citations(
            _thesis([_driver([claim.id])],
                    red_flags=[RedFlag(issue="fine", severity="material",
                                       evidence_claim_ids=["bogus00000000000"])]),
            [claim])
        assert len(cleaned.red_flags) == 1
        assert cleaned.red_flags[0].evidence_claim_ids == []

    def test_a_thesis_reports_what_it_leans_on(self):
        claim = _claim()
        thesis = _thesis([_driver([claim.id])])
        assert thesis.supported_by == {claim.id}


# ---------------------------------------------------------------------------
# Open questions come from the gap analyzer, not from the model
# ---------------------------------------------------------------------------

class TestOpenQuestions:
    def test_questions_are_generated_from_gaps(self):
        gaps = [
            Gap(kind=GapKind.MISSING_METRIC, agent="financial", subject="testco",
                predicate="revenue", detail="no revenue figure was found",
                value_of_information=0.9),
            Gap(kind=GapKind.STALE_EVIDENCE, agent="market", subject="testco",
                detail="market share data is three years old",
                value_of_information=0.3),
        ]
        questions = open_questions_from_gaps(gaps)
        assert questions[0].startswith("No revenue figure")
        assert len(questions) == 2

    def test_questions_are_ranked_by_value_of_information(self):
        gaps = [
            Gap(kind=GapKind.STALE_EVIDENCE, agent="market", subject="t",
                detail="low priority", value_of_information=0.1),
            Gap(kind=GapKind.MISSING_METRIC, agent="financial", subject="t",
                detail="high priority", value_of_information=0.95),
        ]
        assert open_questions_from_gaps(gaps)[0] == "High priority"

    def test_no_gaps_yields_no_questions(self):
        assert open_questions_from_gaps([]) == []


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class TestThesisNode:
    def test_no_claims_short_circuits_without_an_llm_call(self):
        """No API key is set, so reaching a model would raise. This proves the
        short circuit fires first."""
        gaps = [Gap(kind=GapKind.MISSING_METRIC, agent="financial",
                    subject="testco", detail="nothing found",
                    value_of_information=0.9)]
        out = asyncio.run(thesis_node(
            {"company": "TestCo", "claims": [], "gaps": gaps}))

        thesis = out["thesis"]
        assert thesis.recommendation == "Insufficient evidence"
        assert thesis.open_questions == ["Nothing found"]

    def test_model_failure_degrades_rather_than_crashing(self):
        """CLAUDE.md: never crash the pipeline. With no API key the structured
        call raises, and the node must still return a usable thesis."""
        out = asyncio.run(thesis_node(
            {"company": "TestCo", "claims": [_claim()], "gaps": [],
             "conflicts": [], "plan": None}))
        assert out["thesis"].recommendation == "Insufficient evidence"


class TestFinalReport:
    def test_report_carries_the_thesis(self):
        base = inconclusive_report("TestCo", "2026-08-19", "reason")
        final = FinalReport(**base.model_dump(),
                            thesis=insufficient_evidence_thesis("no data"))
        dumped = final.model_dump()
        assert dumped["thesis"]["recommendation"] == "Insufficient evidence"
        assert FinalReport.model_validate(dumped)

    def test_thesis_is_optional(self):
        """The synthesizer must still emit a report if the thesis node failed."""
        base = inconclusive_report("TestCo", "2026-08-19", "reason")
        assert FinalReport(**base.model_dump()).thesis is None
