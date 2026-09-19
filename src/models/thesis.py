"""The investment thesis — what we believe, why, and what would change our mind.

The report already had a verdict, a score and a risk level. What it lacked was
a *thesis*: an explicit statement of the handful of things the case rests on,
each tied to evidence and each falsifiable.

Two deliberate choices here, both aimed at the same failure mode — a fluent memo
that is not actually supported by anything the system found.

**Every driver must cite claim ids.** A driver with no citations fails schema
validation outright. Citations naming claims that do not exist in the run are
dropped by :func:`~src.agents.thesis.validate_citations`. This is only possible
because the claim graph exists and every claim has a stable id — the payoff for
having built typed claims rather than prose findings.

**Every driver must say what would falsify it.** "Revenue is growing" is not a
thesis driver; it is an observation. "Payment volume growth stays above 25%
through FY2026, which fails if the top-five merchant concentration reported in
the S-1 keeps rising" is one, because you can go and check.

Deliberately *not* modelled: SWOT and Porter's Five Forces. Neither appears in
real diligence memos — they are teaching frameworks, and generating them would
make the output look less like the artefact it is imitating, not more.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Recommendation = Literal[
    "Proceed",
    "Proceed with conditions",
    "Pass",
    "Insufficient evidence",
]

Severity = Literal["deal_breaker", "material", "monitor"]


class ThesisDriver(BaseModel):
    """One falsifiable proposition the investment case depends on."""

    statement: str = Field(
        description="A specific, checkable proposition — not an observation."
    )
    direction: Literal["supports", "challenges"] = "supports"
    evidence_claim_ids: list[str] = Field(
        min_length=1,
        description="Ids of claims backing this driver. At least one is "
                    "required: a driver nothing supports is an opinion.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    what_would_falsify_this: str = Field(
        description="The observation that would overturn this driver. If "
                    "nothing could, it is not a driver."
    )

    @field_validator("what_would_falsify_this")
    @classmethod
    def _must_be_falsifiable(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "a driver with no falsification condition is not a thesis driver"
            )
        return value


class ScenarioCase(BaseModel):
    """One of the three ways this could go."""

    narrative: str
    key_assumptions: list[str] = Field(default_factory=list)


class RedFlag(BaseModel):
    """Something wrong, graded by whether it ends the conversation."""

    issue: str
    severity: Severity = "monitor"
    evidence_claim_ids: list[str] = Field(default_factory=list)


class InvestmentThesis(BaseModel):
    """The analytical core of the memo."""

    summary: str = Field(description="The thesis in one sentence.")
    drivers: list[ThesisDriver] = Field(default_factory=list)

    bull_case: ScenarioCase
    base_case: ScenarioCase
    bear_case: ScenarioCase

    red_flags: list[RedFlag] = Field(default_factory=list)

    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved diligence items. Overwritten from the gap "
                    "analyzer's findings rather than invented, so 'further "
                    "work required' reflects what the system actually could "
                    "not establish.",
    )

    recommendation: Recommendation = "Insufficient evidence"
    recommendation_rationale: str = ""

    @property
    def deal_breakers(self) -> list[RedFlag]:
        return [f for f in self.red_flags if f.severity == "deal_breaker"]

    @property
    def supported_by(self) -> set[str]:
        """Every claim id this thesis leans on."""
        ids: set[str] = set()
        for driver in self.drivers:
            ids.update(driver.evidence_claim_ids)
        for flag in self.red_flags:
            ids.update(flag.evidence_claim_ids)
        return ids


def insufficient_evidence_thesis(reason: str) -> InvestmentThesis:
    """A thesis representing 'we cannot form a view'.

    Deliberately not a 'Pass'. Declining to invest is a conclusion; having no
    evidence is the absence of one, and conflating them would let a failed
    pipeline read as a negative judgement about a company.
    """
    empty = ScenarioCase(narrative="Not established.", key_assumptions=[])
    return InvestmentThesis(
        summary=f"No investment thesis could be formed: {reason}",
        drivers=[],
        bull_case=empty,
        base_case=empty.model_copy(),
        bear_case=empty.model_copy(),
        red_flags=[],
        open_questions=[],
        recommendation="Insufficient evidence",
        recommendation_rationale=reason,
    )
