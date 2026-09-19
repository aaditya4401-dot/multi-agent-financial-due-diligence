"""What an evaluation of one run looks like.

Two layers are deliberately kept apart, because they answer different questions
and fail in different ways:

* **Groundedness** — does every assertion in the report trace to evidence the
  system actually gathered? Deterministic, cheap, runs on every report.
* **Quality** — is the thesis falsifiable, does the verdict follow, are the
  conflicts addressed? Needs judgement, so it costs a model call and runs
  offline over a fixture set.

Groundedness comes first, and not merely in file order. A judge scoring prose
without a groundedness check is theatre: a fluent report that cites nothing
real will score well on rubric criteria about clarity and structure.
"""

from pydantic import BaseModel, Field


class CitationAudit(BaseModel):
    """What survived citation checking when the thesis was formed.

    Recorded at validation time because it is unrecoverable afterwards — by the
    time the thesis reaches the report, every dropped driver is simply gone.
    """

    drivers_proposed: int = 0
    drivers_kept: int = 0
    invented_citations: int = 0

    @property
    def grounding_rate(self) -> float:
        """Fraction of proposed drivers that turned out to be evidence-backed."""
        if self.drivers_proposed == 0:
            return 0.0
        return round(self.drivers_kept / self.drivers_proposed, 4)


class GroundednessCheck(BaseModel):
    """One deterministic check over a finished report."""

    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    detail: str = ""


class EvaluationReport(BaseModel):
    """The verdict on the report, as opposed to on the company."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    checks: list[GroundednessCheck] = Field(default_factory=list)
    unsupported_figures: list[str] = Field(
        default_factory=list,
        description="Magnitudes asserted in the report prose that match no "
                    "claim the system gathered.",
    )
    citation_audit: CitationAudit | None = None

    @property
    def failed_checks(self) -> list[GroundednessCheck]:
        return [c for c in self.checks if not c.passed]

    def describe(self) -> str:
        state = "OK" if not self.failed_checks else (
            f"{len(self.failed_checks)} check(s) failed")
        return f"groundedness {self.score:.0%} — {state}"
