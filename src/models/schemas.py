"""Report schemas — the single source of truth for the synthesizer's output.

These models are used directly as the ``with_structured_output`` target, so
the shape the LLM is asked for and the shape the UI renders can never drift
apart. (They previously did: the synthesizer carried its own parallel copy.)
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.eval.models import EvaluationReport
from src.models.thesis import InvestmentThesis

Verdict = Literal["Favorable", "Cautious", "Unfavorable", "Inconclusive"]
RiskLevel = Literal["Low", "Moderate", "High", "Unknown"]
Severity = Literal["Strong", "Watch", "Flag", "Low"]
ConflictType = Literal["factual_contradiction", "complementary_tension", "stale_data"]


class ReportFinding(BaseModel):
    claim: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class ReportSection(BaseModel):
    findings: list[ReportFinding] = Field(default_factory=list)
    section_confidence: float = Field(ge=0.0, le=1.0)
    available: bool = Field(
        default=True,
        description="False when the agent behind this section failed to run. "
                    "An unavailable section means unknown, not negative.",
    )


class SentimentSection(ReportSection):
    news_trajectory: str = "Unknown"
    developer_sentiment: str = "Unknown"
    employee_sentiment: str = "Unknown"


class ConflictModel(BaseModel):
    type: ConflictType
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    resolution: str
    resolved_confidence: float = Field(ge=0.0, le=1.0)


class DueDiligenceReport(BaseModel):
    company_name: str
    report_date: str
    overall_score: int = Field(ge=0, le=100)
    overall_verdict: Verdict
    risk_level: RiskLevel
    overall_confidence: float = Field(ge=0.0, le=1.0)

    financial_section: ReportSection
    market_section: ReportSection
    risk_section: ReportSection
    sentiment_section: SentimentSection

    conflicts_detected: list[ConflictModel] = Field(default_factory=list)
    executive_summary: str

    unavailable_sections: list[str] = Field(
        default_factory=list,
        description="Names of agents that ran and failed. Their sections carry "
                    "no signal and must not be read as negative findings.",
    )

    skipped_sections: list[str] = Field(
        default_factory=list,
        description="Names of agents deliberately left out of the research "
                    "plan. Distinct from a failure: nothing went wrong and no "
                    "evidence is missing, so these must not reduce confidence.",
    )


class FinalReport(DueDiligenceReport):
    """The artefact the UI renders: the drafted report plus the thesis.

    Kept separate from :class:`DueDiligenceReport` so the synthesizer's
    ``with_structured_output`` target contains only what the model should
    author. The thesis is formed by its own node against the claim graph and
    attached here — asking the synthesizer to write one too would produce a
    second, ungrounded version of the same view.
    """

    thesis: Optional[InvestmentThesis] = None
    evaluation: Optional[EvaluationReport] = None


def inconclusive_report(company: str, report_date: str, reason: str) -> DueDiligenceReport:
    """Build a report representing 'we could not analyze this'.

    Deliberately *not* a zero score with an Unfavorable verdict — a pipeline
    failure is an absence of evidence, not evidence of a bad investment.
    """
    empty = ReportSection(findings=[], section_confidence=0.0, available=False)
    return DueDiligenceReport(
        company_name=company,
        report_date=report_date,
        overall_score=0,
        overall_verdict="Inconclusive",
        risk_level="Unknown",
        overall_confidence=0.0,
        financial_section=empty,
        market_section=empty.model_copy(),
        risk_section=empty.model_copy(),
        sentiment_section=SentimentSection(
            findings=[], section_confidence=0.0, available=False
        ),
        conflicts_detected=[],
        executive_summary=f"Analysis could not be completed: {reason}",
        unavailable_sections=["financial", "market", "risk", "sentiment"],
    )
