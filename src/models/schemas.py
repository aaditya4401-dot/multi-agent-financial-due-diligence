from pydantic import BaseModel, Field
from typing import Literal
from src.state import Conflict


class ReportFinding(BaseModel):
    claim: str
    severity: Literal["Strong", "Watch", "Flag", "Low"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class ReportSection(BaseModel):
    findings: list[ReportFinding]
    section_confidence: float = Field(ge=0.0, le=1.0)


class SentimentSection(BaseModel):
    findings: list[ReportFinding]
    section_confidence: float = Field(ge=0.0, le=1.0)
    news_trajectory: str        # "Improving" | "Stable" | "Declining"
    developer_sentiment: str    # qualitative summary
    employee_sentiment: str     # qualitative summary


class DueDiligenceReport(BaseModel):
    company_name: str
    report_date: str
    overall_score: int = Field(ge=0, le=100)
    overall_verdict: Literal["Favorable", "Cautious", "Unfavorable"]
    risk_level: Literal["Low", "Moderate", "High"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    financial_section: ReportSection
    market_section: ReportSection
    risk_section: ReportSection
    sentiment_section: SentimentSection

    conflicts_detected: list[Conflict]
    executive_summary: str
