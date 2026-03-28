import json
import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.state import DueDiligenceState, AgentFindings, Conflict
from src.models.schemas import DueDiligenceReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior investment analyst writing a due diligence report. "
    "Synthesize findings from financial, market, risk, and sentiment analyses. "
    "Account for detected conflicts and their resolutions. "
    "Assign an overall score (0-100), verdict, and risk level. "
    "Be concise but thorough."
)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output (self-contained, no TypedDict refs)
# ---------------------------------------------------------------------------

class LLMReportFinding(BaseModel):
    claim: str
    severity: Literal["Strong", "Watch", "Flag", "Low"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class LLMReportSection(BaseModel):
    findings: list[LLMReportFinding]
    section_confidence: float = Field(ge=0.0, le=1.0)


class LLMSentimentSection(BaseModel):
    findings: list[LLMReportFinding]
    section_confidence: float = Field(ge=0.0, le=1.0)
    news_trajectory: str
    developer_sentiment: str
    employee_sentiment: str


class LLMConflict(BaseModel):
    type: Literal["factual_contradiction", "complementary_tension", "stale_data"]
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    resolution: str
    resolved_confidence: float = Field(ge=0.0, le=1.0)


class LLMReport(BaseModel):
    """Full due diligence report — used as structured output target for the LLM."""
    company_name: str
    report_date: str
    overall_score: int = Field(ge=0, le=100)
    overall_verdict: Literal["Favorable", "Cautious", "Unfavorable"]
    risk_level: Literal["Low", "Moderate", "High"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    financial_section: LLMReportSection
    market_section: LLMReportSection
    risk_section: LLMReportSection
    sentiment_section: LLMSentimentSection

    conflicts_detected: list[LLMConflict]
    executive_summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_findings(findings: AgentFindings | None) -> str:
    if not findings:
        return "(no findings)"
    lines = [f"Agent: {findings['agent_name']}", f"Summary: {findings['summary'][:500]}"]
    for i, f in enumerate(findings["findings"], 1):
        lines.append(
            f"  {i}. [{f['source_quality']}, confidence={f['confidence']}, "
            f"date={f['date_of_data']}] {f['claim'][:300]}"
        )
    return "\n".join(lines)


def _format_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "(no conflicts detected)"
    lines = []
    for i, c in enumerate(conflicts, 1):
        lines.append(
            f"  {i}. [{c['type']}] {c['agent_a']} vs {c['agent_b']}: "
            f"'{c['claim_a'][:150]}' vs '{c['claim_b'][:150]}' — "
            f"Resolution: {c['resolution'][:200]} (confidence={c['resolved_confidence']})"
        )
    return "\n".join(lines)


def _build_user_prompt(state: DueDiligenceState) -> str:
    sections = [
        ("FINANCIAL FINDINGS", state.get("financial_findings")),
        ("MARKET FINDINGS", state.get("market_findings")),
        ("RISK FINDINGS", state.get("risk_findings")),
        ("SENTIMENT FINDINGS", state.get("sentiment_findings")),
    ]
    parts = [f"=== {label} ===\n{_format_findings(f)}" for label, f in sections]

    conflicts = state.get("conflicts", [])
    parts.append(f"=== CONFLICTS DETECTED ===\n{_format_conflicts(conflicts)}")

    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"Produce a due diligence report for {state['company']} (report date: {today}).\n\n"
        + "\n\n".join(parts)
        + "\n\nSynthesize everything into a structured investment memo."
    )


def _to_final_report(llm_report: LLMReport) -> dict:
    """Convert the LLM's Pydantic output to a DueDiligenceReport-compatible dict."""
    return llm_report.model_dump()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def synthesizer_node(state: DueDiligenceState) -> dict:
    """LangGraph node: synthesize all findings into a final DueDiligenceReport."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(LLMReport)

        result: LLMReport = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state)},
        ])

        report = _to_final_report(result)

    except Exception as exc:
        logger.error("Synthesizer failed: %s", exc)
        report = {
            "company_name": state["company"],
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "overall_score": 0,
            "overall_verdict": "Unfavorable",
            "risk_level": "High",
            "overall_confidence": 0.0,
            "financial_section": {"findings": [], "section_confidence": 0.0},
            "market_section": {"findings": [], "section_confidence": 0.0},
            "risk_section": {"findings": [], "section_confidence": 0.0},
            "sentiment_section": {
                "findings": [], "section_confidence": 0.0,
                "news_trajectory": "Unknown",
                "developer_sentiment": "Unknown",
                "employee_sentiment": "Unknown",
            },
            "conflicts_detected": [],
            "executive_summary": f"Report generation failed: {exc}",
        }

    return {"final_report": report}
