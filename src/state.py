from typing import TypedDict, Annotated, Optional
from operator import add


class Finding(TypedDict):
    claim: str
    source: str
    confidence: float           # 0.0 to 1.0
    source_quality: str         # "official_filing" | "news_article" | "social_media" | "forum"
    date_of_data: str


class AgentFindings(TypedDict):
    agent_name: str
    findings: list[Finding]
    summary: str
    data_sources_used: list[str]


class Conflict(TypedDict):
    type: str                   # "factual_contradiction" | "complementary_tension" | "stale_data"
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    resolution: str
    resolved_confidence: float


class DueDiligenceState(TypedDict):
    company: str
    financial_findings: Optional[AgentFindings]
    market_findings: Optional[AgentFindings]
    risk_findings: Optional[AgentFindings]
    sentiment_findings: Optional[AgentFindings]
    conflicts: Annotated[list[Conflict], add]   # reducer: concatenate across parallel writes
    final_report: Optional[dict]
    messages: Annotated[list, add]              # reducer: concatenate across parallel writes
