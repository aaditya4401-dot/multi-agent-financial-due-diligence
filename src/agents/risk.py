import json

from langchain_core.tools import tool

from src.agents.base import make_research_agent
from src.tools.search_tools import asearch_web
from src.tools.sec_tools import search_sec_filings

SYSTEM_PROMPT = (
    "You are a risk analyst specializing in corporate due diligence. "
    "Search for regulatory risks, active litigation, compliance violations, "
    "SEC investigations, key person dependencies, customer concentration, "
    "and any red flags. Be thorough — missed risks are costly."
)

TASK_PROMPT = (
    "Conduct a thorough risk assessment of {company}. Search for regulatory "
    "risks, litigation history, SEC filings, compliance issues, key person "
    "risk, and customer concentration. Flag every red flag you find."
)


@tool
async def tool_search_sec_filings(company: str) -> str:
    """Search SEC EDGAR for filings, regulatory actions, and enforcement history."""
    return json.dumps(await search_sec_filings(company), indent=2)


@tool
async def tool_search_risk_info(query: str) -> str:
    """Search the web for lawsuits, regulatory actions, compliance issues, and risk factors."""
    results = await asearch_web(query, max_results=5, search_depth="advanced")
    return json.dumps(results, indent=2)


TOOLS = [tool_search_sec_filings, tool_search_risk_info]

risk_agent = make_research_agent(
    name="risk",
    state_key="risk_findings",
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    task_prompt=TASK_PROMPT,
)
