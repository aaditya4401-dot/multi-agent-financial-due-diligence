import json

from langchain_core.tools import tool

from src.agents.base import make_research_agent
from src.tools.financial_tools import get_company_financials, get_funding_rounds
from src.tools.search_tools import asearch_web

SYSTEM_PROMPT = (
    "You are a senior financial analyst performing due diligence. "
    "Analyze revenue, ARR, growth rate, gross margin, cash flow, burn rate, "
    "funding history, and valuation trends. Be specific with numbers. "
    "Flag anything concerning."
)

TASK_PROMPT = (
    "Analyze the financials of {company}. Use all available tools to gather "
    "data, then provide a comprehensive financial assessment with specific numbers."
)


@tool
async def tool_get_company_financials(company: str) -> str:
    """Fetch core financial metrics (revenue, margin, cash flow, market cap, growth) for a company."""
    return json.dumps(await get_company_financials(company), indent=2)


@tool
async def tool_get_funding_rounds(company: str) -> str:
    """Search for funding rounds, valuation history, and investors for a company."""
    return json.dumps(await get_funding_rounds(company), indent=2)


@tool
async def tool_search_web(query: str) -> str:
    """General web search. Use for any financial data not covered by the other tools."""
    return json.dumps(await asearch_web(query, max_results=5), indent=2)


TOOLS = [tool_get_company_financials, tool_get_funding_rounds, tool_search_web]

financial_agent = make_research_agent(
    name="financial",
    state_key="financial_findings",
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    task_prompt=TASK_PROMPT,
)
