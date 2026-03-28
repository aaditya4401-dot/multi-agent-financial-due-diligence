import json
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.state import DueDiligenceState
from src.tools.financial_tools import get_company_financials, get_funding_rounds
from src.tools.search_tools import search_web
from src.agents.utils import parse_react_output, error_findings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior financial analyst performing due diligence. "
    "Analyze revenue, ARR, growth rate, gross margin, cash flow, burn rate, "
    "funding history, and valuation trends. Be specific with numbers. "
    "Flag anything concerning."
)


# ---------------------------------------------------------------------------
# LangChain tool wrappers (create_react_agent needs @tool-decorated fns)
# ---------------------------------------------------------------------------

@tool
async def tool_get_company_financials(company: str) -> str:
    """Fetch core financial metrics (revenue, margin, cash flow, market cap, growth) for a company."""
    findings = await get_company_financials(company)
    return json.dumps(findings, indent=2)


@tool
async def tool_get_funding_rounds(company: str) -> str:
    """Search for funding rounds, valuation history, and investors for a company."""
    findings = await get_funding_rounds(company)
    return json.dumps(findings, indent=2)


@tool
def tool_search_web(query: str) -> str:
    """General web search. Use for any financial data not covered by the other tools."""
    results = search_web(query, max_results=5)
    return json.dumps(results, indent=2)


TOOLS = [tool_get_company_financials, tool_get_funding_rounds, tool_search_web]


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def financial_agent(state: DueDiligenceState) -> dict:
    """LangGraph node: run the financial analysis agent."""
    company = state["company"]

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        react_agent = create_react_agent(llm, TOOLS, state_modifier=SYSTEM_PROMPT)

        result = await react_agent.ainvoke({
            "messages": [f"Analyze the financials of {company}. "
                         "Use all available tools to gather data, then provide "
                         "a comprehensive financial assessment with specific numbers."]
        })

        findings = parse_react_output("financial", result)

    except Exception as exc:
        logger.error("Financial agent failed for %r: %s", company, exc)
        findings = error_findings("financial", exc)

    return {"financial_findings": findings}
