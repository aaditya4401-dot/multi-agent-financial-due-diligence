import json
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.state import DueDiligenceState
from src.tools.sec_tools import search_sec_filings
from src.tools.search_tools import search_web
from src.agents.utils import parse_react_output, error_findings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a risk analyst specializing in corporate due diligence. "
    "Search for regulatory risks, active litigation, compliance violations, "
    "SEC investigations, key person dependencies, customer concentration, "
    "and any red flags. Be thorough — missed risks are costly."
)


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------

@tool
async def tool_search_sec_filings(company: str) -> str:
    """Search SEC EDGAR for filings, regulatory actions, and enforcement history."""
    filings = await search_sec_filings(company)
    return json.dumps(filings, indent=2)


@tool
def tool_search_risk_info(query: str) -> str:
    """Search the web for lawsuits, regulatory actions, compliance issues, and risk factors."""
    results = search_web(query, max_results=5, search_depth="advanced")
    return json.dumps(results, indent=2)


TOOLS = [tool_search_sec_filings, tool_search_risk_info]


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def risk_agent(state: DueDiligenceState) -> dict:
    """LangGraph node: run the risk assessment agent."""
    company = state["company"]

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        react_agent = create_react_agent(llm, TOOLS, state_modifier=SYSTEM_PROMPT)

        result = await react_agent.ainvoke({
            "messages": [f"Conduct a thorough risk assessment of {company}. "
                         "Search for regulatory risks, litigation history, "
                         "SEC filings, compliance issues, key person risk, "
                         "and customer concentration. Flag every red flag you find."]
        })

        findings = parse_react_output("risk", result)

    except Exception as exc:
        logger.error("Risk agent failed for %r: %s", company, exc)
        findings = error_findings("risk", exc)

    return {"risk_findings": findings}
