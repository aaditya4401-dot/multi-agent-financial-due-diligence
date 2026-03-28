import json
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.state import DueDiligenceState
from src.tools.search_tools import search_web
from src.agents.utils import parse_react_output, error_findings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a market research analyst. Analyze TAM/SAM/SOM, market share, "
    "competitive landscape, key competitors, and competitive moat. "
    "Identify whether the company is gaining or losing market position."
)


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------

@tool
def tool_search_market_data(query: str) -> str:
    """Search for market size, TAM/SAM/SOM, and industry analysis data."""
    results = search_web(query, max_results=5, search_depth="advanced")
    return json.dumps(results, indent=2)


@tool
def tool_search_competitors(query: str) -> str:
    """Search for competitor analysis, market share data, and competitive landscape."""
    results = search_web(query, max_results=5)
    return json.dumps(results, indent=2)


TOOLS = [tool_search_market_data, tool_search_competitors]


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def market_agent(state: DueDiligenceState) -> dict:
    """LangGraph node: run the market research agent."""
    company = state["company"]

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        react_agent = create_react_agent(llm, TOOLS, state_modifier=SYSTEM_PROMPT)

        result = await react_agent.ainvoke({
            "messages": [f"Research the market position of {company}. "
                         "Analyze total addressable market, current market share, "
                         "key competitors, and the company's competitive moat. "
                         "Determine if they are gaining or losing position."]
        })

        findings = parse_react_output("market", result)

    except Exception as exc:
        logger.error("Market agent failed for %r: %s", company, exc)
        findings = error_findings("market", exc)

    return {"market_findings": findings}
