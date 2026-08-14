import json

from langchain_core.tools import tool

from src.agents.base import make_research_agent
from src.tools.search_tools import asearch_web

SYSTEM_PROMPT = (
    "You are a market research analyst. Analyze TAM/SAM/SOM, market share, "
    "competitive landscape, key competitors, and competitive moat. "
    "Identify whether the company is gaining or losing market position."
)

TASK_PROMPT = (
    "Research the market position of {company}. Analyze total addressable "
    "market, current market share, key competitors, and the company's "
    "competitive moat. Determine if they are gaining or losing position."
)


@tool
async def tool_search_market_data(query: str) -> str:
    """Search for market size, TAM/SAM/SOM, and industry analysis data."""
    results = await asearch_web(query, max_results=5, search_depth="advanced")
    return json.dumps(results, indent=2)


@tool
async def tool_search_competitors(query: str) -> str:
    """Search for competitor analysis, market share data, and competitive landscape."""
    return json.dumps(await asearch_web(query, max_results=5), indent=2)


TOOLS = [tool_search_market_data, tool_search_competitors]

market_agent = make_research_agent(
    name="market",
    state_key="market_findings",
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    task_prompt=TASK_PROMPT,
)
