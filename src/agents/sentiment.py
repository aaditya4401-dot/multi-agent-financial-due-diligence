import json

from langchain_core.tools import tool

from src.agents.spec import AgentSpec
from src.tools.news_tools import search_news, search_reddit_sentiment
from src.tools.search_tools import asearch_web

SYSTEM_PROMPT = (
    "You are a sentiment analyst. Analyze news sentiment trajectory "
    "(improving or declining over time), developer/user sentiment from "
    "Reddit and forums, and employee sentiment from Glassdoor and layoff "
    "news. Focus on trends, not just current state."
)

TASK_PROMPT = (
    "Analyze public sentiment around {company}. Cover three areas: "
    "(1) news sentiment trajectory — is coverage improving or declining? "
    "(2) developer/user sentiment on Reddit and forums, "
    "(3) employee sentiment from Glassdoor and layoff news. "
    "Focus on trends over time, not just a snapshot."
)


@tool
async def tool_search_news(company: str) -> str:
    """Search for recent news articles about a company to assess news sentiment trajectory."""
    return json.dumps(await search_news(company, max_results=5), indent=2)


@tool
async def tool_search_reddit(company: str) -> str:
    """Search Reddit for developer and user discussions about a company."""
    return json.dumps(await search_reddit_sentiment(company, max_results=5), indent=2)


@tool
async def tool_search_sentiment(query: str) -> str:
    """General web search for employee reviews, Glassdoor ratings, layoff news, and public opinion."""
    return json.dumps(await asearch_web(query, max_results=5), indent=2)


TOOLS = [tool_search_news, tool_search_reddit, tool_search_sentiment]

SPEC = AgentSpec(
    name="sentiment",
    system_prompt=SYSTEM_PROMPT,
    task_prompt=TASK_PROMPT,
    tools=tuple(TOOLS),
)
