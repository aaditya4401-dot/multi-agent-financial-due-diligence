import json
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.state import DueDiligenceState
from src.tools.news_tools import search_news, search_reddit_sentiment
from src.tools.search_tools import search_web
from src.agents.utils import parse_react_output, error_findings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a sentiment analyst. Analyze news sentiment trajectory "
    "(improving or declining over time), developer/user sentiment from "
    "Reddit and forums, and employee sentiment from Glassdoor and layoff "
    "news. Focus on trends, not just current state."
)


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------

@tool
async def tool_search_news(company: str) -> str:
    """Search for recent news articles about a company to assess news sentiment trajectory."""
    articles = await search_news(company, max_results=5)
    return json.dumps(articles, indent=2)


@tool
async def tool_search_reddit(company: str) -> str:
    """Search Reddit for developer and user discussions about a company."""
    posts = await search_reddit_sentiment(company, max_results=5)
    return json.dumps(posts, indent=2)


@tool
def tool_search_sentiment(query: str) -> str:
    """General web search for employee reviews, Glassdoor ratings, layoff news, and public opinion."""
    results = search_web(query, max_results=5)
    return json.dumps(results, indent=2)


TOOLS = [tool_search_news, tool_search_reddit, tool_search_sentiment]


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def sentiment_agent(state: DueDiligenceState) -> dict:
    """LangGraph node: run the sentiment analysis agent."""
    company = state["company"]

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        react_agent = create_react_agent(llm, TOOLS, state_modifier=SYSTEM_PROMPT)

        result = await react_agent.ainvoke({
            "messages": [f"Analyze public sentiment around {company}. "
                         "Cover three areas: (1) news sentiment trajectory — is coverage "
                         "improving or declining? (2) developer/user sentiment on Reddit "
                         "and forums, (3) employee sentiment from Glassdoor and layoff news. "
                         "Focus on trends over time, not just a snapshot."]
        })

        findings = parse_react_output("sentiment", result)

    except Exception as exc:
        logger.error("Sentiment agent failed for %r: %s", company, exc)
        findings = error_findings("sentiment", exc)

    return {"sentiment_findings": findings}
