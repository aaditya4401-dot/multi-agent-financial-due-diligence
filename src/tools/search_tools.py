import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from tavily import TavilyClient, MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError

load_dotenv()

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    date: str


def search_web(query: str, max_results: int = 5, **kwargs) -> list[SearchResult]:
    """Search the web via Tavily and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).
        **kwargs: Extra parameters forwarded to TavilyClient.search()
                  (e.g. search_depth, topic, include_domains).

    Returns:
        List of SearchResult dicts. Returns an empty list on any error
        so callers can degrade gracefully without crashing.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — returning empty results")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            **kwargs,
        )
        return [_parse_result(r) for r in response.get("results", [])]

    except MissingAPIKeyError:
        logger.error("Tavily API key is missing")
    except InvalidAPIKeyError:
        logger.error("Tavily API key is invalid")
    except UsageLimitExceededError:
        logger.warning("Tavily usage limit exceeded")
    except Exception as exc:
        logger.error("Tavily search failed for query %r: %s", query, exc)

    return []


def _parse_result(raw: dict) -> SearchResult:
    return SearchResult(
        title=raw.get("title") or "",
        url=raw.get("url") or "",
        content=raw.get("content") or "",
        date=raw.get("published_date") or "",
    )
