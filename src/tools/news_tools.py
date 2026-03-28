import asyncio
import logging
import re
from typing import TypedDict

from src.tools.search_tools import search_web

logger = logging.getLogger(__name__)


class NewsArticle(TypedDict):
    title: str
    url: str
    content: str
    published_date: str


class RedditPost(TypedDict):
    title: str
    url: str
    content: str
    subreddit: str


# ---------------------------------------------------------------------------
# 1. News search (Tavily advanced + news topic)
# ---------------------------------------------------------------------------

async def search_news(company: str, max_results: int = 5) -> list[NewsArticle]:
    """Search for recent news articles about *company*.

    Uses Tavily with search_depth="advanced" and topic="news" for
    higher-quality, time-sorted results.

    Returns a list[NewsArticle] — empty list on failure.
    """
    try:
        results = await asyncio.to_thread(
            search_web,
            f"{company} company latest news",
            max_results,
            search_depth="advanced",
            topic="news",
        )
    except Exception as exc:
        logger.error("News search failed for %r: %s", company, exc)
        return []

    return [
        NewsArticle(
            title=r["title"],
            url=r["url"],
            content=r["content"],
            published_date=r["date"],
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# 2. Reddit sentiment search (Tavily scoped to reddit.com)
# ---------------------------------------------------------------------------

_SUBREDDIT_RE = re.compile(r"reddit\.com/r/([^/]+)")


def _extract_subreddit(url: str) -> str:
    """Pull the subreddit name from a reddit URL, or return empty string."""
    match = _SUBREDDIT_RE.search(url)
    return match.group(1) if match else ""


async def search_reddit_sentiment(company: str, max_results: int = 5) -> list[RedditPost]:
    """Search Reddit for discussions about *company*.

    Scopes Tavily to reddit.com via include_domains so no separate
    Reddit API key is needed.

    Returns a list[RedditPost] — empty list on failure.
    """
    try:
        results = await asyncio.to_thread(
            search_web,
            f"{company} company review discussion opinion",
            max_results,
            search_depth="advanced",
            include_domains=["reddit.com"],
        )
    except Exception as exc:
        logger.error("Reddit search failed for %r: %s", company, exc)
        return []

    return [
        RedditPost(
            title=r["title"],
            url=r["url"],
            content=r["content"],
            subreddit=_extract_subreddit(r["url"]),
        )
        for r in results
    ]
