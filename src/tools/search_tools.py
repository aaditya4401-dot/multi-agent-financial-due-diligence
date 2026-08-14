"""Shared Tavily search layer.

All agents search overlapping ground for the same company, so this module
does three things beyond a plain API call:

1. **One client per event loop** — no per-call ``TavilyClient`` construction,
   so connections are pooled across the whole run.
2. **TTL cache** — repeat queries (within and across runs) are free.
3. **In-flight coalescing** — when four agents fire the same query
   concurrently, one request goes out and the rest ride along.

Every entry point degrades to an empty list on failure rather than raising,
so a dead search provider never takes the pipeline down.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, TypedDict

from dotenv import load_dotenv
from tavily import (
    AsyncTavilyClient,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL", "3600"))
MAX_CACHE_ENTRIES = 512


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    date: str


# ---------------------------------------------------------------------------
# Client pool (one per event loop — httpx clients are loop-bound)
# ---------------------------------------------------------------------------

_clients: dict[asyncio.AbstractEventLoop, AsyncTavilyClient] = {}


def _get_client() -> AsyncTavilyClient | None:
    """Return the client for the running loop, creating it on first use."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    loop = asyncio.get_running_loop()

    # Drop clients belonging to loops that have since closed.
    for dead in [lp for lp in _clients if lp.is_closed()]:
        _clients.pop(dead, None)

    client = _clients.get(loop)
    if client is None:
        client = AsyncTavilyClient(api_key=api_key)
        _clients[loop] = client
    return client


# ---------------------------------------------------------------------------
# Cache + coalescing
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, list[SearchResult]]] = {}
_inflight: dict[str, asyncio.Future] = {}


def _cache_key(query: str, max_results: int, kwargs: dict[str, Any]) -> str:
    payload = json.dumps(
        {"q": " ".join(query.lower().split()), "n": max_results, "kw": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _prune_cache() -> None:
    """Evict expired entries, then the oldest if still over capacity."""
    now = time.monotonic()
    for key in [k for k, (ts, _) in _cache.items() if now - ts >= CACHE_TTL_SECONDS]:
        _cache.pop(key, None)

    while len(_cache) > MAX_CACHE_ENTRIES:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)


def clear_cache() -> None:
    """Drop all cached results. Exposed for tests and manual refreshes."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def asearch_web(query: str, max_results: int = 5, **kwargs: Any) -> list[SearchResult]:
    """Search the web via Tavily, with caching and request coalescing.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        **kwargs: Extra parameters forwarded to the Tavily client
                  (e.g. ``search_depth``, ``topic``, ``include_domains``).

    Returns:
        List of SearchResult dicts — empty on any failure.
    """
    key = _cache_key(query, max_results, kwargs)

    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
        logger.debug("search cache hit: %s", query)
        return cached[1]

    loop = asyncio.get_running_loop()
    pending = _inflight.get(key)
    if pending is not None and pending.get_loop() is loop and not pending.done():
        logger.debug("joining in-flight search: %s", query)
        return await pending

    future = loop.create_future()
    _inflight[key] = future
    results: list[SearchResult] = []
    try:
        results = await _search_uncached(query, max_results, **kwargs)
    except Exception as exc:  # defensive — _search_uncached is non-raising
        logger.error("Unexpected search failure for %r: %s", query, exc)
    finally:
        # Always resolve the future so coalesced waiters can never hang,
        # even if this task is cancelled mid-flight.
        _inflight.pop(key, None)
        if not future.done():
            future.set_result(results)

    _cache[key] = (time.monotonic(), results)
    _prune_cache()
    return results


async def _search_uncached(query: str, max_results: int, **kwargs: Any) -> list[SearchResult]:
    """Perform the actual Tavily call. Never raises — returns [] on failure."""
    client = _get_client()
    if client is None:
        logger.warning("TAVILY_API_KEY not set — returning empty results")
        return []

    try:
        response = await client.search(
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


def search_web(query: str, max_results: int = 5, **kwargs: Any) -> list[SearchResult]:
    """Synchronous wrapper around :func:`asearch_web`.

    Provided for sync callers (tests, scripts). Inside async code call
    :func:`asearch_web` directly — this raises rather than deadlocking if a
    loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asearch_web(query, max_results, **kwargs))

    raise RuntimeError(
        "search_web() cannot be called from a running event loop — use asearch_web()"
    )


def _parse_result(raw: dict) -> SearchResult:
    return SearchResult(
        title=raw.get("title") or "",
        url=raw.get("url") or "",
        content=raw.get("content") or "",
        date=raw.get("published_date") or "",
    )
