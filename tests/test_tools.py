"""Smoke tests for all tool modules.

Each test verifies the function returns the expected type without crashing.
Tavily-based tools will return empty lists if TAVILY_API_KEY is not set —
that's the intended graceful-degradation behaviour and the tests still pass.
"""

import asyncio
import os
import pytest

from src.tools.search_tools import search_web, SearchResult
from src.tools.financial_tools import get_company_financials, get_funding_rounds
from src.tools.news_tools import search_news, search_reddit_sentiment, NewsArticle, RedditPost
from src.tools.sec_tools import search_sec_filings, SECFiling

HAS_TAVILY_KEY = bool(os.getenv("TAVILY_API_KEY"))


# ---------------------------------------------------------------------------
# 1. search_tools
# ---------------------------------------------------------------------------

class TestSearchTools:
    def test_search_web_returns_list(self):
        results = search_web("Stripe payments company", max_results=3)
        assert isinstance(results, list)
        if HAS_TAVILY_KEY:
            assert len(results) > 0
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "content" in r
            assert "date" in r

    def test_search_web_bad_key_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "")
        results = search_web("anything")
        assert results == []


# ---------------------------------------------------------------------------
# 2. financial_tools
# ---------------------------------------------------------------------------

class TestFinancialTools:
    def test_get_company_financials_public(self):
        """yfinance should return data for a public ticker like AAPL."""
        findings = asyncio.run(get_company_financials("AAPL"))
        assert isinstance(findings, list)
        assert len(findings) > 0
        for f in findings:
            assert "claim" in f
            assert "source" in f
            assert "confidence" in f
            assert "source_quality" in f
            assert "date_of_data" in f

    def test_get_company_financials_private_no_crash(self):
        """Private company: yfinance fails, falls back to web search or returns empty."""
        findings = asyncio.run(get_company_financials("Stripe"))
        assert isinstance(findings, list)
        # Should always have at least one finding (either web data or the "no data" placeholder)
        assert len(findings) >= 1

    def test_get_funding_rounds(self):
        findings = asyncio.run(get_funding_rounds("Stripe"))
        assert isinstance(findings, list)
        assert len(findings) >= 1
        for f in findings:
            assert "claim" in f
            assert "source" in f


# ---------------------------------------------------------------------------
# 3. news_tools
# ---------------------------------------------------------------------------

class TestNewsTools:
    def test_search_news_returns_list(self):
        articles = asyncio.run(search_news("Stripe", max_results=3))
        assert isinstance(articles, list)
        for a in articles:
            assert "title" in a
            assert "url" in a
            assert "content" in a
            assert "published_date" in a

    def test_search_reddit_sentiment_returns_list(self):
        posts = asyncio.run(search_reddit_sentiment("Stripe", max_results=3))
        assert isinstance(posts, list)
        for p in posts:
            assert "title" in p
            assert "url" in p
            assert "content" in p
            assert "subreddit" in p


# ---------------------------------------------------------------------------
# 4. sec_tools
# ---------------------------------------------------------------------------

class TestSECTools:
    def test_search_sec_filings_returns_list(self):
        filings = asyncio.run(search_sec_filings("Stripe", max_results=3))
        assert isinstance(filings, list)
        for f in filings:
            assert "filing_type" in f
            assert "date" in f
            assert "url" in f
            assert "summary" in f
