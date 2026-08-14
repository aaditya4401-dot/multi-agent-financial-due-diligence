import asyncio
import logging
from datetime import datetime

import yfinance as yf

from src.state import Finding
from src.tools.search_tools import asearch_web

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _pct(current: float, previous: float) -> float | None:
    """Return YoY percentage change, or None if inputs are unusable."""
    if previous and previous != 0:
        return round((current - previous) / abs(previous) * 100, 2)
    return None


# ---------------------------------------------------------------------------
# 1. Company financials (yfinance → Tavily fallback)
# ---------------------------------------------------------------------------

async def get_company_financials(company: str) -> list[Finding]:
    """Fetch core financial metrics for *company*.

    Tries yfinance first (works for public tickers / company names).
    If yfinance returns no data (private company, bad ticker, etc.),
    falls back to a Tavily web search for financial information.

    Always returns a list[Finding] — empty on total failure.
    """
    findings = await _financials_from_yfinance(company)
    if findings:
        return findings

    logger.info("yfinance returned no data for %r — falling back to web search", company)
    return await _financials_from_web(company)


async def _financials_from_yfinance(company: str) -> list[Finding]:
    """Pull structured financials from yfinance (runs sync IO in a thread)."""
    try:
        ticker = await asyncio.to_thread(yf.Ticker, company)
        info: dict = await asyncio.to_thread(lambda: ticker.info)
    except Exception as exc:
        logger.error("yfinance Ticker lookup failed for %r: %s", company, exc)
        return []

    if not info or info.get("quoteType") is None:
        return []

    findings: list[Finding] = []
    source = f"Yahoo Finance ({info.get('shortName', company)})"
    today = _today()

    # Revenue
    revenue = info.get("totalRevenue")
    if revenue is not None:
        findings.append(Finding(
            claim=f"Total revenue: ${revenue:,.0f}",
            source=source,
            confidence=0.85,
            source_quality="official_filing",
            date_of_data=today,
        ))

    # Gross margin
    gross_margin = info.get("grossMargins")
    if gross_margin is not None:
        findings.append(Finding(
            claim=f"Gross margin: {gross_margin * 100:.1f}%",
            source=source,
            confidence=0.85,
            source_quality="official_filing",
            date_of_data=today,
        ))

    # Operating cash flow
    ocf = info.get("operatingCashflow")
    if ocf is not None:
        findings.append(Finding(
            claim=f"Operating cash flow: ${ocf:,.0f}",
            source=source,
            confidence=0.85,
            source_quality="official_filing",
            date_of_data=today,
        ))

    # Market cap
    market_cap = info.get("marketCap")
    if market_cap is not None:
        findings.append(Finding(
            claim=f"Market cap: ${market_cap:,.0f}",
            source=source,
            confidence=0.90,
            source_quality="official_filing",
            date_of_data=today,
        ))

    # YoY revenue growth
    revenue_growth = info.get("revenueGrowth")
    if revenue_growth is not None:
        findings.append(Finding(
            claim=f"YoY revenue growth: {revenue_growth * 100:.1f}%",
            source=source,
            confidence=0.80,
            source_quality="official_filing",
            date_of_data=today,
        ))
    else:
        # Try computing from annual financials
        yoy = await _compute_yoy_growth(ticker)
        if yoy is not None:
            findings.append(Finding(
                claim=f"YoY revenue growth (computed): {yoy:.1f}%",
                source=source,
                confidence=0.70,
                source_quality="official_filing",
                date_of_data=today,
            ))

    return findings


async def _compute_yoy_growth(ticker: yf.Ticker) -> float | None:
    """Compute YoY revenue growth from the annual income statement."""
    try:
        financials = await asyncio.to_thread(lambda: ticker.financials)
        if financials is None or financials.empty:
            return None
        if "Total Revenue" not in financials.index:
            return None
        revenues = financials.loc["Total Revenue"].dropna().sort_index()
        if len(revenues) < 2:
            return None
        return _pct(revenues.iloc[-1], revenues.iloc[-2])
    except Exception as exc:
        logger.warning("Could not compute YoY growth: %s", exc)
        return None


async def _financials_from_web(company: str) -> list[Finding]:
    """Fall back to Tavily web search for financial data (private companies)."""
    results = await asearch_web(f"{company} company revenue valuation financials", 5)

    findings: list[Finding] = []
    for r in results:
        if not r["content"]:
            continue
        findings.append(Finding(
            claim=r["content"][:500],
            source=r["url"] or r["title"],
            confidence=0.50,
            source_quality="news_article",
            date_of_data=r["date"] or _today(),
        ))

    if not findings:
        findings.append(Finding(
            claim=f"No financial data found for {company} via yfinance or web search",
            source="system",
            confidence=0.0,
            source_quality="news_article",
            date_of_data=_today(),
        ))

    return findings


# ---------------------------------------------------------------------------
# 2. Funding rounds (Tavily web search)
# ---------------------------------------------------------------------------

async def get_funding_rounds(company: str) -> list[Finding]:
    """Search the web for funding history, valuation, and investors.

    Returns a list[Finding] — empty-with-note on total failure.
    """
    results = await asearch_web(f"{company} funding rounds valuation investors Series", 5)

    findings: list[Finding] = []
    for r in results:
        if not r["content"]:
            continue
        findings.append(Finding(
            claim=r["content"][:500],
            source=r["url"] or r["title"],
            confidence=0.55,
            source_quality="news_article",
            date_of_data=r["date"] or _today(),
        ))

    if not findings:
        findings.append(Finding(
            claim=f"No funding round data found for {company}",
            source="system",
            confidence=0.0,
            source_quality="news_article",
            date_of_data=_today(),
        ))

    return findings
