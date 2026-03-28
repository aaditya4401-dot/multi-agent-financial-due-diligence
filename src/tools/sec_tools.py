import asyncio
import logging
import re
from typing import TypedDict

from src.tools.search_tools import search_web

logger = logging.getLogger(__name__)

_FILING_TYPE_RE = re.compile(
    r"\b(10-K|10-Q|8-K|S-1|DEF 14A|13F|20-F|6-K|SC 13D|SC 13G|proxy statement)\b",
    re.IGNORECASE,
)


class SECFiling(TypedDict):
    filing_type: str
    date: str
    url: str
    summary: str


def _guess_filing_type(text: str) -> str:
    """Extract a known SEC filing type from text, or return empty string."""
    match = _FILING_TYPE_RE.search(text)
    return match.group(1).upper() if match else ""


async def search_sec_filings(company: str, max_results: int = 5) -> list[SECFiling]:
    """Search for SEC EDGAR filings and regulatory actions for *company*.

    Uses Tavily scoped to sec.gov for official filings, plus a broader
    search for regulatory actions and enforcement news.

    Returns a list[SECFiling] — empty list on failure.
    """
    edgar_results, regulatory_results = await asyncio.gather(
        asyncio.to_thread(
            search_web,
            f"{company} SEC EDGAR filing 10-K 10-Q 8-K",
            max_results,
            search_depth="advanced",
            include_domains=["sec.gov"],
        ),
        asyncio.to_thread(
            search_web,
            f"{company} SEC regulatory action enforcement fine",
            max_results,
            search_depth="advanced",
        ),
    )

    filings: list[SECFiling] = []
    seen_urls: set[str] = set()

    for r in edgar_results + regulatory_results:
        if not r["content"] or r["url"] in seen_urls:
            continue
        seen_urls.add(r["url"])

        combined_text = f"{r['title']} {r['content']}"
        filings.append(SECFiling(
            filing_type=_guess_filing_type(combined_text),
            date=r["date"],
            url=r["url"],
            summary=r["content"][:500],
        ))

    return filings
