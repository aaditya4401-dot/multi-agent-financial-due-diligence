"""Working out what kind of company we are looking at.

The only probabilistic step in the planner. It is split from
:mod:`src.planning.plan` so that the routing rules stay pure, and it is
structured so the model is never the last word on a checkable fact:

* the model supplies **recall** — it knows that Block, Inc. trades as ``XYZ``
  and that Stripe does not trade at all;
* a market-data lookup supplies **verification** — it decides whether that
  ticker actually resolves.

A model asserting something a cheap deterministic check could settle is a
design smell, and the cost of being wrong here is asymmetric: misclassifying a
public company as private only costs a slightly worse toolset, but
misclassifying a private company as public sends the financial agent to look
for filings that do not exist.

Every failure path lands on :attr:`CompanyType.UNKNOWN`, which routes to the
unrestricted plan. Degrading to "do all the work" is always safe.
"""

import logging

from src.llm import Tier, get_llm
from src.planning.models import Classification, CompanyType
from src.tools.financial_tools import lookup_ticker_info

logger = logging.getLogger(__name__)

#: yfinance resolves plenty of things that are not companies. Only an equity
#: listing makes a due-diligence target "public" in the sense we mean.
_EQUITY_QUOTE_TYPES = {"EQUITY"}

SYSTEM_PROMPT = (
    "You identify companies for a due-diligence system.\n\n"
    "Given a company name, report:\n"
    "- `company_type`: 'public' if it has listed shares traded on an exchange, "
    "'private' if it is privately held, 'unknown' if you are not confident.\n"
    "- `ticker`: the primary exchange ticker if public, otherwise empty. Give "
    "the plain symbol only — no exchange prefix.\n"
    "- `aliases`: other names the company is known or listed under.\n"
    "- `sector`: a short sector description.\n"
    "- `rationale`: one sentence on how you identified it.\n\n"
    "Say 'unknown' rather than guessing. A wrong confident answer is worse "
    "than an admitted uncertainty, because downstream research is routed on it."
)


async def _propose(company: str) -> Classification:
    """Ask the FAST tier what this company is. Returns UNKNOWN on any failure."""
    try:
        model = get_llm(Tier.FAST).with_structured_output(Classification)
        result: Classification = await model.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Identify this company: {company}"},
        ])
        return result
    except Exception:
        logger.exception("Classification call failed for %r", company)
        return Classification(rationale="classification call failed")


async def _verify(candidate: str) -> str | None:
    """Return the resolved equity symbol for *candidate*, or ``None``.

    Resolving to a non-equity instrument (an ETF, a fund, a currency pair)
    counts as a failure to verify: it is not a company we can diligence.
    """
    info = await lookup_ticker_info(candidate)
    if info is None:
        return None
    if info.get("quoteType") not in _EQUITY_QUOTE_TYPES:
        logger.info(
            "%r resolved to quoteType=%r, not an equity — not treating as public",
            candidate, info.get("quoteType"),
        )
        return None
    return str(info.get("symbol") or candidate).upper()


async def classify_company(company: str) -> Classification:
    """Classify *company*, verifying any proposed ticker. Never raises."""
    proposal = await _propose(company)

    # The user may have typed a ticker straight in, so try that too. Ordered:
    # the model's guess first, the raw input as a fallback.
    candidates = [c for c in (proposal.ticker, company) if c and c.strip()]
    verified: str | None = None
    for candidate in candidates:
        verified = await _verify(candidate)
        if verified:
            break

    if verified:
        return proposal.model_copy(update={
            "company_type": CompanyType.PUBLIC,
            "ticker": verified,
        })

    # Nothing resolved. What that means depends on what was claimed.
    if proposal.company_type is CompanyType.PRIVATE:
        # Consistent: a private company has no ticker to resolve.
        return proposal.model_copy(update={"ticker": ""})

    if proposal.company_type is CompanyType.PUBLIC:
        # It claimed public but no symbol resolves. Do not upgrade that to
        # PRIVATE — absence of a ticker is not proof of privateness (foreign
        # listing, delisting, a flaky data provider). Report not-established.
        logger.info(
            "%r was proposed public but no ticker verified — downgrading to unknown",
            company,
        )
        return proposal.model_copy(update={
            "company_type": CompanyType.UNKNOWN,
            "ticker": "",
            "rationale": (
                f"{proposal.rationale} (Proposed public, but no ticker resolved.)"
            ).strip(),
        })

    return proposal.model_copy(update={"ticker": ""})
