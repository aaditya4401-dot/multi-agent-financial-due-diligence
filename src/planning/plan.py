"""The routing rules — what to research, given what the company is.

This module is deliberately pure: no LLM, no network, no I/O. Given a
:class:`Classification` it returns a :class:`ResearchPlan`, and that is all.
Everything probabilistic lives in :mod:`src.planning.classify`.

That split is the point of the phase. The orchestrator's *perception* (which
company is this?) is a model's guess, verified where possible. Its *judgment*
(therefore, research these things with these tools) is ordinary code, and
ordinary code can be unit tested in milliseconds. An orchestrator whose routing
decisions can only be checked by paying for four agent runs is not one you can
reason about.

Tools are referenced by name. The names are cross-checked against the real
agent registry by a test, so a typo here fails the suite rather than silently
dropping a tool at runtime.
"""

from src.planning.models import Classification, CompanyType, ResearchPlan, ResearchTask

#: Every agent the system can dispatch to. Order is the report's section order.
ALL_AGENTS: tuple[str, ...] = ("financial", "market", "risk", "sentiment")

# --- Tool names, mirroring the @tool function names in src/agents/* ---------
FINANCIALS = "tool_get_company_financials"
FUNDING_ROUNDS = "tool_get_funding_rounds"
SEARCH_WEB = "tool_search_web"

MARKET_DATA = "tool_search_market_data"
COMPETITORS = "tool_search_competitors"

SEC_FILINGS = "tool_search_sec_filings"
RISK_INFO = "tool_search_risk_info"

NEWS = "tool_search_news"
REDDIT = "tool_search_reddit"
SENTIMENT_WEB = "tool_search_sentiment"

#: Said to the synthesizer, not to the scorer. The confidence model already
#: discounts a news-tier source against a filing; what it cannot express is
#: *why* the whole section is softer than it looks.
PRIVATE_FINANCIAL_CAVEAT = (
    "No audited public financials exist for this company. Every figure in this "
    "section is a third-party estimate or a disclosed headline number, and "
    "should be read as such."
)


# ---------------------------------------------------------------------------
# Per-agent task builders
# ---------------------------------------------------------------------------

def _financial_task(company_type: CompanyType) -> ResearchTask:
    """The task that actually differs by company type.

    Public companies file. Private ones do not, so the tool that reads filings
    is not merely unhelpful for them — it is a guaranteed miss that still costs
    a lookup before falling through to web search.
    """
    if company_type is CompanyType.PUBLIC:
        return ResearchTask(
            agent="financial",
            tools=[FINANCIALS, SEARCH_WEB],
            focus=(
                "This is a publicly listed company, so prefer figures traceable "
                "to its most recent 10-K or 10-Q over anything a secondary "
                "source reports. Cover revenue and YoY growth, gross and "
                "operating margin, operating cash flow, and the direction of "
                "the margin trend. Note the fiscal period every figure belongs to."
            ),
        )

    if company_type is CompanyType.PRIVATE:
        return ResearchTask(
            agent="financial",
            # No FINANCIALS: that path is a market-data lookup, and a private
            # company has no ticker behind it to look up.
            tools=[FUNDING_ROUNDS, SEARCH_WEB],
            focus=(
                "This company is privately held and files no public accounts. "
                "Work from funding rounds, disclosed valuation marks, headline "
                "metrics the company has published itself, and credible "
                "reporting. Attribute every number to whoever estimated it and "
                "say when it was estimated. Never present an estimate as a "
                "reported result."
            ),
            caveat=PRIVATE_FINANCIAL_CAVEAT,
        )

    return ResearchTask(
        agent="financial",
        tools=[FINANCIALS, FUNDING_ROUNDS, SEARCH_WEB],
        focus=(
            "It is not established whether this company is public or private. "
            "Try both filed financials and funding history, and state which "
            "kind of source each figure came from."
        ),
    )


def _market_task(company_type: CompanyType) -> ResearchTask:
    """Market research does not depend on listing status, so this is constant."""
    return ResearchTask(agent="market", tools=[MARKET_DATA, COMPETITORS])


def _risk_task(company_type: CompanyType) -> ResearchTask:
    """Risk keeps its SEC tool in every case.

    Private companies still appear in enforcement actions, litigation dockets
    and S-1 registrations, so dropping the filing search for them would lose
    real signal. Only the emphasis changes.
    """
    if company_type is CompanyType.PUBLIC:
        focus = (
            "Start from the risk factors and legal proceedings disclosed in the "
            "company's own filings, then look for anything material that the "
            "filings omit."
        )
    elif company_type is CompanyType.PRIVATE:
        focus = (
            "This company files no annual report, so there is no risk-factors "
            "section to lean on. Look for regulatory enforcement, litigation, "
            "S-1 or other registration filings, and reporting on governance or "
            "key-person concentration."
        )
    else:
        focus = ""
    return ResearchTask(agent="risk", tools=[SEC_FILINGS, RISK_INFO], focus=focus)


def _sentiment_task(company_type: CompanyType) -> ResearchTask:
    """Sentiment is listing-agnostic: people talk about both kinds of company."""
    return ResearchTask(agent="sentiment", tools=[NEWS, REDDIT, SENTIMENT_WEB])


_BUILDERS = {
    "financial": _financial_task,
    "market": _market_task,
    "risk": _risk_task,
    "sentiment": _sentiment_task,
}


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------

def _rationale(company: str, classification: Classification) -> str:
    """Plain-language account of why this plan looks the way it does.

    Written for a human reading an approval gate, so it says what changed and
    what it costs — not just what the classifier decided.
    """
    kind = classification.company_type
    if kind is CompanyType.PUBLIC:
        return (
            f"{company} resolved to the verified ticker "
            f"{classification.ticker or '?'}, so financial research works from "
            f"filed statements and the funding-history tool is dropped as noise."
        )
    if kind is CompanyType.PRIVATE:
        return (
            f"{company} appears to be privately held, so the financial agent "
            f"drops its market-data tool — there is no ticker for it to resolve "
            f"— and works from funding and valuation reporting instead. Its "
            f"figures will be third-party estimates, not reported results."
        )
    return (
        f"Could not establish whether {company} is public or private, so every "
        f"agent runs with its full toolset. Nothing is skipped on a guess."
    )


def plan_for(company: str, classification: Classification | None = None) -> ResearchPlan:
    """Build the research plan for *company*.

    With no classification — or an unknown one — this returns the unrestricted
    four-agent plan, which is exactly the behaviour the system had before it
    could classify anything. Degrading to "do all the work" is always safe;
    degrading to "skip an agent" on a guess is not.
    """
    classification = classification or Classification()
    kind = classification.company_type

    tasks = [_BUILDERS[agent](kind) for agent in ALL_AGENTS]

    return ResearchPlan(
        company=company,
        company_type=kind,
        ticker=classification.ticker or None,
        rationale=_rationale(company, classification),
        tasks=tasks,
    )
