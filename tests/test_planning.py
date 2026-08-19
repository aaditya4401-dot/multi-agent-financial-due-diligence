"""Tests for the planner: routing rules, and classification's trust boundary.

All offline — no API keys and no network required. That is the point of the
phase: the orchestrator's judgment is ordinary code, so checking it costs
milliseconds rather than four agent runs.
"""

import asyncio

import pytest

from src.planning import classify as cls
from src.planning.models import Classification, CompanyType, ResearchPlan
from src.planning.plan import (
    ALL_AGENTS,
    FINANCIALS,
    FUNDING_ROUNDS,
    SEARCH_WEB,
    plan_for,
)


def _plan(kind: CompanyType, ticker: str = "") -> ResearchPlan:
    return plan_for("Acme", Classification(company_type=kind, ticker=ticker))


def _fin_tools(plan: ResearchPlan) -> list[str]:
    return plan.tasks_for("financial")[0].tools


# ---------------------------------------------------------------------------
# Routing rules — pure, no mocks needed
# ---------------------------------------------------------------------------

class TestRoutingRules:
    def test_public_uses_market_data_and_drops_funding_history(self):
        tools = _fin_tools(_plan(CompanyType.PUBLIC, "ACME"))
        assert FINANCIALS in tools
        assert FUNDING_ROUNDS not in tools, "a listed company's Series C is noise"

    def test_private_drops_the_ticker_lookup(self):
        """The whole point of the phase: stop paying for a guaranteed miss."""
        tools = _fin_tools(_plan(CompanyType.PRIVATE))
        assert FINANCIALS not in tools, "private companies have no ticker to resolve"
        assert FUNDING_ROUNDS in tools
        assert SEARCH_WEB in tools

    def test_unknown_keeps_every_tool(self):
        tools = _fin_tools(_plan(CompanyType.UNKNOWN))
        assert {FINANCIALS, FUNDING_ROUNDS, SEARCH_WEB} <= set(tools)

    @pytest.mark.parametrize("kind", list(CompanyType))
    def test_every_plan_dispatches_all_four_agents(self, kind):
        """Public/private changes *tools*, not which agents run.

        Skipping a whole agent is not justifiable on listing status alone —
        all four have something to say about either kind of company.
        """
        plan = _plan(kind)
        assert plan.agents == list(ALL_AGENTS)
        assert plan.skipped(ALL_AGENTS) == []

    def test_only_private_carries_a_financial_caveat(self):
        assert _plan(CompanyType.PRIVATE).caveat_for("financial")
        assert not _plan(CompanyType.PUBLIC).caveat_for("financial")
        assert not _plan(CompanyType.UNKNOWN).caveat_for("financial")

    @pytest.mark.parametrize("kind", list(CompanyType))
    def test_rationale_is_always_populated(self, kind):
        """Phase 3 shows this to a human at an approval gate."""
        assert _plan(kind).rationale.strip()

    def test_no_classification_degrades_to_unrestricted(self):
        """Absent information must widen the search, never narrow it."""
        plan = plan_for("Acme")
        assert plan.company_type is CompanyType.UNKNOWN
        assert {FINANCIALS, FUNDING_ROUNDS, SEARCH_WEB} <= set(_fin_tools(plan))


class TestToolNameDrift:
    def test_every_planned_tool_name_exists(self):
        """Plans name tools as strings; a typo must fail here, not at runtime."""
        import src.agents.financial as financial
        import src.agents.market as market
        import src.agents.risk as risk
        import src.agents.sentiment as sentiment

        real = {
            t.name
            for module in (financial, market, risk, sentiment)
            for t in module.TOOLS
        }
        planned = {
            tool
            for kind in CompanyType
            for task in _plan(kind).tasks
            for tool in task.tools
        }
        assert planned <= real, f"planned but nonexistent: {sorted(planned - real)}"


# ---------------------------------------------------------------------------
# Classification — the model proposes, the market-data lookup disposes
# ---------------------------------------------------------------------------

def _stub_propose(monkeypatch, classification: Classification):
    async def fake(company: str) -> Classification:
        return classification
    monkeypatch.setattr(cls, "_propose", fake)


def _stub_lookup(monkeypatch, table: dict[str, dict | None]):
    async def fake(candidate: str):
        return table.get((candidate or "").strip())
    monkeypatch.setattr(cls, "lookup_ticker_info", fake)


class TestClassification:
    def test_verified_ticker_makes_it_public(self, monkeypatch):
        _stub_propose(monkeypatch, Classification(
            company_type=CompanyType.PUBLIC, ticker="acme"))
        _stub_lookup(monkeypatch, {"acme": {"quoteType": "EQUITY", "symbol": "ACME"}})

        result = asyncio.run(cls.classify_company("Acme Corp"))
        assert result.company_type is CompanyType.PUBLIC
        assert result.ticker == "ACME", "should carry the resolved symbol, not the guess"

    def test_unverifiable_public_claim_becomes_unknown_not_private(self, monkeypatch):
        """The asymmetry that matters.

        No ticker resolving is not proof of privateness — it could be a foreign
        listing, a delisting, or a flaky provider. Asserting PRIVATE here would
        send the financial agent away from filings that may well exist.
        """
        _stub_propose(monkeypatch, Classification(
            company_type=CompanyType.PUBLIC, ticker="NOPE"))
        _stub_lookup(monkeypatch, {})

        result = asyncio.run(cls.classify_company("Mystery Inc"))
        assert result.company_type is CompanyType.UNKNOWN
        assert result.company_type is not CompanyType.PRIVATE
        assert result.ticker == ""

    def test_private_claim_survives_a_failed_lookup(self, monkeypatch):
        """Consistent evidence: a private company has no ticker to resolve."""
        _stub_propose(monkeypatch, Classification(company_type=CompanyType.PRIVATE))
        _stub_lookup(monkeypatch, {})

        result = asyncio.run(cls.classify_company("Stripe"))
        assert result.company_type is CompanyType.PRIVATE
        assert result.ticker == ""

    def test_non_equity_instrument_is_not_public(self, monkeypatch):
        """An ETF resolves fine but is not a company you can diligence."""
        _stub_propose(monkeypatch, Classification(
            company_type=CompanyType.PUBLIC, ticker="SPY"))
        _stub_lookup(monkeypatch, {"SPY": {"quoteType": "ETF", "symbol": "SPY"}})

        result = asyncio.run(cls.classify_company("SPY"))
        assert result.company_type is CompanyType.UNKNOWN

    def test_raw_input_is_tried_when_the_model_offers_no_ticker(self, monkeypatch):
        """Someone typing 'NVDA' straight in should still resolve."""
        _stub_propose(monkeypatch, Classification(company_type=CompanyType.UNKNOWN))
        _stub_lookup(monkeypatch, {"NVDA": {"quoteType": "EQUITY", "symbol": "NVDA"}})

        result = asyncio.run(cls.classify_company("NVDA"))
        assert result.company_type is CompanyType.PUBLIC
        assert result.ticker == "NVDA"

    def test_classifier_failure_degrades_to_unknown(self, monkeypatch):
        """CLAUDE.md: never crash the pipeline."""
        def boom(*args, **kwargs):
            raise RuntimeError("model unavailable")
        monkeypatch.setattr(cls, "get_llm", boom)  # get_llm is sync
        _stub_lookup(monkeypatch, {})

        result = asyncio.run(cls.classify_company("Acme"))
        assert result.company_type is CompanyType.UNKNOWN

    def test_classification_result_routes_to_a_usable_plan(self, monkeypatch):
        """End to end across the trust boundary, still offline."""
        _stub_propose(monkeypatch, Classification(company_type=CompanyType.PRIVATE))
        _stub_lookup(monkeypatch, {})

        result = asyncio.run(cls.classify_company("Stripe"))
        plan = plan_for("Stripe", result)
        assert FINANCIALS not in _fin_tools(plan)
        assert plan.caveat_for("financial")
