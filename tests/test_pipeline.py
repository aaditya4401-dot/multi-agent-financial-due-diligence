"""Tests for pipeline semantics: caching, failure handling, and graph wiring.

All offline — no API keys and no network required.
"""

import asyncio

import pytest

import src.tools.search_tools as st
from src.agents.conflict_resolver import conflict_resolver_node
from src.agents.synthesizer import failed_agents, synthesizer_node
from src.agents.utils import error_findings, parse_react_output
from src.graph import app
from src.models.schemas import DueDiligenceReport, inconclusive_report
from src.state import AgentFindings, Finding


def _findings(name: str, n: int = 1, ok: bool = True) -> AgentFindings:
    return AgentFindings(
        agent_name=name,
        findings=[
            Finding(
                claim=f"{name} claim {i}",
                source="http://example.com",
                confidence=0.8,
                source_quality="news_article",
                date_of_data="2026-01-01",
            )
            for i in range(n)
        ],
        summary=f"{name} summary",
        data_sources_used=["tool"],
        ok=ok,
    )


# ---------------------------------------------------------------------------
# Search caching + coalescing
# ---------------------------------------------------------------------------

class TestSearchCaching:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        st.clear_cache()
        self.calls: list[str] = []

        async def fake(query, max_results, **kwargs):
            self.calls.append(query)
            await asyncio.sleep(0.01)
            return [st.SearchResult(title="t", url="u", content="c", date="")]

        monkeypatch.setattr(st, "_search_uncached", fake)
        yield
        st.clear_cache()

    def test_concurrent_identical_queries_coalesce(self):
        async def go():
            return await asyncio.gather(*[st.asearch_web("same query") for _ in range(5)])

        results = asyncio.run(go())
        assert len(self.calls) == 1, "concurrent identical queries should hit upstream once"
        assert all(r == results[0] for r in results)

    def test_repeat_query_served_from_cache(self):
        async def go():
            await st.asearch_web("cached query")
            await st.asearch_web("cached query")

        asyncio.run(go())
        assert len(self.calls) == 1

    def test_differing_kwargs_do_not_collide(self):
        async def go():
            await st.asearch_web("q")
            await st.asearch_web("q", include_domains=["sec.gov"])

        asyncio.run(go())
        assert len(self.calls) == 2, "scoped searches must not share a cache entry"

    def test_query_normalisation(self):
        async def go():
            await st.asearch_web("Stripe  Revenue")
            await st.asearch_web("stripe revenue")

        asyncio.run(go())
        assert len(self.calls) == 1, "whitespace/case should normalise to one key"


class TestSyncShim:
    def test_search_web_rejects_async_context(self):
        async def go():
            with pytest.raises(RuntimeError, match="asearch_web"):
                st.search_web("anything")

        asyncio.run(go())


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------

class TestFailureSemantics:
    def test_error_findings_marked_not_ok(self):
        f = error_findings("financial", RuntimeError("boom"))
        assert f["ok"] is False
        assert f["findings"] == [], "a failed agent must not fabricate findings"

    def test_successful_parse_marked_ok(self):
        parsed = parse_react_output("market", {"messages": []})
        assert parsed["ok"] is True

    def test_failed_agents_detected(self):
        state = {
            "company": "TestCo",
            "financial_findings": _findings("financial"),
            "market_findings": _findings("market", ok=False),
            "risk_findings": None,
            "sentiment_findings": _findings("sentiment"),
        }
        assert failed_agents(state) == ["market", "risk"]

    def test_inconclusive_is_not_unfavorable(self):
        r = inconclusive_report("TestCo", "2026-08-14", "everything failed")
        assert r.overall_verdict == "Inconclusive"
        assert r.risk_level == "Unknown"
        assert r.overall_confidence == 0.0
        assert not r.financial_section.available

    def test_all_agents_failed_short_circuits_without_llm(self):
        """No API key is set — reaching the LLM would raise, so this proves
        the short-circuit fires before any model call."""
        state = {
            "company": "TestCo",
            "conflicts": [],
            **{f"{a}_findings": _findings(a, ok=False)
               for a in ("financial", "market", "risk", "sentiment")},
        }
        result = asyncio.run(synthesizer_node(state))
        report = result["final_report"]
        assert report["overall_verdict"] == "Inconclusive"
        assert report["risk_level"] == "Unknown"
        assert sorted(report["unavailable_sections"]) == [
            "financial", "market", "risk", "sentiment"
        ]

    def test_report_dump_matches_schema(self):
        r = inconclusive_report("TestCo", "2026-08-14", "reason")
        assert DueDiligenceReport.model_validate(r.model_dump())


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------

class TestConflictResolver:
    def test_skips_llm_with_fewer_than_two_sources(self):
        state = {
            "company": "TestCo",
            "financial_findings": _findings("financial"),
            "market_findings": _findings("market", ok=False),
            "risk_findings": None,
            "sentiment_findings": _findings("sentiment", n=0),
        }
        assert asyncio.run(conflict_resolver_node(state)) == {"conflicts": []}

    def test_failed_agents_excluded_from_comparison(self):
        from src.agents.conflict_resolver import _usable_sections

        state = {
            "financial_findings": _findings("financial"),
            "market_findings": _findings("market", ok=False),
            "risk_findings": _findings("risk"),
            "sentiment_findings": _findings("sentiment", n=0),
        }
        labels = [label for label, _ in _usable_sections(state)]
        assert labels == ["FINANCIAL", "RISK"]


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

class TestGraphTopology:
    def test_all_nodes_present(self):
        nodes = set(app.get_graph().nodes)
        assert {
            "orchestrator", "financial", "market", "risk",
            "sentiment", "conflict_resolver", "synthesizer",
        } <= nodes

    def test_agents_fan_out_and_back_in(self):
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        for agent in ("financial", "market", "risk", "sentiment"):
            assert ("orchestrator", agent) in edges
            assert (agent, "conflict_resolver") in edges
        assert ("conflict_resolver", "synthesizer") in edges
