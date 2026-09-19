"""Tests for pipeline semantics: caching, failure handling, and graph wiring.

All offline — no API keys and no network required.
"""

import asyncio
from datetime import date

import pytest

import src.tools.search_tools as st
from src.claims.confidence import score_claims
from src.claims.models import Claim, SourceTier
from src.claims.ontology import Predicate, Unit
from src.agents.conflict_resolver import detect_tensions
from src.agents.evidence import evidence_node
from src.agents.synthesizer import (
    agent_status,
    failed_agents,
    section_confidences,
    synthesizer_node,
)
import src.agents.base as base
from src.agents.utils import error_findings, parse_react_output
from src.graph import app, dispatch_research
from src.planning.models import ResearchTask
from src.planning.plan import plan_for
from src.models.schemas import DueDiligenceReport, inconclusive_report
from src.state import AgentFindings, Finding


def _quant_claim(
    value: float,
    agent: str,
    url: str,
    tier: SourceTier = SourceTier.NEWS,
    predicate: Predicate = Predicate.PAYMENT_VOLUME,
) -> Claim:
    return Claim(
        subject="testco",
        assertion=f"{predicate.value} was {value}",
        source_url=url,
        source_tier=tier,
        observed_at=date.today(),
        extracted_by=agent,
        predicate=predicate,
        period="FY2025",
        value=value,
        unit=Unit.USD,
    )


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
        """An agent that errored and one that produced nothing both count as
        failed, so long as the plan asked for them."""
        state = {
            "company": "TestCo",
            "findings": [
                _findings("financial"),
                _findings("market", ok=False),
                # risk produced nothing at all
                _findings("sentiment"),
            ],
        }
        assert failed_agents(state) == ["market", "risk"]

    def test_skipped_is_not_failed(self):
        """A section the plan never asked for is not an evidence gap.

        Counting it as a failure would make the system score itself down for
        following its own plan.
        """
        plan = plan_for("TestCo")
        plan.tasks = [t for t in plan.tasks if t.agent != "sentiment"]
        state = {
            "company": "TestCo",
            "plan": plan,
            "findings": [_findings(a) for a in ("financial", "market", "risk")],
        }
        failed, skipped = agent_status(state)
        assert skipped == ["sentiment"]
        assert failed == [], "nothing failed — one section was simply not requested"

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
            "findings": [_findings(a, ok=False)
                         for a in ("financial", "market", "risk", "sentiment")],
        }
        result = asyncio.run(synthesizer_node(state))
        report = result["final_report"]
        assert report["overall_verdict"] == "Inconclusive"
        assert report["risk_level"] == "Unknown"
        assert sorted(report["unavailable_sections"]) == [
            "financial", "market", "risk", "sentiment"
        ]

    def test_cancelled_run_is_not_reported_as_a_failure(self):
        """A reviewer cancelling and every agent crashing both end with no
        findings, but calling the first one a failure is a false statement
        about the system's own behaviour.

        Found by driving the Streamlit gate, not by a unit test — the empty
        plan only arises through the approval path.
        """
        cancelled = plan_for("TestCo").model_copy(update={"tasks": []})
        state = {"company": "TestCo", "conflicts": [], "findings": [],
                 "plan": cancelled}

        report = asyncio.run(synthesizer_node(state))["final_report"]
        summary = report["executive_summary"]
        assert "no research was run" in summary
        assert "failed" not in summary
        assert report["overall_verdict"] == "Inconclusive"

    def test_genuine_total_failure_still_says_failed(self):
        """The other side of the same coin — don't soften a real failure."""
        state = {
            "company": "TestCo", "conflicts": [],
            "findings": [_findings(a, ok=False)
                         for a in ("financial", "market", "risk", "sentiment")],
        }
        report = asyncio.run(synthesizer_node(state))["final_report"]
        assert "failed" in report["executive_summary"]

    def test_report_dump_matches_schema(self):
        r = inconclusive_report("TestCo", "2026-08-14", "reason")
        assert DueDiligenceReport.model_validate(r.model_dump())


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------

class TestEvidenceNode:
    """No API key is set, so any LLM call would raise. These prove the
    deterministic paths run without one."""

    def test_no_findings_yields_no_claims_or_conflicts(self):
        state = {
            "company": "TestCo",
            "findings": [_findings(a, ok=False)
                         for a in ("financial", "market", "risk", "sentiment")],
        }
        assert asyncio.run(evidence_node(state)) == {
            "claims": [], "conflicts": [], "extracted_upto": 4,
        }

    def test_tension_detection_skipped_below_two_claims(self):
        assert asyncio.run(detect_tensions("testco", [])) == []

    def test_tension_detection_skipped_for_single_agent(self):
        """Tension is interesting between perspectives, not within one."""
        claims = [
            Claim(subject="testco", assertion=f"observation {i}",
                  observed_at=date.today(), extracted_by="risk",
                  source_url=f"https://a{i}.com")
            for i in range(3)
        ]
        assert asyncio.run(detect_tensions("testco", claims)) == []

    def test_detected_contradiction_becomes_a_report_conflict(self):
        """The deterministic detector's output must land in the report shape."""
        from src.agents.evidence import _to_conflict
        from src.claims.contradictions import detect

        a = _quant_claim(1.9e12, "financial", "https://ft.com/x", SourceTier.TIER1_NEWS)
        b = _quant_claim(1.4e12, "market", "https://blog.com/y", SourceTier.NEWS)
        result = detect([a, b])
        score_claims([a, b], result)

        conflict = _to_conflict(result.contradictions[0])
        assert conflict["type"] == "factual_contradiction"
        assert {conflict["agent_a"], conflict["agent_b"]} == {"financial", "market"}
        assert "payment_volume" in conflict["resolution"]
        assert 0.0 <= conflict["resolved_confidence"] <= 1.0


class TestComputedSectionConfidence:
    def test_confidence_derived_from_claims_not_guessed(self):
        strong = _quant_claim(1.0e12, "financial", "https://sec.gov/a", SourceTier.FILING)
        weak = _quant_claim(5.0e11, "market", "https://x.com/b", SourceTier.SOCIAL)
        score_claims([strong, weak], None)

        confidences = section_confidences({"claims": [strong, weak]})
        assert confidences["financial"] > confidences["market"]

    def test_no_claims_yields_no_sections(self):
        assert section_confidences({"claims": []}) == {}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

class TestImportHygiene:
    """`src.state` and `src.claims` reference each other. Every module must
    still import standalone, in any order — a cycle here only shows up as an
    ImportError in whichever entry point happens to be imported first."""

    @pytest.mark.parametrize("module", [
        "src.state",
        "src.claims",
        "src.claims.extract",
        "src.claims.models",
        "src.graph",
        "src.planning.plan",
        "src.planning.classify",
        "src.agents.registry",
        "src.agents.evidence",
        "src.agents.synthesizer",
    ])
    def test_module_imports_standalone(self, module):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"


class TestResearchNode:
    """The node dispatched by Send. Its input is the payload, not graph state."""

    def _fake_agent(self, monkeypatch, recorder=None):
        class FakeAgent:
            def __init__(self, tools): self.tools = tools
            async def ainvoke(self, payload, config=None):
                if recorder is not None:
                    recorder["tools"] = [t.name for t in self.tools]
                    recorder["prompt"] = payload["messages"][0]["content"]
                class M:
                    type, content, tool_calls = "ai", "stub summary", []
                return {"messages": [M()]}
        monkeypatch.setattr(
            base, "_compiled_agent", lambda name, tools, sp, tier: FakeAgent(tools))

    def test_appends_one_findings_block(self, monkeypatch):
        self._fake_agent(monkeypatch)
        out = asyncio.run(base.research_node(
            {"company": "TestCo", "task": ResearchTask(agent="market")}))
        assert list(out) == ["findings"]
        assert len(out["findings"]) == 1
        assert out["findings"][0]["agent_name"] == "market"

    def test_planner_narrows_the_toolset(self, monkeypatch):
        """The whole point of the phase: tools are a runtime decision."""
        rec = {}
        self._fake_agent(monkeypatch, rec)
        asyncio.run(base.research_node({
            "company": "TestCo",
            "task": ResearchTask(agent="financial", tools=["tool_search_web"]),
        }))
        assert rec["tools"] == ["tool_search_web"]

    def test_focus_is_appended_to_the_task_prompt(self, monkeypatch):
        rec = {}
        self._fake_agent(monkeypatch, rec)
        asyncio.run(base.research_node({
            "company": "TestCo",
            "task": ResearchTask(agent="risk", focus="Look at enforcement actions."),
        }))
        assert "TestCo" in rec["prompt"]
        assert "Look at enforcement actions." in rec["prompt"]

    def test_caveat_travels_into_the_findings(self, monkeypatch):
        """It has to reach the synthesizer, which is what honours it in prose."""
        self._fake_agent(monkeypatch)
        out = asyncio.run(base.research_node({
            "company": "TestCo",
            "task": ResearchTask(agent="financial", caveat="Estimates only."),
        }))
        assert out["findings"][0]["caveat"] == "Estimates only."

    def test_serialised_task_is_accepted(self, monkeypatch):
        """A checkpointer round-trips state through JSON, so a resumed run
        hands this node a dict where a fresh run hands it a model."""
        self._fake_agent(monkeypatch)
        out = asyncio.run(base.research_node(
            {"company": "TestCo", "task": ResearchTask(agent="market").model_dump()}))
        assert out["findings"][0]["agent_name"] == "market"

    def test_unknown_agent_degrades_instead_of_raising(self):
        """CLAUDE.md: never crash the pipeline."""
        out = asyncio.run(base.research_node(
            {"company": "TestCo", "task": ResearchTask(agent="astrology")}))
        assert out["findings"][0]["ok"] is False


class TestGraphTopology:
    def test_all_nodes_present(self):
        nodes = set(app.get_graph().nodes)
        assert {"planner", "research", "evidence", "synthesizer"} <= nodes

    def test_research_fan_out_is_conditional(self):
        """The planner must *choose*. A plain edge here would mean the fan-out
        is constant again, which is the thing this phase removed."""
        edges = {(e.source, e.target): e for e in app.get_graph().edges}
        assert edges[("approval", "research")].conditional
        # The gate sits between planning and any spending.
        assert ("planner", "approval") in edges

    def test_fan_in_is_a_single_edge(self):
        """One edge, so it fires once for any number of dispatched tasks.

        The old join edge named all four agents and could never have tolerated
        a variable-width fan-out.
        """
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        assert ("research", "evidence") in edges
        assert ("evidence", "gap_analyzer") in edges

    def test_empty_plan_routes_past_research(self):
        """Reachable once a human can deselect every agent at an approval gate.

        Without this the graph would stall instead of reporting inconclusive.
        """
        assert dispatch_research({"plan": plan_for("TestCo").model_copy(
            update={"tasks": []})}) == "thesis"

    def test_graph_is_cyclic(self):
        """gap_analyzer must be able to send work back to research.

        Without this edge the graph is a line again and the synthesizer can
        only write up whatever the first pass happened to find.
        """
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        assert ("gap_analyzer", "research") in edges
        assert ("gap_analyzer", "thesis") in edges
        assert ("thesis", "synthesizer") in edges

    def test_dispatch_emits_one_send_per_task(self):
        plan = plan_for("TestCo")
        sends = dispatch_research({"plan": plan})
        assert len(sends) == len(plan.tasks)
        assert {s.node for s in sends} == {"research"}
        # Payload must be self-contained: a Send'd node cannot read graph state.
        for send in sends:
            assert send.arg["company"] == "TestCo"
            assert send.arg["task"] in plan.tasks
