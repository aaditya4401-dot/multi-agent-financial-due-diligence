"""Tests for the refinement loop: gap detection, priority, and termination.

All offline. The point of doing gap analysis deterministically is that
"is this evidence good enough" can be tested without paying for a model to
have an opinion about it.
"""

import asyncio
from datetime import date, timedelta

import pytest

import src.agents.gap_analyzer as ga
from src.claims.models import Claim, SourceTier
from src.claims.ontology import Predicate, Unit
from src.gaps.detect import find_gaps, owner_for
from src.gaps.models import Gap, GapKind
from src.graph import route_after_gaps
from src.planning.models import CompanyType, ResearchTask
from src.planning.plan import FINANCIALS, plan_for


def _claim(predicate, value, agent="financial", tier=SourceTier.NEWS,
           url="https://news.com/a", days_old=0):
    return Claim(
        subject="testco",
        assertion=f"{predicate.value} is {value}",
        source_url=url,
        source_tier=tier,
        observed_at=date.today() - timedelta(days=days_old),
        extracted_by=agent,
        predicate=predicate,
        period="FY2025",
        value=value,
        unit=Unit.USD,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestGapDetection:
    def test_missing_core_metric_is_a_gap(self):
        gaps = find_gaps([], "testco", CompanyType.PUBLIC)
        predicates = {g.predicate for g in gaps if g.kind is GapKind.MISSING_METRIC}
        assert "revenue" in predicates

    def test_core_metrics_differ_by_company_type(self):
        """A private company has no market cap to find, so its absence is not
        a gap — chasing it would be a guaranteed wasted round."""
        public = {g.predicate for g in find_gaps([], "testco", CompanyType.PUBLIC)}
        private = {g.predicate for g in find_gaps([], "testco", CompanyType.PRIVATE)}
        assert "market_cap" in public and "market_cap" not in private
        assert "valuation" in private and "valuation" not in public

    def test_present_metric_is_not_a_gap(self):
        claims = [_claim(Predicate.REVENUE, 1.0e9)]
        gaps = find_gaps(claims, "testco", CompanyType.PUBLIC)
        assert "revenue" not in {
            g.predicate for g in gaps if g.kind is GapKind.MISSING_METRIC}

    def test_gaps_are_ranked_by_value_of_information(self):
        gaps = find_gaps([], "testco", CompanyType.PUBLIC)
        scores = [g.value_of_information for g in gaps]
        assert scores == sorted(scores, reverse=True)

    def test_revenue_outranks_market_share(self):
        """Priority must track what moves a verdict, not what is merely absent."""
        gaps = {g.predicate: g.value_of_information
                for g in find_gaps([], "testco", CompanyType.PUBLIC)}
        assert gaps["revenue"] > gaps["market_share"]

    def test_stale_evidence_is_flagged(self):
        old = _claim(Predicate.REVENUE, 1.0e9, days_old=4000)
        kinds = {g.kind for g in find_gaps([old], "testco", CompanyType.PUBLIC)}
        assert GapKind.STALE_EVIDENCE in kinds

    def test_fresh_evidence_is_not_stale(self):
        fresh = _claim(Predicate.REVENUE, 1.0e9, days_old=1)
        stale = [g for g in find_gaps([fresh], "testco", CompanyType.PUBLIC)
                 if g.kind is GapKind.STALE_EVIDENCE]
        assert stale == []

    def test_owner_routing(self):
        assert owner_for(Predicate.REVENUE) == "financial"
        assert owner_for(Predicate.MARKET_SHARE) == "market"
        assert owner_for(Predicate.GLASSDOOR_RATING) == "sentiment"

    def test_gap_id_survives_a_priority_change(self):
        """The same hole must hash the same across rounds, or the attempted
        set cannot suppress it and the loop could chase it forever."""
        g = Gap(kind=GapKind.MISSING_METRIC, agent="financial",
                subject="testco", predicate="revenue")
        assert g.id == g.model_copy(update={"value_of_information": 0.99}).id


# ---------------------------------------------------------------------------
# Loop policy and termination
# ---------------------------------------------------------------------------

class TestRefinementPolicy:
    def _state(self, **over):
        base = {
            "company": "TestCo",
            "plan": plan_for("TestCo"),
            "claims": [],
            "research_round": 0,
            "attempted_gaps": [],
        }
        base.update(over)
        return base

    def test_thin_evidence_triggers_refinement(self):
        out = asyncio.run(ga.gap_analyzer_node(self._state()))
        assert out["refine_tasks"], "no evidence at all should buy another round"
        assert out["research_round"] == 1

    def test_round_ceiling_stops_the_loop(self):
        """Guarantee 1: a hard ceiling on laps, regardless of open gaps."""
        out = asyncio.run(ga.gap_analyzer_node(
            self._state(research_round=ga.MAX_REFINEMENT_ROUNDS)))
        assert out["refine_tasks"] == []
        assert out["gaps"], "gaps still exist — we stopped on budget, not on success"

    def test_attempted_gaps_are_never_reissued(self):
        """Guarantee 2: the candidate set strictly shrinks."""
        first = asyncio.run(ga.gap_analyzer_node(self._state()))
        attempted = first["attempted_gaps"]
        assert attempted

        second = asyncio.run(ga.gap_analyzer_node(
            self._state(attempted_gaps=attempted)))
        assert not ({g.id for g in second["gaps"]} & set(attempted))

    def test_low_value_gaps_do_not_buy_a_round(self):
        """An unresolved detail that would not move the verdict is not worth
        money, however unresolved it is."""
        out = asyncio.run(ga.gap_analyzer_node(self._state()))
        for task in out["refine_tasks"]:
            assert task.focus
        chosen = [g for g in out["gaps"]
                  if g.value_of_information >= ga.MIN_VALUE_OF_INFORMATION]
        assert len(out["refine_tasks"]) <= min(len(chosen), ga.MAX_TASKS_PER_ROUND)

    def test_task_budget_is_respected(self):
        out = asyncio.run(ga.gap_analyzer_node(self._state()))
        assert len(out["refine_tasks"]) <= ga.MAX_TASKS_PER_ROUND

    def test_followup_inherits_the_plans_tool_restriction(self):
        """Otherwise refinement hands a private company back the market-data
        tool the planner deliberately removed, and the extra round is spent
        re-making the original mistake."""
        from src.planning.models import Classification
        plan = plan_for("Stripe", Classification(company_type=CompanyType.PRIVATE))
        out = asyncio.run(ga.gap_analyzer_node(
            self._state(company="Stripe", plan=plan)))

        financial = [t for t in out["refine_tasks"] if t.agent == "financial"]
        assert financial, "expected a financial follow-up"
        for task in financial:
            assert FINANCIALS not in task.tools

    def test_followup_asks_something_narrower(self):
        """A loop that re-runs the original prompt is a retry, not a loop."""
        out = asyncio.run(ga.gap_analyzer_node(self._state()))
        for task in out["refine_tasks"]:
            assert "targeted follow-up" in task.focus
            assert "Do not repeat" in task.focus


class TestRouting:
    def test_no_tasks_routes_onward_to_the_thesis(self):
        """Onward means thesis, not synthesizer: a view is formed from the
        evidence before the memo is written."""
        assert route_after_gaps({"company": "X", "refine_tasks": []}) == "thesis"

    def test_tasks_route_back_into_research(self):
        sends = route_after_gaps({
            "company": "TestCo",
            "refine_tasks": [ResearchTask(agent="financial", focus="find revenue")],
        })
        assert [s.node for s in sends] == ["research"]
        assert sends[0].arg["company"] == "TestCo"


# ---------------------------------------------------------------------------
# The loop, on the real graph
# ---------------------------------------------------------------------------

class TestLoopOnTheRealGraph:
    """Runs the actual compiled graph with every LLM call stubbed.

    Unit-testing the policy is not enough: the thing that would bite is the
    wiring — a cycle that never fires, or one that never stops.
    """

    def _run(self, monkeypatch, claims_per_block):
        import src.agents.base as base
        import src.agents.evidence as ev
        import src.graph as g
        from src.planning.models import Classification

        invocations = []

        class FakeAgent:
            def __init__(self, tools): self.tools = tools
            async def ainvoke(self, payload, config=None):
                invocations.append(payload["messages"][0]["content"])
                class M:
                    type, content, tool_calls = "ai", "stub", []
                return {"messages": [M()]}

        monkeypatch.setattr(
            base, "_compiled_agent", lambda n, tools, sp, tier: FakeAgent(tools))

        async def fake_classify(company):
            return Classification(company_type=CompanyType.PUBLIC, ticker="TEST")
        monkeypatch.setattr(g, "classify_company", fake_classify)

        async def fake_extract(block, subject):
            return list(claims_per_block)
        monkeypatch.setattr(ev, "extract_claims", fake_extract)

        async def fake_tensions(subject, claims):
            return []
        monkeypatch.setattr(ev, "detect_tensions", fake_tensions)

        async def fake_synth(state):
            return {"final_report": {"rounds": state.get("research_round")}}
        monkeypatch.setattr(g, "synthesizer_node", fake_synth)

        app = g.build_workflow().compile()
        state = asyncio.run(app.ainvoke(
            {"company": "TestCo", "findings": [], "claims": [], "conflicts": [],
             "extracted_upto": 0, "research_round": 0, "attempted_gaps": [],
             "messages": []},
            config={"recursion_limit": 50}))
        return state, invocations

    def test_thin_evidence_actually_triggers_a_second_round(self, monkeypatch):
        """With no claims extracted, every core metric is missing, so the
        gap analyzer should buy another round of narrow research."""
        state, invocations = self._run(monkeypatch, claims_per_block=[])

        assert len(invocations) > 4, "expected refinement beyond the initial four tasks"
        followups = [i for i in invocations if "targeted follow-up" in i]
        assert followups, "the extra round must ask narrow questions, not re-run the survey"

    def test_the_loop_terminates(self, monkeypatch):
        """The guarantee that matters. If this hangs or blows the recursion
        limit, the termination argument is wrong."""
        state, _ = self._run(monkeypatch, claims_per_block=[])
        assert state["final_report"]["rounds"] == ga.MAX_REFINEMENT_ROUNDS + 1

    def test_extraction_is_not_repeated_for_old_blocks(self, monkeypatch):
        """Re-extracting round-one findings on round two would pay twice for
        the same text."""
        state, _ = self._run(monkeypatch, claims_per_block=[])
        assert state["extracted_upto"] == len(state["findings"])

    def test_good_evidence_skips_refinement(self, monkeypatch):
        """The loop must be conditional on the evidence, not unconditional."""
        strong = [
            _claim(p, 1.0e9, tier=SourceTier.FILING, url="https://sec.gov/a")
            for p in (Predicate.REVENUE, Predicate.REVENUE_GROWTH_YOY,
                      Predicate.GROSS_MARGIN, Predicate.OPERATING_MARGIN,
                      Predicate.NET_INCOME, Predicate.MARKET_CAP,
                      Predicate.MARKET_SHARE)
        ]
        state, invocations = self._run(monkeypatch, claims_per_block=strong)
        assert len(invocations) == 4, "complete evidence should not buy another round"
