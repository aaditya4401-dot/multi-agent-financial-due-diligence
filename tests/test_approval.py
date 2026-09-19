"""Tests for the human-in-the-loop plan gate.

All offline. The interrupt machinery is exercised for real — these run the
compiled graph with a checkpointer and actually pause it — but every LLM call
is stubbed.
"""

import asyncio

import pytest

import src.agents.base as base
import src.agents.evidence as ev
import src.agents.gap_analyzer as ga
import src.graph as g
from src.agents.approval import _as_decision, review_payload
from src.planning.models import Classification, CompanyType, PlanDecision
from src.planning.plan import plan_for


# ---------------------------------------------------------------------------
# The decision itself
# ---------------------------------------------------------------------------

class TestPlanDecision:
    def test_approve_leaves_the_plan_alone(self):
        plan = plan_for("TestCo")
        assert len(PlanDecision().applied_to(plan).tasks) == len(plan.tasks)

    def test_revise_drops_named_agents(self):
        plan = plan_for("TestCo")
        revised = PlanDecision(
            action="revise", drop_agents=["sentiment"]).applied_to(plan)
        assert "sentiment" not in {t.agent for t in revised.tasks}

    def test_cancel_empties_the_plan_rather_than_nulling_it(self):
        """An empty plan flows through the same path as any other empty plan,
        so cancellation needs no special case downstream."""
        cancelled = PlanDecision(action="cancel").applied_to(plan_for("TestCo"))
        assert cancelled.tasks == []
        assert cancelled.company == "TestCo"

    def test_reviewer_note_is_kept_in_the_rationale(self):
        """The audit trail should record who decided what, and why."""
        revised = PlanDecision(note="wrong Acme").applied_to(plan_for("Acme"))
        assert "wrong Acme" in revised.rationale

    @pytest.mark.parametrize("raw,expected", [
        ({"action": "cancel"}, "cancel"),
        ("cancel", "cancel"),
        ("approve", "approve"),
        ({"nonsense": True}, "approve"),
        (None, "approve"),
        (42, "approve"),
    ])
    def test_decision_parsing_is_forgiving(self, raw, expected):
        """The value crosses a process boundary. A malformed answer is not a
        reason to discard a run the reviewer already looked at."""
        assert _as_decision(raw).action == expected

    def test_payload_shows_what_will_be_spent(self):
        payload = review_payload(plan_for("TestCo"))
        assert "4 research agent(s)" in payload["question"]
        assert len(payload["tasks"]) == 4


# ---------------------------------------------------------------------------
# The gate, on the real compiled graph
# ---------------------------------------------------------------------------

class TestApprovalGate:
    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch):
        self.planner_calls = []

        # Isolate the gate from the refinement loop: with no claims extracted
        # every core metric is missing, so gap analysis would otherwise buy an
        # extra round and confound the task counts below. The loop has its own
        # tests in test_gaps.py.
        monkeypatch.setattr(ga, "MAX_REFINEMENT_ROUNDS", 0)

        class FakeAgent:
            def __init__(self, tools): self.tools = tools
            async def ainvoke(self, payload, config=None):
                class M:
                    type, content, tool_calls = "ai", "stub", []
                return {"messages": [M()]}
        monkeypatch.setattr(
            base, "_compiled_agent", lambda n, t, sp, tier: FakeAgent(t))

        async def fake_classify(company):
            self.planner_calls.append(company)
            return Classification(company_type=CompanyType.PRIVATE)
        monkeypatch.setattr(g, "classify_company", fake_classify)

        async def no_claims(block, subject): return []
        monkeypatch.setattr(ev, "extract_claims", no_claims)

        async def no_tensions(subject, claims): return []
        monkeypatch.setattr(ev, "detect_tensions", no_tensions)

        async def fake_synth(state):
            from src.agents.synthesizer import agent_status
            failed, skipped = agent_status(state)
            return {"final_report": {"failed": failed, "skipped": skipped,
                                     "tasks": len(state.get("findings") or [])}}
        monkeypatch.setattr(g, "synthesizer_node", fake_synth)

    def test_unattended_runs_are_not_gated(self):
        """The gate must be invisible to the CLI, cron, and every test.

        Asserts on graph state rather than the report: an unattended run takes
        the pre-compiled module-level `app`, so the stubbed synthesizer this
        fixture installs is deliberately not in play here.
        """
        state = asyncio.run(g.run("TestCo"))
        assert g.pending_review(state) is None
        assert len(state["findings"]) == 4, "all four agents should have run"

    def test_review_pauses_before_spending(self):
        """The whole point: nothing is researched until a human says so."""
        state = asyncio.run(g.run("Stripe", human_review=True))
        payload = g.pending_review(state)

        assert payload is not None
        assert payload["company"] == "Stripe"
        assert len(payload["tasks"]) == 4
        assert not state.get("findings"), "no research may run before approval"

    def test_approval_resumes_the_run(self):
        state = asyncio.run(g.run("Stripe", human_review=True))
        resumed = asyncio.run(g.resume(
            {"action": "approve"}, state["__thread_id__"]))

        assert g.pending_review(resumed) is None
        assert resumed["final_report"]["tasks"] == 4

    def test_planner_is_not_paid_for_twice(self):
        """An interrupting node re-executes from the top on resume, so the gate
        is a separate node. If classification ever moves inside it, this fails.
        """
        state = asyncio.run(g.run("Stripe", human_review=True))
        asyncio.run(g.resume({"action": "approve"}, state["__thread_id__"]))
        assert self.planner_calls == ["Stripe"], (
            f"classification ran {len(self.planner_calls)}x across one approval"
        )

    def test_dropping_an_agent_yields_a_skipped_section(self):
        """The genuine skip Phase 1 built the machinery for — and it must be
        reported as skipped, not as a failure."""
        state = asyncio.run(g.run("Stripe", human_review=True))
        resumed = asyncio.run(g.resume(
            {"action": "revise", "drop_agents": ["sentiment"], "note": "immaterial"},
            state["__thread_id__"]))

        report = resumed["final_report"]
        assert report["skipped"] == ["sentiment"]
        assert report["failed"] == [], "a deselected agent did not fail"
        assert report["tasks"] == 3

    def test_cancelling_runs_no_research(self):
        state = asyncio.run(g.run("Stripe", human_review=True))
        resumed = asyncio.run(g.resume(
            {"action": "cancel", "note": "wrong company"}, state["__thread_id__"]))

        assert resumed["final_report"]["tasks"] == 0
        assert "CANCELLED" in resumed["plan"].rationale

    def test_decision_survives_a_json_round_trip(self):
        """The reviewer may be in another process. Whatever the UI sends must
        arrive as something the gate understands."""
        import json
        state = asyncio.run(g.run("Stripe", human_review=True))
        wire = json.loads(json.dumps(
            PlanDecision(action="revise", drop_agents=["risk"]).model_dump()))
        resumed = asyncio.run(g.resume(wire, state["__thread_id__"]))
        assert resumed["final_report"]["skipped"] == ["risk"]
