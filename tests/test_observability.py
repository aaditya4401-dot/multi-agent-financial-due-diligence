"""Tests for LangSmith tracing.

All offline. Two rules keep them that way, and both matter:

* ``conftest.py`` scrubs ``LANGSMITH_*`` from the environment, so the default
  state under test is genuinely "tracing off" no matter what the developer
  running the suite has exported.
* Tests that need tracing *on* monkeypatch ``observability.tracing_enabled``
  rather than setting ``LANGSMITH_TRACING``. That distinction is deliberate:
  the env var also switches on LangChain's own global tracer, which would start
  POSTing every stubbed run to a real endpoint. Patching our predicate
  exercises this module's logic and leaves the global tracer asleep.

The client is a hand-written recording stub, matching the suite's existing
style — there is no ``unittest.mock`` anywhere in this project.
"""

import asyncio
import threading
import time

import pytest
from langchain_core.callbacks import BaseCallbackHandler

import src.agents.base as base
import src.agents.evidence as ev
import src.agents.gap_analyzer as ga
import src.graph as g
import src.observability as obs
from src.eval.models import CitationAudit, EvaluationReport, GroundednessCheck
from src.llm import Tier, get_llm
from src.planning.models import Classification, CompanyType


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class RecordingClient:
    """Stands in for ``langsmith.Client``, remembering what it was asked to do.

    It also has to absorb the run-ingestion calls, not just feedback: the root
    run is opened through this client, so without ``create_run``/``update_run``
    the tracer would fall back to a real one and the suite would start posting
    to LangSmith for real.
    """

    def __init__(self):
        self.feedback = []
        self.updates = []
        self.created_runs = []

    def create_feedback(self, run_id, key=None, score=None, comment=None,
                        **kwargs):
        self.feedback.append({"run_id": run_id, "key": key, "score": score,
                              "comment": comment})

    def create_run(self, name=None, **kwargs):
        self.created_runs.append({"name": name, **kwargs})

    def update_run(self, run_id, **kwargs):
        self.updates.append({"run_id": run_id, **kwargs})

    def flush(self):
        pass

    # The tracer probes for these; absorb whatever it asks for rather than
    # letting an AttributeError push it onto a real client.
    def __getattr__(self, item):
        return lambda *a, **k: None


class ExplodingClient:
    """Every call raises. The pipeline must not notice."""

    def __init__(self):
        self.calls = 0

    def create_feedback(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("langsmith is down")

    def update_run(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("langsmith is down")


class TreeRecorder(BaseCallbackHandler):
    """Records the chain-run tree LangGraph emits, without a network client.

    This is how the run tree gets asserted offline: the tree LangSmith would
    display is built from exactly these callbacks, so recording them proves the
    shape without anything being sent anywhere.
    """

    def __init__(self):
        self.runs = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name")
        self.runs.append({"name": name, "run_id": run_id,
                          "parent_run_id": parent_run_id})

    # -- queries -----------------------------------------------------------

    def named(self, name):
        return [r for r in self.runs if r["name"] == name]

    @property
    def roots(self):
        return [r for r in self.runs if r["parent_run_id"] is None]


def _llm_result(input_tokens=0, output_tokens=0, style="usage_metadata"):
    """Build an LLMResult-shaped object the usage callback can read.

    Two shapes, because the callback has to cope with both: chat models expose
    ``usage_metadata`` on the message, while some responses only carry the
    OpenAI-style aggregate in ``llm_output``.
    """
    total = input_tokens + output_tokens

    if style == "usage_metadata":
        class Message:
            usage_metadata = {"input_tokens": input_tokens,
                              "output_tokens": output_tokens,
                              "total_tokens": total}

        class Generation:
            message = Message()

        class Result:
            generations = [[Generation()]]
            llm_output = {}

        return Result()

    class Result:
        generations = []
        llm_output = {"token_usage": {"prompt_tokens": input_tokens,
                                      "completion_tokens": output_tokens,
                                      "total_tokens": total}}

    return Result()


def _evaluation(score=0.75):
    return EvaluationReport(
        score=score,
        checks=[
            GroundednessCheck(name="claims_have_sources", passed=True,
                              score=1.0, detail="all claims sourced"),
            GroundednessCheck(name="figures_traceable", passed=False,
                              score=0.0, detail="1 figure unsupported"),
        ],
        unsupported_figures=["$88B"],
        citation_audit=CitationAudit(drivers_proposed=3, drivers_kept=2,
                                     invented_citations=1),
    )


@pytest.fixture
def tracing_on(monkeypatch):
    """Turn tracing on without waking LangChain's global tracer."""
    monkeypatch.setattr(obs, "tracing_enabled", lambda: True)
    obs.reset_client()
    yield
    obs.reset_client()


@pytest.fixture
def recording_client(monkeypatch, tracing_on):
    client = RecordingClient()
    monkeypatch.setattr(obs, "get_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# 1. Off by default
# ---------------------------------------------------------------------------

class TestOffByDefault:
    """Absent env vars must mean absent tracing — not degraded tracing."""

    def test_disabled_when_env_absent(self):
        assert obs.tracing_enabled() is False

    def test_switch_without_key_stays_off(self, monkeypatch):
        """A half-configured environment degrades to off, not to noisy errors."""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert obs.tracing_enabled() is False

    def test_key_without_switch_stays_off(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_fake")
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        assert obs.tracing_enabled() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
    def test_truthy_spellings_enable(self, monkeypatch, value):
        monkeypatch.setenv("LANGSMITH_TRACING", value)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_fake")
        assert obs.tracing_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "off"])
    def test_falsy_spellings_stay_off(self, monkeypatch, value):
        monkeypatch.setenv("LANGSMITH_TRACING", value)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_fake")
        assert obs.tracing_enabled() is False

    def test_no_client_is_built(self):
        obs.reset_client()
        assert obs.get_client() is None

    def test_config_is_untouched(self):
        """The whole point: the pipeline's config is what it always was."""
        handle = obs.start_trace("TestCo")
        assert handle.enabled is False
        assert handle.run_id is None

        config = {"recursion_limit": 50}
        handle.apply_to(config)
        assert config == {"recursion_limit": 50}

    def test_finalize_is_a_noop(self):
        assert obs.finalize(obs.start_trace("TestCo"), {}) == {}

    def test_annotate_is_a_noop_outside_a_run(self):
        assert obs.annotate_current_run(name="x", foo=1) is None


# ---------------------------------------------------------------------------
# 2. Tier metadata reaches the model client
# ---------------------------------------------------------------------------

class TestTierMetadata:
    """The mechanism all seven call sites depend on, pinned in one place.

    Tier is stamped on the cached client rather than at the call sites, so if
    this stops working every model call in every trace silently loses its tier
    and the cost-by-tier view goes blank. Cheap to assert, so assert it.
    """

    @pytest.fixture(autouse=True)
    def _constructible_client(self, monkeypatch):
        """Let ChatOpenAI be built, without letting it be *called*.

        These are the only tests here that construct a real client, and
        ``conftest`` scrubs ``OPENAI_API_KEY`` precisely so that constructing
        one fails. A placeholder key restores construction; nothing in this
        class invokes the model, so no request is ever made. The cache is
        cleared either side so neither the suite nor these tests inherit a
        client built under the other's environment.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
        get_llm.cache_clear()
        yield
        get_llm.cache_clear()

    @pytest.mark.parametrize("tier", list(Tier))
    def test_client_carries_routing_tier(self, tier):
        model = get_llm(tier)
        assert model.metadata["routing_tier"] == tier.value
        assert f"tier:{tier.value}" in model.tags

    @pytest.mark.parametrize("tier", list(Tier))
    def test_client_records_the_model_behind_the_tier(self, tier):
        model = get_llm(tier)
        assert model.metadata["routing_model"] == model.model_name

    def test_key_is_not_bare_tier(self):
        """`SourceTier` is an unrelated concept; the names must not collide."""
        assert "tier" not in get_llm(Tier.FAST).metadata


# ---------------------------------------------------------------------------
# 3. The run tree
# ---------------------------------------------------------------------------

class TestRunTree:
    """One run per execution, with a child per node and per planning iteration."""

    @pytest.fixture(autouse=True)
    def _stub_pipeline(self, monkeypatch):
        """Drive the real graph with every model call stubbed out."""
        class FakeAgent:
            def __init__(self, tools): self.tools = tools
            async def ainvoke(self, payload, config=None):
                class M: type, content, tool_calls = "ai", "stub", []
                return {"messages": [M()]}

        monkeypatch.setattr(base, "_compiled_agent",
                            lambda n, t, sp, tier: FakeAgent(t))

        async def fake_classify(company):
            return Classification(company_type=CompanyType.PRIVATE)
        monkeypatch.setattr(g, "classify_company", fake_classify)

        async def no_claims(block, subject): return []
        monkeypatch.setattr(ev, "extract_claims", no_claims)

        async def no_tensions(subject, claims): return []
        monkeypatch.setattr(ev, "detect_tensions", no_tensions)

        async def fake_synth(state):
            return {"final_report": {"company_name": state.get("company")}}
        monkeypatch.setattr(g, "synthesizer_node", fake_synth)

    def _run(self, recorder, rounds=1):
        state = {
            "company": "TestCo", "human_review": False, "findings": [],
            "claims": [], "conflicts": [], "extracted_upto": 0,
            "research_round": 0, "attempted_gaps": [], "thesis": None,
            "thesis_audit": None, "evaluation": None, "approvals": [],
            "messages": [],
        }
        return asyncio.run(g.app.ainvoke(
            state, config={"callbacks": [recorder], "recursion_limit": 50}))

    def test_exactly_one_root_run(self):
        rec = TreeRecorder()
        self._run(rec)
        assert len(rec.roots) == 1, "a run must trace as one tree, not several"

    def test_every_node_appears_as_a_child_run(self):
        rec = TreeRecorder()
        self._run(rec)

        for node in ("planner", "approval", "research", "evidence",
                     "gap_analyzer", "thesis", "synthesizer", "evaluation"):
            assert rec.named(node), f"no child run for node {node!r}"

    def test_node_runs_descend_from_the_root(self):
        rec = TreeRecorder()
        self._run(rec)
        root_id = rec.roots[0]["run_id"]

        for node in ("planner", "research", "gap_analyzer", "evaluation"):
            for run in rec.named(node):
                assert run["parent_run_id"] == root_id

    def test_one_research_run_per_dispatched_agent(self):
        """Four agents fan out via Send, so the first round is four runs."""
        rec = TreeRecorder()
        self._run(rec)
        assert len(rec.named("research")) >= 4

    def test_one_gap_analyzer_run_per_planning_iteration(self):
        """The refinement cycle re-enters the node, so laps are countable."""
        rec = TreeRecorder()
        final = self._run(rec)
        assert len(rec.named("gap_analyzer")) == final["research_round"]

    def test_loop_is_traced_more_than_once(self, monkeypatch):
        """With budget for two laps, two laps show up in the tree."""
        monkeypatch.setattr(ga, "MAX_REFINEMENT_ROUNDS", 2)
        rec = TreeRecorder()
        self._run(rec)
        assert len(rec.named("gap_analyzer")) >= 2


class TestSpanAnnotation:
    """Labelling the span a node is already running in.

    Exercised against a real LangSmith run tree, because the whole mechanism is
    ``get_current_run_tree()`` returning something — a test with our own double
    would assert nothing about whether that holds. The tree is given an inert
    client, so the run is built in memory and never sent.
    """

    @pytest.fixture
    def run_tree(self, monkeypatch):
        # `tracing_context` opens a run context without setting any env var,
        # so LangChain's global tracer stays asleep — and the client is inert,
        # so the run is assembled in memory and never sent anywhere.
        monkeypatch.setattr(obs, "tracing_enabled", lambda: True)

        class InertClient:
            def create_run(self, *a, **k): pass
            def update_run(self, *a, **k): pass

        from langsmith.run_helpers import trace, tracing_context
        with tracing_context(enabled=True):
            with trace(name="research", run_type="chain",
                       client=InertClient()) as handle:
                yield handle

    def test_renames_the_current_span(self, run_tree):
        from langsmith.run_helpers import get_current_run_tree

        obs.annotate_current_run(name="agent:financial")
        assert get_current_run_tree().name == "agent:financial"

    def test_merges_metadata_into_the_current_span(self, run_tree):
        from langsmith.run_helpers import get_current_run_tree

        obs.annotate_current_run(**{"dd.agent": "financial",
                                    "dd.routing_tier": "reasoning"})

        meta = get_current_run_tree().extra["metadata"]
        assert meta["dd.agent"] == "financial"
        assert meta["dd.routing_tier"] == "reasoning"
        # Pre-existing keys survive the merge.
        assert "ls_method" in meta

    def test_name_and_metadata_together(self, run_tree):
        from langsmith.run_helpers import get_current_run_tree

        obs.annotate_current_run(name="planning_iteration_2",
                                 **{"dd.research_round": 2})

        run = get_current_run_tree()
        assert run.name == "planning_iteration_2"
        assert run.extra["metadata"]["dd.research_round"] == 2


# ---------------------------------------------------------------------------
# 4. Per-tier token accounting
# ---------------------------------------------------------------------------

class TestUsageByTier:
    """The one piece that cannot be read off return values.

    ``.with_structured_output()`` discards the AIMessage at six of the seven
    call sites, so these numbers exist only in the callback stream.
    """

    def _call(self, cb, run_id, tier, inp, out, style="usage_metadata",
              metadata=True):
        cb.on_chat_model_start(
            {}, [], run_id=run_id,
            metadata={"routing_tier": tier} if metadata else None,
            tags=[f"tier:{tier}"] if not metadata else None)
        cb.on_llm_end(_llm_result(inp, out, style), run_id=run_id)

    def test_counts_are_split_by_tier(self):
        cb = obs.UsageByTierCallback()
        self._call(cb, "r1", "fast", 100, 10)
        self._call(cb, "r2", "synthesis", 500, 200)
        self._call(cb, "r3", "fast", 50, 5)

        totals = cb.totals()
        assert totals["fast"] == {"input": 150, "output": 15, "total": 165}
        assert totals["synthesis"] == {"input": 500, "output": 200, "total": 700}

    def test_openai_shaped_usage_is_read_too(self):
        cb = obs.UsageByTierCallback()
        self._call(cb, "r1", "reasoning", 80, 20, style="llm_output")
        assert cb.totals()["reasoning"]["total"] == 100

    def test_tier_falls_back_to_the_tag(self):
        cb = obs.UsageByTierCallback()
        self._call(cb, "r1", "fast", 10, 1, metadata=False)
        assert "fast" in cb.totals()

    def test_untagged_call_is_attributed_to_unknown(self):
        """Visible rather than silently folded into a real tier."""
        cb = obs.UsageByTierCallback()
        cb.on_chat_model_start({}, [], run_id="r1", metadata=None, tags=None)
        cb.on_llm_end(_llm_result(5, 5), run_id="r1")
        assert obs.UNKNOWN_TIER in cb.totals()

    def test_summary_totals_across_tiers(self):
        cb = obs.UsageByTierCallback()
        self._call(cb, "r1", "fast", 100, 10)
        self._call(cb, "r2", "synthesis", 500, 200)

        summary = cb.summary()
        assert summary["total_tokens"] == 110 + 700
        assert summary["calls_by_tier"] == {"fast": 1, "synthesis": 1}

    def test_failed_call_does_not_leak_its_mapping(self):
        cb = obs.UsageByTierCallback()
        cb.on_chat_model_start({}, [], run_id="r1",
                               metadata={"routing_tier": "fast"}, tags=None)
        cb.on_llm_error(RuntimeError("boom"), run_id="r1")
        assert cb.totals() == {}

    def test_usage_free_response_is_ignored(self):
        cb = obs.UsageByTierCallback()

        class Empty:
            generations, llm_output = [], {}

        cb.on_chat_model_start({}, [], run_id="r1",
                               metadata={"routing_tier": "fast"}, tags=None)
        cb.on_llm_end(Empty(), run_id="r1")
        assert cb.totals() == {}


# ---------------------------------------------------------------------------
# 5. Feedback, from the evaluators that already ran
# ---------------------------------------------------------------------------

class TestFeedbackWiring:
    """The evaluators' existing output, transported — not recomputed."""

    def test_keys_cover_aggregate_and_each_check(self):
        items = obs._feedback_items({"evaluation": _evaluation()})
        keys = {i["key"] for i in items}

        assert keys == {
            "groundedness",
            "groundedness:claims_have_sources",
            "groundedness:figures_traceable",
            "citation_grounding_rate",
            "invented_citations",
        }

    def test_scores_are_passed_through_unchanged(self):
        items = {i["key"]: i["score"]
                 for i in obs._feedback_items({"evaluation": _evaluation(0.75)})}

        assert items["groundedness"] == 0.75
        assert items["groundedness:claims_have_sources"] == 1.0
        assert items["groundedness:figures_traceable"] == 0.0
        # 2 of 3 drivers kept.
        assert items["citation_grounding_rate"] == pytest.approx(0.6667, abs=1e-4)
        assert items["invented_citations"] == 1.0

    def test_unsupported_figures_ride_along_as_a_comment(self):
        """The diagnostic belongs with the score it explains."""
        items = {i["key"]: i["comment"]
                 for i in obs._feedback_items({"evaluation": _evaluation()})}
        assert "$88B" in items["groundedness"]

    def test_new_checks_appear_without_touching_this_module(self):
        """Keys are generated from the evaluator's output, not hardcoded."""
        report = _evaluation()
        report.checks.append(GroundednessCheck(
            name="a_brand_new_check", passed=True, score=1.0))

        keys = {i["key"] for i in obs._feedback_items({"evaluation": report})}
        assert "groundedness:a_brand_new_check" in keys

    def test_audit_alone_still_yields_citation_scores(self):
        """A run that died before evaluation still has its citation audit."""
        state = {"thesis_audit": CitationAudit(drivers_proposed=4,
                                               drivers_kept=4,
                                               invented_citations=0)}
        keys = {i["key"] for i in obs._feedback_items(state)}
        assert keys == {"citation_grounding_rate", "invented_citations"}

    def test_empty_state_produces_nothing(self):
        assert obs._feedback_items({}) == []

    def test_feedback_reaches_the_client(self, recording_client):
        pushed = obs.push_feedback("run-123", {"evaluation": _evaluation()})

        assert pushed == 5
        assert len(recording_client.feedback) == 5
        assert {f["run_id"] for f in recording_client.feedback} == {"run-123"}

    def test_nothing_is_pushed_without_a_run_id(self, recording_client):
        assert obs.push_feedback(None, {"evaluation": _evaluation()}) == 0
        assert recording_client.feedback == []


# ---------------------------------------------------------------------------
# 6. Run metadata
# ---------------------------------------------------------------------------

class TestRunMetadata:

    def _state(self, **overrides):
        state = {"company": "TestCo", "research_round": 2,
                 "attempted_gaps": ["g1", "g2"], "human_review": False,
                 "approvals": [], "refine_tasks": [], "findings": [],
                 "claims": [], "conflicts": []}
        state.update(overrides)
        return state

    def test_records_planning_iterations(self):
        meta = obs.run_metadata(self._state(research_round=3))
        assert meta["dd.planning_iterations"] == 3
        assert meta["dd.gaps_attempted"] == 2

    def test_convergence_is_reported_not_inferred(self):
        """Stopping because nothing was worth chasing is not the same fact as
        stopping because the budget ran out.

        The distinction cannot be recovered from the round count — that counter
        is incremented on both exits, so a run that converged on its first pass
        finishes sitting exactly on the ceiling. gap_analyzer records which it
        was; this only reads it back.
        """
        converged = self._state(research_round=1,
                                refine_stop_reason="converged")
        assert obs.run_metadata(converged)["dd.convergence"] == "converged"

        exhausted = self._state(research_round=1,
                                refine_stop_reason="round_ceiling")
        assert obs.run_metadata(exhausted)["dd.convergence"] == "round_ceiling"

    def test_converged_run_at_the_ceiling_is_not_mislabelled(self,
                                                             monkeypatch):
        """The exact case that was wrong in the first real trace."""
        monkeypatch.setattr(ga, "MAX_REFINEMENT_ROUNDS", 1)
        meta = obs.run_metadata(self._state(research_round=1,
                                            attempted_gaps=[],
                                            refine_stop_reason="converged"))
        assert meta["dd.convergence"] == "converged"

    def test_records_that_the_gate_did_not_fire(self):
        meta = obs.run_metadata(self._state())
        assert meta["dd.approval_triggered"] is False
        assert "dd.approval_stage" not in meta

    def test_records_the_gate_and_its_stage(self):
        meta = obs.run_metadata(self._state(
            human_review=True,
            approvals=[{"stage": "plan", "action": "revise",
                        "dropped_agents": ["sentiment"]}]))

        assert meta["dd.approval_triggered"] is True
        assert meta["dd.approval_stage"] == "plan"
        assert meta["dd.approval_actions"] == ["revise"]
        assert meta["dd.agents_dropped_by_reviewer"] == ["sentiment"]

    def test_paused_run_is_not_reported_as_converged(self):
        """A run waiting at the approval gate has not finished, it has stopped.

        Without this the paused segment reports zero rounds against a ceiling
        of one and gets labelled "converged" — exactly backwards, since it has
        not started researching yet.
        """
        meta = obs.run_metadata(self._state(research_round=0,
                                            __interrupt__=[object()]))
        assert meta["dd.status"] == "paused"
        assert meta["dd.convergence"] == "paused"

    def test_finished_run_is_marked_complete(self):
        assert obs.run_metadata(self._state())["dd.status"] == "complete"

    def test_records_tokens_by_tier(self):
        cb = obs.UsageByTierCallback()
        cb.on_chat_model_start({}, [], run_id="r1",
                               metadata={"routing_tier": "fast"}, tags=None)
        cb.on_llm_end(_llm_result(100, 10), run_id="r1")

        meta = obs.run_metadata(self._state(), cb)
        assert meta["dd.tokens_by_tier"]["fast"]["total"] == 110
        assert meta["dd.total_tokens"] == 110

    def test_record_writes_onto_the_open_root_run(self, recording_client):
        """Metadata goes onto the live run tree, not via a second update.

        LangSmith answers a second update to a finished run with 409, so this
        has to land while the run is still open. Asserting on the run tree is
        asserting on exactly what the tracer will send when it closes.
        """
        with obs.trace_run("TestCo") as handle:
            handle.record(self._state(evaluation=_evaluation(),
                                      refine_stop_reason="round_ceiling"))
            written = handle.run_tree.extra["metadata"]

        assert written["dd.planning_iterations"] == 2
        assert written["dd.convergence"] == "round_ceiling"
        # The metadata set when the run was opened survives the merge.
        assert written["dd.company"] == "TestCo"

    def test_record_is_safe_without_a_run_tree(self):
        """A handle whose root run could not be opened still reports locally."""
        handle = obs.start_trace("TestCo")
        meta = handle.record(self._state())
        assert meta["dd.planning_iterations"] == 2

    def test_resumed_run_is_renamed_once_state_reveals_the_company(
            self, recording_client):
        with obs.trace_run("resumed abc123", "t-1", stage="resume") as handle:
            handle.record(self._state(company="Stripe"))
            assert handle.run_tree.name == "due-diligence (resumed): Stripe"

    def test_finalize_pushes_feedback(self, recording_client):
        handle = obs.start_trace("TestCo")
        obs.finalize(handle, self._state(evaluation=_evaluation()))
        assert recording_client.feedback, "feedback should be pushed"


# ---------------------------------------------------------------------------
# 7. Fail-open
# ---------------------------------------------------------------------------

class TestFailOpen:
    """Requirement: no exception from tracing may reach the pipeline.

    Each case breaks a different link in the chain. The assertion is always the
    same — the call returns normally and the pipeline keeps its result.
    """

    def test_decorator_swallows_and_returns_default(self):
        @obs.fail_open(default="fallback")
        def boom():
            raise RuntimeError("nope")

        assert boom() == "fallback"

    def test_decorator_builds_a_fresh_mutable_default(self):
        """Callers must not share one dict between failures."""
        @obs.fail_open(default=dict)
        def boom():
            raise RuntimeError("nope")

        first = boom()
        first["x"] = 1
        assert boom() == {}

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
    def test_interrupts_still_propagate(self, exc):
        """Swallowing these would make a run unkillable."""
        @obs.fail_open()
        def boom():
            raise exc()

        with pytest.raises(exc):
            boom()

    def test_client_construction_failure_yields_none(self, monkeypatch,
                                                     tracing_on):
        def explode():
            raise RuntimeError("bad key")
        monkeypatch.setattr(obs, "tracing_enabled", explode)
        obs.reset_client()
        assert obs.get_client() is None

    def test_feedback_errors_do_not_propagate(self, monkeypatch, tracing_on):
        client = ExplodingClient()
        monkeypatch.setattr(obs, "get_client", lambda: client)

        assert obs.push_feedback("run-1", {"evaluation": _evaluation()}) == 0
        assert client.calls == 5, "each score tried on its own"

    def test_one_bad_score_does_not_lose_the_others(self, monkeypatch,
                                                    tracing_on):
        class Picky(RecordingClient):
            def create_feedback(self, run_id, key=None, **kwargs):
                if key == "groundedness":
                    raise RuntimeError("rejected")
                super().create_feedback(run_id, key=key, **kwargs)

        client = Picky()
        monkeypatch.setattr(obs, "get_client", lambda: client)

        assert obs.push_feedback("run-1", {"evaluation": _evaluation()}) == 4

    def test_update_run_errors_do_not_propagate(self, monkeypatch, tracing_on):
        monkeypatch.setattr(obs, "get_client", lambda: ExplodingClient())
        handle = obs.start_trace("TestCo")
        meta = obs.finalize(handle, {"research_round": 1, "approvals": []})
        assert meta["dd.planning_iterations"] == 1

    def test_callback_errors_do_not_propagate(self):
        """A raising callback would surface inside the model call itself."""
        cb = obs.UsageByTierCallback()

        class Hostile:
            @property
            def generations(self):
                raise RuntimeError("boom")

        cb.on_chat_model_start({}, [], run_id="r1",
                               metadata={"routing_tier": "fast"}, tags=None)
        cb.on_llm_end(Hostile(), run_id="r1")  # must not raise
        assert cb.totals() == {}

    def test_bounded_returns_the_value_when_work_finishes(self):
        assert obs._bounded(lambda: "done", 5.0) == "done"

    def test_bounded_swallows_errors(self):
        def boom():
            raise RuntimeError("nope")
        assert obs._bounded(boom, 5.0) is None

    def test_bounded_abandons_work_that_overruns(self):
        """The half of fail-open that catching exceptions does not cover.

        An unreachable LangSmith does not raise, it stalls. Without this bound
        a finished report would wait on telemetry that is still retrying.
        """
        started = threading.Event()

        def stall():
            started.set()
            time.sleep(30)
            return "should never be seen"

        elapsed = time.monotonic()
        result = obs._bounded(stall, 0.2)
        elapsed = time.monotonic() - elapsed

        assert started.wait(5), "work should actually have started"
        assert result is None
        assert elapsed < 5, f"returned in {elapsed:.1f}s, should be ~0.2s"

    def test_finalize_returns_promptly_when_the_client_hangs(self,
                                                             monkeypatch,
                                                             tracing_on):
        """The end-to-end version: a hanging client costs seconds, not minutes."""
        class HangingClient:
            def update_run(self, *a, **k):
                time.sleep(30)

            def create_feedback(self, *a, **k):
                time.sleep(30)

        monkeypatch.setattr(obs, "get_client", lambda: HangingClient())
        monkeypatch.setattr(obs, "FINALIZE_TIMEOUT_SECONDS", 0.2)

        handle = obs.start_trace("TestCo")
        started = time.monotonic()
        meta = obs.finalize(handle, {"research_round": 1, "approvals": []})
        elapsed = time.monotonic() - started

        assert elapsed < 5, f"finalize blocked for {elapsed:.1f}s"
        # The metadata is computed locally, so it survives the timeout.
        assert meta["dd.planning_iterations"] == 1

    def test_abandoned_work_runs_on_a_daemon_thread(self):
        """Otherwise an abandoned request would still block process exit."""
        names = []

        def capture():
            names.append(threading.current_thread().daemon)
            time.sleep(5)

        obs._bounded(capture, 0.2)
        assert names == [True]

    def test_broken_state_does_not_break_metadata(self):
        class Hostile(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError("boom")

        assert obs.run_metadata(Hostile()) == {}

    def test_annotate_never_raises(self, monkeypatch, tracing_on):
        def explode():
            raise RuntimeError("boom")
        monkeypatch.setattr(obs, "tracing_enabled", explode)
        assert obs.annotate_current_run(name="x") is None


class TestPipelineSurvivesBrokenTracing:
    """The end-to-end version of the rule: a dead LangSmith costs nothing."""

    @pytest.fixture(autouse=True)
    def _stub_pipeline(self, monkeypatch):
        # No refinement lap, so the findings count below is exactly the four
        # dispatched agents rather than four plus however many gaps round two
        # decided to chase.
        monkeypatch.setattr(ga, "MAX_REFINEMENT_ROUNDS", 0)

        class FakeAgent:
            def __init__(self, tools): self.tools = tools
            async def ainvoke(self, payload, config=None):
                class M: type, content, tool_calls = "ai", "stub", []
                return {"messages": [M()]}

        monkeypatch.setattr(base, "_compiled_agent",
                            lambda n, t, sp, tier: FakeAgent(t))

        async def fake_classify(company):
            return Classification(company_type=CompanyType.PRIVATE)
        monkeypatch.setattr(g, "classify_company", fake_classify)

        async def no_claims(block, subject): return []
        monkeypatch.setattr(ev, "extract_claims", no_claims)

        async def no_tensions(subject, claims): return []
        monkeypatch.setattr(ev, "detect_tensions", no_tensions)

    def test_run_completes_when_every_client_call_raises(self, monkeypatch,
                                                         tracing_on):
        monkeypatch.setattr(obs, "get_client", lambda: ExplodingClient())

        state = asyncio.run(g.run("TestCo"))

        assert len(state["findings"]) == 4
        assert state["evaluation"] is not None, "the report kept its score"

    def test_run_completes_when_the_client_cannot_be_built(self, monkeypatch,
                                                           tracing_on):
        def explode():
            raise RuntimeError("no key")
        monkeypatch.setattr(obs, "get_client", explode)

        state = asyncio.run(g.run("TestCo"))
        assert len(state["findings"]) == 4

    def test_run_completes_when_finalize_itself_explodes(self, monkeypatch,
                                                         tracing_on):
        def explode(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(obs, "run_metadata", explode)
        monkeypatch.setattr(obs, "get_client", lambda: RecordingClient())

        state = asyncio.run(g.run("TestCo"))
        assert len(state["findings"]) == 4

    def test_pipeline_errors_are_not_masked_by_tracing(self, monkeypatch,
                                                       tracing_on):
        """finalize() runs in a `finally`, and must not swallow the real error.

        The failure is provoked with an empty company, which makes the real
        ``planner_node`` raise — rather than by patching a node, because an
        unattended run uses the pre-compiled module-level ``app`` and would not
        see the patch. The error that surfaces must be the pipeline's own, not
        one from the exploding client that finalize() is talking to.
        """
        monkeypatch.setattr(obs, "get_client", lambda: ExplodingClient())

        with pytest.raises(ValueError, match="No company name"):
            asyncio.run(g.run("   "))

    def test_traced_run_publishes_the_evaluator_scores(self, recording_client):
        """The whole chain, end to end, on the real graph.

        A run's own groundedness evaluator produces a score inside the graph,
        and it comes back out as LangSmith feedback on that run — without this
        file computing any of it.
        """
        state = asyncio.run(g.run("TestCo"))

        keys = {f["key"] for f in recording_client.feedback}
        assert "groundedness" in keys
        assert any(k.startswith("groundedness:") for k in keys)

        # The score pushed is the one the evaluator actually produced.
        pushed = next(f for f in recording_client.feedback
                      if f["key"] == "groundedness")
        assert pushed["score"] == state["evaluation"].score

        # And all of it hangs off one run id.
        assert len({f["run_id"] for f in recording_client.feedback}) == 1

    def test_traced_run_records_loop_and_approval_metadata(self, monkeypatch,
                                                           recording_client):
        """The end-of-run facts land on the root run, via the live run tree."""
        seen = {}
        real = obs.TraceHandle.record

        def capture(handle, state):
            meta = real(handle, state)
            if handle.run_tree is not None:
                seen["metadata"] = dict(handle.run_tree.extra["metadata"])
            seen["returned"] = meta
            return meta

        monkeypatch.setattr(obs.TraceHandle, "record", capture)
        asyncio.run(g.run("TestCo"))

        meta = seen.get("metadata") or seen["returned"]
        assert meta["dd.planning_iterations"] == 1
        assert meta["dd.convergence"] == "round_ceiling"
        assert meta["dd.approval_triggered"] is False
        assert meta["dd.status"] == "complete"
        assert "dd.tokens_by_tier" in meta

    def test_human_reviewed_run_groups_its_segments_by_thread(
            self, monkeypatch, recording_client):
        """The answer to "one run per execution" when there are two invocations.

        A reviewed execution is ``run()`` plus a ``resume()``, each its own
        root run. They are stitched together by thread id rather than forced
        into one run — so the assertion is that both segments carry the same
        thread, and that the segment which actually finished is the one
        carrying the scores.
        """
        import contextlib

        seen = []
        real_trace_run = obs.trace_run

        @contextlib.contextmanager
        def recording_trace_run(company, thread_id=None, stage="run"):
            with real_trace_run(company, thread_id, stage) as handle:
                seen.append({"stage": stage,
                             "thread": handle.metadata.get("session_id"),
                             "run_id": handle.run_id})
                yield handle

        monkeypatch.setattr(obs, "trace_run", recording_trace_run)
        monkeypatch.setattr(g, "trace_run", recording_trace_run)

        paused = asyncio.run(g.run("TestCo", human_review=True))
        assert g.pending_review(paused) is not None, "should be waiting"

        final = asyncio.run(g.resume({"action": "approve"},
                                     paused["__thread_id__"]))

        assert [s["stage"] for s in seen] == ["run", "resume"]
        assert seen[0]["thread"] == seen[1]["thread"] == paused["__thread_id__"]
        assert seen[0]["run_id"] != seen[1]["run_id"], "separate root runs"

        # Feedback belongs to the segment that reached evaluation.
        assert final["evaluation"] is not None
        scored = {f["run_id"] for f in recording_client.feedback
                  if f["key"] == "groundedness"}
        assert scored == {seen[1]["run_id"]}

    def test_approval_is_recorded_as_structured_state(self, monkeypatch,
                                                      recording_client):
        """Requirement 4's "at which stage", read from data rather than prose."""
        paused = asyncio.run(g.run("TestCo", human_review=True))
        final = asyncio.run(g.resume({"action": "revise",
                                      "drop_agents": ["sentiment"]},
                                     paused["__thread_id__"]))

        assert final["approvals"] == [{"stage": "plan", "action": "revise",
                                       "dropped_agents": ["sentiment"]}]

        meta = recording_client.updates[-1]["extra"]["metadata"]
        assert meta["dd.approval_triggered"] is True
        assert meta["dd.approval_stage"] == "plan"
        assert meta["dd.agents_dropped_by_reviewer"] == ["sentiment"]

    def test_tracing_off_adds_no_client_calls(self, monkeypatch):
        """The default path must not touch LangSmith at all."""
        client = RecordingClient()
        monkeypatch.setattr(obs, "get_client", lambda: client)

        asyncio.run(g.run("TestCo"))

        assert client.feedback == []
        assert client.updates == []
