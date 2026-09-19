import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from src.agents.base import research_node
from src.agents.approval import approval_node
from src.checkpointing import serializer
from src.agents.evaluation import evaluation_node
from src.agents.evidence import evidence_node
from src.agents.gap_analyzer import gap_analyzer_node
from src.agents.synthesizer import synthesizer_node
from src.agents.thesis import thesis_node
from src.observability import finalize, trace_run
from src.planning.classify import classify_company
from src.planning.plan import plan_for
from pydantic import BaseModel

from src.state import DueDiligenceState

logger = logging.getLogger(__name__)

# Generous: the graph is shallow but now cyclic (research → evidence →
# gap_analyzer → research). Loop termination is enforced by gap_analyzer's
# round ceiling; this limit is only a backstop. Per-agent tool loops are
# capped separately in agents/base.py.
GRAPH_RECURSION_LIMIT = int(os.getenv("DD_GRAPH_RECURSION_LIMIT", "50"))


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

async def planner_node(state: DueDiligenceState) -> dict:
    """Work out what this company is, and decide what research to run.

    This used to strip whitespace and log. It now produces a
    :class:`ResearchPlan`, which is what makes the fan-out below conditional
    rather than constant: the plan decides how many research tasks exist and
    which tools each one may reach.

    Classification failure is not fatal — it degrades to the unrestricted
    four-task plan, which is the behaviour the system had before it could
    classify anything.
    """
    company = state.get("company", "").strip()
    if not company:
        raise ValueError("No company name provided in state['company']")

    classification = await classify_company(company)
    plan = plan_for(company, classification)

    logger.info(
        "Plan for %s [%s%s]: %d task(s) — %s",
        company,
        plan.company_type.value,
        f" {plan.ticker}" if plan.ticker else "",
        len(plan.tasks),
        ", ".join(plan.agents),
    )
    logger.info("Rationale: %s", plan.rationale)

    return {
        "company": company,
        "plan": plan,
        "messages": [
            f"Planned {len(plan.tasks)} research task(s) for {company} "
            f"({plan.company_type.value}). {plan.rationale}"
        ],
    }


def dispatch_research(state: DueDiligenceState) -> Any:
    """Fan out one ``Send`` per planned task.

    ``Send`` rather than a fixed set of edges because the *number* of tasks is
    a runtime decision. A conditional edge chooses a path among known nodes;
    ``Send`` chooses a population. The payload must be self-contained — a node
    invoked this way receives the payload as its entire input and cannot read
    graph state.
    """
    plan = state.get("plan")
    tasks = plan.tasks if plan else []

    if not tasks:
        # Reachable once a human can deselect agents at an approval gate.
        # Synthesis will report inconclusive rather than the graph stalling.
        logger.error("Plan dispatched no research tasks — going straight to synthesis")
        return "thesis"

    return [
        Send("research", {"company": plan.company, "task": task})
        for task in tasks
    ]


def route_after_gaps(state: DueDiligenceState) -> Any:
    """Close the loop, or proceed to synthesis.

    Deliberately mechanical: every judgement about whether more research is
    worth buying already happened in ``gap_analyzer_node``, which wrote its
    decision into ``refine_tasks``. A router cannot write state, so putting the
    policy here would mean the round counter and the attempted-gap set could
    not be updated in step with the decision they guard — and a loop whose
    termination bookkeeping can drift from its termination decision is a loop
    that can run forever.
    """
    tasks = state.get("refine_tasks") or []
    if not tasks:
        return "thesis"

    company = state["company"]
    return [
        Send("research", {"company": company, "task": task})
        for task in tasks
    ]


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Wire the due diligence graph (uncompiled, so callers can add a checkpointer)."""
    workflow = StateGraph(DueDiligenceState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("research", research_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("gap_analyzer", gap_analyzer_node)
    workflow.add_node("thesis", thesis_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluation", evaluation_node)

    workflow.set_entry_point("planner")

    # The plan is reviewable before anything is spent on it. The gate is a
    # no-op unless human_review is set, so unattended runs are unaffected.
    workflow.add_edge("planner", "approval")

    # Fan-out: approval → N research tasks, decided at runtime.
    workflow.add_conditional_edges(
        "approval", dispatch_research, ["research", "thesis"]
    )

    # Fan-in: a single edge that fires once, after every dispatched task has
    # finished, whatever N was. The old join edge named all four agents and so
    # could never have tolerated one being skipped.
    workflow.add_edge("research", "evidence")

    # The cycle: evidence is reviewed, and thin evidence buys another round of
    # narrow, targeted research rather than being written up as-is.
    #
    #     research → evidence → gap_analyzer ─┬─> research  (refine)
    #                                          └─> synthesizer (good enough)
    #
    # Termination is guaranteed by gap_analyzer, not by this wiring: a round
    # ceiling, a monotonically growing attempted-gap set, and a finite gap
    # space. The recursion limit is a backstop, not the mechanism.
    workflow.add_edge("evidence", "gap_analyzer")
    workflow.add_conditional_edges(
        "gap_analyzer", route_after_gaps, ["research", "thesis"]
    )

    # A view is formed before the memo is written, so the executive summary
    # reflects the thesis rather than the two being written independently.
    workflow.add_edge("thesis", "synthesizer")

    # Every report is scored against the evidence behind it before it leaves
    # the graph. A groundedness number that only exists when someone runs a
    # benchmark says nothing about the report actually in front of the reader.
    workflow.add_edge("synthesizer", "evaluation")
    workflow.add_edge("evaluation", END)

    return workflow


workflow = build_workflow()
app = workflow.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MEMORY_SAVER: Any = None


def _memory_saver() -> Any:
    """Process-local checkpointer, reused so a resume can find its thread.

    Good enough for a CLI session or a test. Not good enough for Streamlit,
    which reruns its script on every interaction — that needs a real file.
    """
    global _MEMORY_SAVER
    if _MEMORY_SAVER is None:
        from langgraph.checkpoint.memory import InMemorySaver
        _MEMORY_SAVER = InMemorySaver(serde=serializer())
    return _MEMORY_SAVER


@asynccontextmanager
async def _checkpointed_app(checkpoint_path: str | None):
    """Compile the graph with a checkpointer.

    An interrupt *is* a checkpoint: the graph writes its whole state and stops,
    and resuming reads that state back. With no checkpointer there is nothing
    to resume from, so human review silently could not work. Hence a
    checkpointer is always provided here, falling back to memory rather than
    to none.
    """
    saver = None
    if checkpoint_path:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed — using in-memory "
                "checkpoints, which do not survive the process"
            )
        else:
            saver = AsyncSqliteSaver.from_conn_string(checkpoint_path)

    if saver is not None:
        async with saver as checkpointer:
            # from_conn_string does not forward a serializer, so set it here.
            checkpointer.serde = serializer()
            yield build_workflow().compile(checkpointer=checkpointer)
    else:
        yield build_workflow().compile(checkpointer=_memory_saver())


def pending_review(state: dict) -> dict | None:
    """The payload awaiting a human, or ``None`` if the run finished.

    LangGraph reports a paused graph by putting an ``__interrupt__`` entry in
    the returned state rather than by raising, so callers must look for it.
    """
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


async def run(
    company: str,
    thread_id: str | None = None,
    checkpoint_path: str | None = None,
    human_review: bool = False,
) -> dict:
    """Run the due diligence pipeline and return the final state.

    Args:
        company: Company to analyze.
        thread_id: Checkpoint thread id. Required to make a run resumable;
            generated automatically when human review is on.
        checkpoint_path: SQLite file for checkpoints. Defaults to
            ``DD_CHECKPOINT_PATH``. Falls back to in-process memory.
        human_review: Pause for plan approval before spending on research.

    With ``human_review`` the returned state may be *paused* rather than
    finished — check :func:`pending_review`, then call :func:`resume`.
    """
    initial_state: dict[str, Any] = {
        "company": company,
        "human_review": human_review,
        "findings": [],
        "claims": [],
        "conflicts": [],
        "extracted_upto": 0,
        "research_round": 0,
        "attempted_gaps": [],
        "thesis": None,
        "thesis_audit": None,
        "evaluation": None,
        "approvals": [],
        "messages": [],
    }
    config: dict[str, Any] = {"recursion_limit": GRAPH_RECURSION_LIMIT}
    path = checkpoint_path or os.getenv("DD_CHECKPOINT_PATH")

    # Unattended and unresumable: no checkpointer needed, so don't pay for one.
    if not human_review and not thread_id:
        # Tracing attaches here, to the config, rather than at compile time —
        # this function has two invocation paths (this one and the checkpointed
        # one below) and compile-time callbacks would cover only one of them.
        # A disabled handle leaves `config` exactly as it is.
        state = {}
        with trace_run(company) as handle:
            handle.apply_to(config)
            try:
                state = await app.ainvoke(initial_state, config=config)
            finally:
                # Inside the `with`, so the metadata lands while the root run
                # is still open — LangSmith refuses a second update once it
                # closes. In `finally` so a run that raised is still described,
                # and so tracing can never mask the pipeline's own exception.
                handle.record(state)
        # Feedback goes after the run is closed, which is fine: it is a
        # separate resource and wants the run to already exist.
        finalize(handle, state)
        return state

    thread_id = thread_id or uuid4().hex
    config["configurable"] = {"thread_id": thread_id}
    state = {}
    with trace_run(company, thread_id) as handle:
        handle.apply_to(config)
        try:
            async with _checkpointed_app(path) as graph:
                state = await graph.ainvoke(initial_state, config=config)
        finally:
            handle.record(state)
    finalize(handle, state)

    state["__thread_id__"] = thread_id
    return state


async def resume(
    decision: Any,
    thread_id: str,
    checkpoint_path: str | None = None,
) -> dict:
    """Continue a paused run with the reviewer's *decision*.

    Args:
        decision: A :class:`PlanDecision`, an equivalent dict, or a bare string
            like ``"approve"`` / ``"cancel"``.
        thread_id: The thread the paused run was checkpointed under.
        checkpoint_path: Must point at the same store the run used.
    """
    if isinstance(decision, BaseModel):
        decision = decision.model_dump()

    config: dict[str, Any] = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }
    path = checkpoint_path or os.getenv("DD_CHECKPOINT_PATH")

    # A resumed segment is its own root run — the original ended when the graph
    # interrupted, possibly in another process on another day. Passing the same
    # thread_id groups the segments in LangSmith instead of pretending they are
    # one run. Feedback lands on whichever segment actually reached evaluation,
    # which is this one if the run finishes here.
    # The company name lives in the checkpoint, and reading it back just to
    # label a span is not worth an extra round-trip — handle.record() corrects
    # the name once the returned state reveals it, while the run is still open.
    state = {}
    with trace_run(f"resumed {thread_id[:8]}", thread_id,
                   stage="resume") as handle:
        handle.apply_to(config)
        try:
            async with _checkpointed_app(path) as graph:
                state = await graph.ainvoke(Command(resume=decision),
                                            config=config)
        finally:
            handle.record(state)
    finalize(handle, state)

    state["__thread_id__"] = thread_id
    return state


def _console_review(payload: dict) -> dict:
    """Render a pending plan on stdout and collect a decision."""
    print("\n" + "=" * 68)
    print(f"  PLAN REVIEW: {payload.get('company')}")
    print("=" * 68)
    kind = payload.get("company_type")
    ticker = payload.get("ticker")
    print(f"  Classified : {kind}{f' ({ticker})' if ticker else ''}")
    print(f"  Rationale  : {payload.get('rationale')}")
    print(f"\n  Will run {len(payload.get('tasks', []))} research agent(s):")
    for task in payload.get("tasks", []):
        print(f"    - {task['agent']:10} tools: {', '.join(task['tools']) or 'all'}")
        if task.get("caveat"):
            print(f"      caveat: {task['caveat'][:70]}")
    print("=" * 68)
    print("  [enter] approve   |   drop <agents>   |   cancel")

    reply = input("  > ").strip()
    if not reply:
        return {"action": "approve"}
    if reply.lower().startswith("cancel"):
        return {"action": "cancel", "note": reply[6:].strip()}
    if reply.lower().startswith("drop"):
        agents = [a.strip(" ,") for a in reply[4:].split() if a.strip(" ,")]
        return {"action": "revise", "drop_agents": agents}
    return {"action": "approve", "note": reply}


async def _cli(target: str, review: bool) -> dict:
    """Run to completion, pausing at the console for each review gate."""
    state = await run(target, human_review=review)
    while (payload := pending_review(state)) is not None:
        decision = await asyncio.to_thread(_console_review, payload)
        state = await resume(decision, state["__thread_id__"])
    return state


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run due diligence on a company.")
    parser.add_argument("company", nargs="?", default="Stripe")
    parser.add_argument("--review", action="store_true",
                        help="Pause for plan approval before spending.")
    parser.add_argument("--save", metavar="PATH",
                        help="Write the report JSON here, for scripts/evaluate.py.")
    parsed = parser.parse_args()

    target, review = parsed.company, parsed.review
    state = asyncio.run(_cli(target, review))

    report = state.get("final_report", {})
    print("\n" + "=" * 60)
    print(f"  DUE DILIGENCE REPORT: {report.get('company_name', target)}")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    if parsed.save:
        path = Path(parsed.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {path}")
