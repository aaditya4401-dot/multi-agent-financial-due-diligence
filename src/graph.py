import asyncio
import json
import logging
import os
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.conflict_resolver import conflict_resolver_node
from src.agents.financial import financial_agent
from src.agents.market import market_agent
from src.agents.risk import risk_agent
from src.agents.sentiment import sentiment_agent
from src.agents.synthesizer import synthesizer_node
from src.state import DueDiligenceState

logger = logging.getLogger(__name__)

# Generous: the graph itself is shallow (orchestrator → agents → resolver →
# synthesizer). Per-agent tool loops are capped separately in agents/base.py.
GRAPH_RECURSION_LIMIT = int(os.getenv("DD_GRAPH_RECURSION_LIMIT", "50"))


# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

async def orchestrator_node(state: DueDiligenceState) -> dict:
    """Validate input and initialise state fields before fanning out."""
    company = state.get("company", "").strip()
    if not company:
        raise ValueError("No company name provided in state['company']")

    logger.info("Starting due diligence for: %s", company)
    return {
        "company": company,
        "messages": [f"Starting due diligence analysis for {company}"],
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Wire the due diligence graph (uncompiled, so callers can add a checkpointer)."""
    workflow = StateGraph(DueDiligenceState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("financial", financial_agent)
    workflow.add_node("market", market_agent)
    workflow.add_node("risk", risk_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("conflict_resolver", conflict_resolver_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("orchestrator")

    # Fan-out: orchestrator → all 4 agents in parallel
    for agent in ("financial", "market", "risk", "sentiment"):
        workflow.add_edge("orchestrator", agent)

    # Fan-in: all 4 agents → conflict resolver
    workflow.add_edge(["financial", "market", "risk", "sentiment"], "conflict_resolver")

    # Linear: resolver → synthesizer → END
    workflow.add_edge("conflict_resolver", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow


workflow = build_workflow()
app = workflow.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(company: str, thread_id: str | None = None, checkpoint_path: str | None = None) -> dict:
    """Run the full due diligence pipeline and return the final state.

    Args:
        company: Company to analyze.
        thread_id: Checkpoint thread id. Required to make a run resumable.
        checkpoint_path: SQLite file for checkpoints. Defaults to
            ``DD_CHECKPOINT_PATH``, or in-memory-only when unset.

    Passing a ``thread_id`` (and having ``langgraph-checkpoint-sqlite``
    installed) means a crashed or interrupted run resumes from the last
    completed node instead of re-paying for every agent.
    """
    initial_state: dict[str, Any] = {"company": company, "conflicts": [], "messages": []}
    config: dict[str, Any] = {"recursion_limit": GRAPH_RECURSION_LIMIT}

    path = checkpoint_path or os.getenv("DD_CHECKPOINT_PATH")
    if thread_id and path:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed — running without checkpoints"
            )
        else:
            config["configurable"] = {"thread_id": thread_id}
            async with AsyncSqliteSaver.from_conn_string(path) as checkpointer:
                checkpointed = build_workflow().compile(checkpointer=checkpointer)
                return await checkpointed.ainvoke(initial_state, config=config)

    return await app.ainvoke(initial_state, config=config)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    target = sys.argv[1] if len(sys.argv) > 1 else "Stripe"
    state = asyncio.run(run(target))

    report = state.get("final_report", {})
    print("\n" + "=" * 60)
    print(f"  DUE DILIGENCE REPORT: {report.get('company_name', target)}")
    print("=" * 60)
    print(json.dumps(report, indent=2))
