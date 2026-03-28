import asyncio
import json
import logging

from langgraph.graph import StateGraph, END

from src.state import DueDiligenceState
from src.agents.financial import financial_agent
from src.agents.market import market_agent
from src.agents.risk import risk_agent
from src.agents.sentiment import sentiment_agent
from src.agents.conflict_resolver import conflict_resolver_node
from src.agents.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


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

workflow = StateGraph(DueDiligenceState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("financial", financial_agent)
workflow.add_node("market", market_agent)
workflow.add_node("risk", risk_agent)
workflow.add_node("sentiment", sentiment_agent)
workflow.add_node("conflict_resolver", conflict_resolver_node)
workflow.add_node("synthesizer", synthesizer_node)

# Entry point
workflow.set_entry_point("orchestrator")

# Fan-out: orchestrator → all 4 agents in parallel
workflow.add_edge("orchestrator", "financial")
workflow.add_edge("orchestrator", "market")
workflow.add_edge("orchestrator", "risk")
workflow.add_edge("orchestrator", "sentiment")

# Fan-in: all 4 agents → conflict resolver
workflow.add_edge(["financial", "market", "risk", "sentiment"], "conflict_resolver")

# Linear: resolver → synthesizer → END
workflow.add_edge("conflict_resolver", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile
app = workflow.compile()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def run(company: str) -> dict:
    """Run the full due diligence pipeline and return the final state."""
    initial_state = {
        "company": company,
        "conflicts": [],
        "messages": [],
    }
    result = await app.ainvoke(initial_state)
    return result


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    company = sys.argv[1] if len(sys.argv) > 1 else "Stripe"
    state = asyncio.run(run(company))

    report = state.get("final_report", {})
    print("\n" + "=" * 60)
    print(f"  DUE DILIGENCE REPORT: {report.get('company_name', company)}")
    print("=" * 60)
    print(json.dumps(report, indent=2))
