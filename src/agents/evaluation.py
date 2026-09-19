"""Scoring the report against the evidence, on every run.

The last node in the graph, and deliberately not an optional offline script.
A groundedness score that only exists when someone remembers to run a benchmark
tells you nothing about the report in front of you; attached to the report, it
is a number the reader can act on.

All deterministic. The judgement-based half of evaluation — is the thesis
falsifiable, does the verdict follow — lives in :mod:`src.eval.judge` and runs
offline over a fixture set, because it costs a model call and answers a
different question.
"""

import logging

from src.eval.groundedness import evaluate
from src.state import DueDiligenceState

logger = logging.getLogger(__name__)


async def evaluation_node(state: DueDiligenceState) -> dict:
    """LangGraph node: attach a groundedness score to the finished report."""
    report = state.get("final_report") or {}
    claims = state.get("claims") or []
    audit = state.get("thesis_audit")

    try:
        result = evaluate(report, claims, audit)
    except Exception:
        # Evaluation is the last thing that should be able to lose a report.
        logger.exception("Evaluation failed — returning the report unscored")
        return {}

    logger.info("Evaluation: %s", result.describe())
    return {
        "evaluation": result,
        "final_report": {**report, "evaluation": result.model_dump()},
    }
