"""The human approval gate.

Due diligence is exactly the shape of task where a person should see the plan
before the machine spends real money on it: the research is expensive, the
target may simply be the wrong company, and a reviewer often knows in one
glance that a whole line of enquiry is pointless.

**Why this is its own node rather than an interrupt at the end of the planner.**
A node that interrupts does not resume *after* the interrupt — it re-executes
from the top, and ``interrupt()`` returns the reviewer's answer the second time
through. Anything above that call therefore runs twice. Putting the gate inside
``planner_node`` would pay for classification twice per approval. So this node
contains nothing but the interrupt and the cheap bookkeeping around it, and
every expensive thing lives in a node of its own.

That also means this node is safe to re-run, which is the property LangGraph
actually requires of an interrupting node.

**Why an interrupt rather than stopping and restarting the process.** An
interrupt is a checkpoint: the graph's whole state is durably written, and the
run resumes from that point — possibly in a different process, possibly the
next day. Without a checkpointer there is nothing to resume from, which is why
``run(human_review=True)`` compiles a checkpointer whether or not one was asked
for.
"""

import logging
from typing import Any

from langgraph.types import interrupt

from src.planning.models import PlanDecision, ResearchPlan
from src.state import DueDiligenceState

logger = logging.getLogger(__name__)

#: Which gate this is. There is exactly one today — before any research is
#: dispatched, which is the last moment a reviewer can stop the spending — but
#: the graph's resume loop already tolerates several, so the stage travels with
#: the decision rather than being assumed by whoever reads it back.
APPROVAL_STAGE = "plan"


def _as_decision(raw: Any) -> PlanDecision:
    """Interpret whatever came back through the interrupt.

    Deliberately forgiving: the value crosses a process boundary and may be
    typed by a UI, a test, or a person. Anything unintelligible is treated as
    approval of the plan as proposed — the reviewer was shown it, and a
    malformed answer is not a reason to throw away a run.
    """
    if isinstance(raw, PlanDecision):
        return raw
    if isinstance(raw, dict):
        try:
            return PlanDecision(**raw)
        except Exception:
            logger.warning("Unreadable plan decision %r — treating as approval", raw)
            return PlanDecision()
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"cancel", "reject", "abort", "no"}:
            return PlanDecision(action="cancel", note=raw)
        return PlanDecision(note="" if text in {"approve", "yes", "ok"} else raw)
    return PlanDecision()


def review_payload(plan: ResearchPlan) -> dict:
    """What the reviewer is shown. The decision-relevant facts, nothing else."""
    return {
        "question": (
            f"Approve this research plan for {plan.company}? "
            f"It will run {len(plan.tasks)} research agent(s)."
        ),
        "company": plan.company,
        "company_type": plan.company_type.value,
        "ticker": plan.ticker,
        "rationale": plan.rationale,
        "tasks": [
            {
                "agent": task.agent,
                "tools": task.tools,
                "focus": task.focus,
                "caveat": task.caveat,
            }
            for task in plan.tasks
        ],
        "respond_with": {
            "action": "approve | revise | cancel",
            "drop_agents": "[agent names] — only for revise",
            "note": "optional reasoning, kept in the audit trail",
        },
    }


async def approval_node(state: DueDiligenceState) -> dict:
    """LangGraph node: pause for human review of the plan, if enabled.

    A no-op unless ``human_review`` is set, so the CLI, the tests and any
    unattended run keep working unchanged.
    """
    plan = state.get("plan")

    if not state.get("human_review") or plan is None:
        return {}

    # Nothing expensive above this line: it runs a second time on resume.
    decision = _as_decision(interrupt(review_payload(plan)))
    revised = decision.applied_to(plan)

    dropped = sorted({t.agent for t in plan.tasks} - {t.agent for t in revised.tasks})
    logger.info(
        "Reviewer chose %r%s%s",
        decision.action,
        f", dropping {', '.join(dropped)}" if dropped else "",
        f" — {decision.note}" if decision.note else "",
    )

    return {
        "plan": revised,
        # The same event twice, for two different readers. `messages` is the
        # human-readable audit trail; `approvals` is the machine-readable one
        # that tracing reports as run metadata. Keeping them separate means the
        # sentence above stays free to be reworded without breaking anything.
        "approvals": [{
            "stage": APPROVAL_STAGE,
            "action": decision.action,
            "dropped_agents": dropped,
        }],
        "messages": [
            f"Human review: {decision.action}"
            + (f", dropped {', '.join(dropped)}" if dropped else "")
            + (f" ({decision.note})" if decision.note else "")
        ],
    }
