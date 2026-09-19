"""Deciding whether the evidence is good enough, and what to ask if not.

This is the node that makes the graph a loop rather than a line. It sits
between evidence and synthesis, and it answers one question: *is what we have
worth writing up, or is there a specific thing worth paying for another round
of research to find?*

**Why this is not the synthesizer's job.** The synthesizer writes prose. Giving
a prose-writer the authority to spend another few dollars conflates two
decisions that fail differently — "is the evidence sufficient" is a question
about coverage and confidence, and it is answerable with arithmetic. Keeping it
here means the spend decision never depends on a model's mood.

**Why the loop terminates.** Three independent guarantees, any one of which
would be enough:

1. ``research_round`` increments on every pass and refinement is refused past
   ``MAX_REFINEMENT_ROUNDS`` — a hard ceiling on laps.
2. ``attempted_gaps`` grows monotonically and already-attempted gaps are
   filtered out, so the candidate set strictly shrinks each lap.
3. The gap space itself is finite — bounded by the core predicate table times
   the number of sections.

**Why it converges rather than just repeating.** Each dispatched task carries
the gap's own question as its focus, so round two asks something narrower than
round one did. A refinement loop that re-runs the original prompt is not a
loop, it is a retry: same prompt, same tools, same answer, twice the cost.
"""

import logging
import os

from src.gaps.detect import RESEARCH_AGENTS, find_gaps
from src.gaps.models import Gap
from src.observability import annotate_current_run
from src.planning.models import CompanyType, ResearchPlan, ResearchTask
from src.state import DueDiligenceState

logger = logging.getLogger(__name__)

#: How many times the graph may go back for more research. One extra lap buys
#: most of the benefit; the second rarely changes a verdict and doubles cost.
MAX_REFINEMENT_ROUNDS = int(os.getenv("DD_MAX_REFINEMENT_ROUNDS", "1"))

#: Tasks dispatched per refinement round. A budget, not a target.
MAX_TASKS_PER_ROUND = int(os.getenv("DD_MAX_REFINE_TASKS", "3"))

#: Gaps below this value of information are not worth a round trip. A gap that
#: would not move the verdict is not worth money, however unresolved it is.
MIN_VALUE_OF_INFORMATION = float(os.getenv("DD_MIN_GAP_VOI", "0.5"))


def _task_for(gap: Gap, plan: ResearchPlan | None) -> ResearchTask:
    """Turn a gap into a narrow follow-up task.

    The follow-up inherits the plan's tool selection for that agent rather than
    getting a fresh full toolset. Otherwise refinement would quietly hand a
    private company back the market-data tool the planner deliberately removed,
    and the loop would spend its extra round re-making the original mistake.
    """
    tools: list[str] = []
    caveat = ""
    if plan is not None:
        existing = plan.tasks_for(gap.agent)
        if existing:
            tools = list(existing[0].tools)
        caveat = plan.caveat_for(gap.agent)

    return ResearchTask(
        agent=gap.agent,
        tools=tools,
        caveat=caveat,
        focus=(
            "This is a targeted follow-up. Earlier research left a specific "
            f"gap:\n\n{gap.question}\n\n"
            "Answer only that. Do not repeat the earlier broad survey."
        ),
    )


async def gap_analyzer_node(state: DueDiligenceState) -> dict:
    """LangGraph node: score the evidence and decide whether to research again."""
    subject = state["company"].strip().lower()
    plan = state.get("plan")
    company_type = plan.company_type if plan else CompanyType.UNKNOWN
    planned = tuple(plan.agents) if plan else RESEARCH_AGENTS

    round_no = int(state.get("research_round") or 0)
    attempted = set(state.get("attempted_gaps") or [])

    # Label this lap. Every round re-enters this node and so already gets its
    # own span; without the round number they are indistinguishable in the UI.
    annotate_current_run(
        name=f"planning_iteration_{round_no}",
        **{
            "dd.research_round": round_no,
            "dd.gaps_attempted_so_far": len(attempted),
            "dd.round_ceiling": MAX_REFINEMENT_ROUNDS,
        },
    )

    gaps = find_gaps(
        state.get("claims") or [], subject, company_type, planned)
    open_gaps = [g for g in gaps if g.id not in attempted]

    def stop(reason: str, code: str) -> dict:
        logger.info(
            "Round %d: proceeding to synthesis — %s (%d open gap(s))",
            round_no, reason, len(open_gaps),
        )
        return {
            "gaps": open_gaps,
            "refine_tasks": [],
            "research_round": round_no + 1,
            # Why the loop stopped, as a code rather than a sentence. It cannot
            # be inferred afterwards: this counter is incremented on both exits,
            # so a run that genuinely converged ends at the ceiling and is
            # indistinguishable from one that ran out of budget.
            "refine_stop_reason": code,
            "messages": [f"Evidence review round {round_no + 1}: {reason}."],
        }

    if round_no >= MAX_REFINEMENT_ROUNDS:
        return stop(f"refinement budget of {MAX_REFINEMENT_ROUNDS} round(s) spent",
                    "round_ceiling")

    worth_it = [
        g for g in open_gaps
        if g.value_of_information >= MIN_VALUE_OF_INFORMATION
    ][:MAX_TASKS_PER_ROUND]

    if not worth_it:
        return stop("no remaining gap would move the verdict", "converged")

    tasks = [_task_for(gap, plan) for gap in worth_it]
    logger.info(
        "Round %d: %d gap(s) worth chasing — %s",
        round_no, len(worth_it), "; ".join(g.describe() for g in worth_it),
    )

    return {
        "gaps": open_gaps,
        "refine_tasks": tasks,
        "research_round": round_no + 1,
        # Marked at dispatch, not on success: a gap we chased and still could
        # not close must not be chased again next lap.
        "attempted_gaps": [g.id for g in worth_it],
        "messages": [
            f"Evidence review round {round_no + 1}: chasing "
            f"{len(worth_it)} gap(s) — " + "; ".join(g.detail for g in worth_it)
        ],
    }
