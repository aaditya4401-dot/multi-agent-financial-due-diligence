"""What the planner decides, expressed as data.

A :class:`ResearchPlan` is the orchestrator's entire output. Making it a value
rather than a side effect buys three things:

* the dispatch step becomes a pure function of the plan, so routing is testable
  without an LLM or a network;
* the plan can be shown to a human for approval before any expensive call runs
  (phase 3), which is why :attr:`ResearchPlan.rationale` exists now rather than
  later;
* "we deliberately skipped this agent" becomes derivable, instead of being
  indistinguishable from "this agent crashed".

Tools are named by string rather than held as objects. That keeps this module
free of any dependency on ``src.agents``, so the routing rules stay importable
in a test that never constructs an LLM. The names are resolved against the
agent registry at dispatch time.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CompanyType(str, Enum):
    """What kind of target we are researching.

    ``UNKNOWN`` is a real answer, not a failure. It is what we report when the
    evidence does not settle the question, and it maps to the unrestricted
    plan — the same behaviour the system had before it could classify at all.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    """The FAST-tier model's proposal about a company. Never trusted directly.

    ``ticker`` is a *candidate*, verified against a market-data lookup before
    it is allowed to set :attr:`ResearchPlan.company_type` to ``PUBLIC``.
    """

    company_type: CompanyType = CompanyType.UNKNOWN
    ticker: str = Field(default="", description="Best-guess exchange ticker, or empty.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names the company trades or is written under.",
    )
    sector: str = ""
    rationale: str = ""


class ResearchTask(BaseModel):
    """One unit of research dispatched to one agent.

    There may be more than one task per agent: the state's findings list is
    reducer-backed, so two tasks aimed at the same agent append rather than
    collide.
    """

    agent: str
    focus: str = Field(
        default="",
        description="Extra instruction appended to the agent's base task prompt.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names this task may use. Empty means the agent's full set.",
    )
    caveat: str = Field(
        default="",
        description="Carried into the report section. Labels evidence quality "
                    "that the confidence model cannot express — e.g. that a "
                    "private company's figures are third-party estimates.",
    )


class ResearchPlan(BaseModel):
    """The set of research tasks to run, and why."""

    company: str
    company_type: CompanyType = CompanyType.UNKNOWN
    ticker: str | None = None
    rationale: str = ""
    tasks: list[ResearchTask] = Field(default_factory=list)

    @property
    def agents(self) -> list[str]:
        """Agents this plan dispatches to, in order, without duplicates."""
        seen: list[str] = []
        for task in self.tasks:
            if task.agent not in seen:
                seen.append(task.agent)
        return seen

    def skipped(self, all_agents: tuple[str, ...]) -> list[str]:
        """Agents deliberately left out of this plan.

        Distinct from agents that ran and failed. A skipped section carries no
        penalty in the report; a failed one does.
        """
        return [agent for agent in all_agents if agent not in self.agents]

    def tasks_for(self, agent: str) -> list["ResearchTask"]:
        """Every task dispatched to *agent*. Usually one; not necessarily."""
        return [t for t in self.tasks if t.agent == agent]

    def caveat_for(self, agent: str) -> str:
        """First non-empty caveat among *agent*'s tasks."""
        return next(
            (t.caveat for t in self.tasks if t.agent == agent and t.caveat), ""
        )


class PlanDecision(BaseModel):
    """A human's verdict on a proposed research plan.

    Returned through LangGraph's ``interrupt`` mechanism, so it must survive a
    JSON round trip — the reviewer may not be in the same process, or even the
    same day, as the run they are approving.
    """

    action: Literal["approve", "revise", "cancel"] = "approve"
    drop_agents: list[str] = Field(
        default_factory=list,
        description="Agents to remove before dispatch. Only meaningful for "
                    "'revise'. These become skipped sections, not failures.",
    )
    note: str = Field(
        default="",
        description="Reviewer's reasoning, carried into the report's audit trail.",
    )

    def applied_to(self, plan: ResearchPlan) -> ResearchPlan:
        """Return *plan* as the reviewer wants it run.

        ``cancel`` yields a plan with no tasks rather than a null plan, so the
        downstream graph handles it through the same path as any other empty
        plan instead of needing a special case.
        """
        if self.action == "cancel":
            return plan.model_copy(update={
                "tasks": [],
                "rationale": f"{plan.rationale} REVIEWER CANCELLED: {self.note}".strip(),
            })

        if self.action == "revise" and self.drop_agents:
            dropped = set(self.drop_agents)
            kept = [t for t in plan.tasks if t.agent not in dropped]
            note = f" REVIEWER DROPPED {', '.join(sorted(dropped))}"
            if self.note:
                note += f": {self.note}"
            return plan.model_copy(update={
                "tasks": kept,
                "rationale": plan.rationale + note,
            })

        if self.note:
            return plan.model_copy(update={
                "rationale": f"{plan.rationale} REVIEWER: {self.note}",
            })
        return plan
