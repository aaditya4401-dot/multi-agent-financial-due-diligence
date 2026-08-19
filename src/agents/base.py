"""The research node: one task in, one findings block out.

There used to be four nodes, each built at import time around a fixed toolset.
The planner decides toolsets per run, so there is now a single node that
executes whichever :class:`ResearchTask` it is handed, and the graph invokes it
once per task via ``Send``.

Three things worth knowing about this node:

* **Its input is the Send payload, not graph state.** LangGraph passes a
  ``Send`` payload as the node's entire input — ``state["company"]`` is not
  readable here. Everything a task needs must travel in the payload.
* **It writes to a reducer-backed list.** Two parallel writes to a single state
  key raise ``InvalidUpdateError``, so tasks append to ``findings`` rather than
  owning a key each. That is what allows two tasks aimed at the same agent.
* **It never raises.** A failed agent returns ``ok=False`` findings so the
  synthesizer can report that section as unknown, rather than taking the whole
  pipeline down with it.

Compiled agents are cached per (agent, toolset). Caching on the agent alone
would return an agent built with the previous run's tools.
"""

import logging
import os
from typing import Any

from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from src.agents.registry import AGENT_SPECS
from src.agents.utils import error_findings, parse_react_output
from src.llm import get_llm
from src.planning.models import ResearchTask

logger = logging.getLogger(__name__)

# LangGraph counts super-steps, and a react loop burns roughly two per
# tool call (model turn + tool turn), so this is ~10 tool calls.
DEFAULT_MAX_STEPS = int(os.getenv("DD_AGENT_MAX_STEPS", "20"))

#: (agent name, tool names) → compiled agent.
_COMPILED: dict[tuple[str, tuple[str, ...]], Any] = {}


def _as_task(raw: Any) -> ResearchTask:
    """Accept a ResearchTask or its serialised form.

    A checkpointer round-trips state through JSON, so a resumed run hands this
    node a plain dict where a fresh run hands it a model.
    """
    if isinstance(raw, ResearchTask):
        return raw
    if isinstance(raw, dict):
        return ResearchTask(**raw)
    raise TypeError(f"cannot interpret {type(raw).__name__} as a ResearchTask")


def _compiled_agent(agent_name: str, tools: list[Any], system_prompt: str, tier) -> Any:
    key = (agent_name, tuple(sorted(t.name for t in tools)))
    if key not in _COMPILED:
        _COMPILED[key] = create_agent(
            model=get_llm(tier),
            tools=tools,
            system_prompt=system_prompt,
        )
    return _COMPILED[key]


async def research_node(payload: dict) -> dict:
    """LangGraph node: run one research task and append its findings.

    Args:
        payload: The ``Send`` payload — ``{"company": str, "task": ResearchTask}``.
            This is the node's whole input; graph state is not visible.

    Returns:
        ``{"findings": [AgentFindings]}``, appended by the state reducer.
    """
    company = payload["company"]
    task = _as_task(payload["task"])
    max_steps = int(payload.get("max_steps") or DEFAULT_MAX_STEPS)

    spec = AGENT_SPECS.get(task.agent)
    if spec is None:
        logger.error("Plan named unknown agent %r — skipping", task.agent)
        return {"findings": [
            error_findings(task.agent, ValueError(f"unknown agent {task.agent!r}"))
        ]}

    tools = spec.resolve_tools(task.tools)
    logger.info(
        "%s agent researching %r with %s",
        spec.name, company, [t.name for t in tools],
    )

    try:
        agent = _compiled_agent(spec.name, tools, spec.system_prompt, spec.tier)
        result = await agent.ainvoke(
            {"messages": [
                {"role": "user", "content": spec.compose_task(company, task.focus)}
            ]},
            config={"recursion_limit": max_steps},
        )
        findings = parse_react_output(spec.name, result)

    except GraphRecursionError:
        # The agent ran out of steps. Whatever it gathered is lost, but this is
        # a budget problem, not a failure of the pipeline.
        logger.warning(
            "%s agent hit the %d-step cap for %r — returning no findings",
            spec.name, max_steps, company,
        )
        findings = error_findings(
            spec.name, RuntimeError(f"exceeded {max_steps}-step budget")
        )

    except Exception as exc:
        logger.exception("%s agent failed for %r", spec.name, company)
        findings = error_findings(spec.name, exc)

    if task.caveat:
        findings["caveat"] = task.caveat
    return {"findings": [findings]}
