"""Factory for research agent nodes.

The four research agents (financial, market, risk, sentiment) differ only in
their prompt, their tools, and the state key they write to. This factory holds
the shared machinery so each agent module is just configuration.

Two things the old per-agent code got wrong and this fixes:

* The agent graph was rebuilt on every invocation. It is now built once,
  lazily, and reused.
* Tool loops were unbounded. Each agent now runs under a step cap, and hitting
  that cap degrades to partial findings instead of raising.
"""

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from src.agents.utils import error_findings, parse_react_output
from src.llm import Tier, get_llm
from src.state import DueDiligenceState

logger = logging.getLogger(__name__)

# LangGraph counts super-steps, and a react loop burns roughly two per
# tool call (model turn + tool turn), so this is ~10 tool calls.
DEFAULT_MAX_STEPS = int(os.getenv("DD_AGENT_MAX_STEPS", "20"))


def make_research_agent(
    *,
    name: str,
    state_key: str,
    system_prompt: str,
    tools: Sequence[Any],
    task_prompt: str,
    tier: Tier = Tier.REASONING,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Callable:
    """Build a LangGraph node that runs one research agent.

    Args:
        name: Agent name, used in findings and logs (e.g. ``"financial"``).
        state_key: State field the node writes to (e.g. ``"financial_findings"``).
        system_prompt: The agent's persona and standing instructions.
        tools: Tools available to this agent.
        task_prompt: The task, containing a ``{company}`` placeholder.
        tier: Which model tier to run on.
        max_steps: Recursion limit for the agent's tool loop.

    Returns:
        An async callable suitable for ``workflow.add_node``.
    """
    compiled: list[Any] = []  # single-slot lazy cache

    def _agent():
        if not compiled:
            compiled.append(
                create_agent(
                    model=get_llm(tier),
                    tools=list(tools),
                    system_prompt=system_prompt,
                )
            )
        return compiled[0]

    async def node(state: DueDiligenceState) -> dict:
        company = state["company"]

        try:
            result = await _agent().ainvoke(
                {"messages": [{"role": "user", "content": task_prompt.format(company=company)}]},
                config={"recursion_limit": max_steps},
            )
            findings = parse_react_output(name, result)

        except GraphRecursionError:
            # The agent ran out of steps. Whatever it gathered is lost, but
            # this is a budget problem, not a failure of the pipeline.
            logger.warning(
                "%s agent hit the %d-step cap for %r — returning no findings",
                name, max_steps, company,
            )
            findings = error_findings(
                name, RuntimeError(f"exceeded {max_steps}-step budget")
            )

        except Exception as exc:
            logger.exception("%s agent failed for %r", name, company)
            findings = error_findings(name, exc)

        return {state_key: findings}

    node.__name__ = f"{name}_agent"
    node.__doc__ = f"LangGraph node: run the {name} research agent."
    return node
