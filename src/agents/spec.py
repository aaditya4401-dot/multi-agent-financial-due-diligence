"""What a research agent *is*, separated from how it runs.

Each agent module used to build its own graph node at import time, binding one
fixed toolset forever. The planner needs the toolset chosen per run, so an
agent is now described as data — prompts plus an available toolset — and a
single node in :mod:`src.agents.base` executes whichever description it is
handed.

This module deliberately imports nothing from :mod:`src.agents.base`, so the
agent modules can depend on it without a cycle.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm import Tier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    """One research agent's persona, task, and the tools it *may* use."""

    name: str
    system_prompt: str
    task_prompt: str                      # contains a {company} placeholder
    tools: tuple[Any, ...]                # every tool this agent can reach
    tier: Tier = Tier.REASONING

    def tool_map(self) -> dict[str, Any]:
        return {t.name: t for t in self.tools}

    def resolve_tools(self, names: list[str] | None) -> list[Any]:
        """Select this agent's tools by name.

        An empty or missing selection means the full set — the planner opting
        not to narrow, rather than opting into nothing. A name that does not
        match any tool is logged and skipped rather than raised: a stale plan
        should degrade the research, not abort the run. The drift test in
        ``tests/test_planning.py`` is what stops that going unnoticed.
        """
        available = self.tool_map()
        if not names:
            return list(self.tools)

        selected = []
        for name in names:
            tool = available.get(name)
            if tool is None:
                logger.warning(
                    "Plan named unknown tool %r for agent %r — skipping. Known: %s",
                    name, self.name, sorted(available),
                )
                continue
            selected.append(tool)

        if not selected:
            logger.warning(
                "No named tool resolved for agent %r — falling back to full toolset",
                self.name,
            )
            return list(self.tools)
        return selected

    def compose_task(self, company: str, focus: str = "") -> str:
        """The base task for *company*, narrowed by the planner's focus."""
        task = self.task_prompt.format(company=company)
        return f"{task}\n\n{focus}" if focus else task
