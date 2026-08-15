import json
import logging
from datetime import datetime

from src.state import AgentFindings, Finding

logger = logging.getLogger(__name__)


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def parse_react_output(agent_name: str, agent_output: dict) -> AgentFindings:
    """Extract structured AgentFindings from a react agent's message output.

    Walks the message list, pulling Finding dicts out of ToolMessage content
    and using the last AI message (with no tool calls) as the summary.
    """
    messages = agent_output.get("messages", [])

    findings: list[Finding] = []
    data_sources: list[str] = []

    for msg in messages:
        if msg.type == "tool":
            data_sources.append(msg.name)
            try:
                parsed = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "claim" in item:
                        findings.append(Finding(
                            claim=item["claim"],
                            source=item.get("source", msg.name),
                            confidence=float(item.get("confidence", 0.6)),
                            source_quality=item.get("source_quality", "news_article"),
                            date_of_data=item.get("date_of_data", today()),
                        ))
                    elif isinstance(item, dict) and "content" in item:
                        # SearchResult-shaped dicts from search_web
                        findings.append(Finding(
                            claim=item["content"][:500],
                            source=item.get("url") or item.get("title", msg.name),
                            confidence=0.5,
                            source_quality="news_article",
                            date_of_data=item.get("date") or today(),
                        ))

    summary = ""
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            summary = msg.content
            break

    return AgentFindings(
        agent_name=agent_name,
        findings=findings,
        summary=summary,
        data_sources_used=list(set(data_sources)),
        ok=True,
    )


def error_findings(agent_name: str, exc: Exception) -> AgentFindings:
    """Return a placeholder AgentFindings for a completely failed agent.

    Marked ``ok=False`` so the synthesizer reports the section as unknown
    rather than folding the failure into a low score.
    """
    return AgentFindings(
        agent_name=agent_name,
        findings=[],
        summary=f"{agent_name.title()} agent encountered an error: {exc}",
        data_sources_used=[],
        ok=False,
    )
