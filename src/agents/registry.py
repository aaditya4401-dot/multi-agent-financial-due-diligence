"""Agent name → :class:`AgentSpec`.

The one place that knows the full cast. The planner names agents and tools as
strings so its routing rules stay pure and importable without an LLM; this is
where those strings become objects, at dispatch time.
"""

from src.agents import financial, market, risk, sentiment
from src.agents.spec import AgentSpec

AGENT_SPECS: dict[str, AgentSpec] = {
    spec.name: spec
    for spec in (financial.SPEC, market.SPEC, risk.SPEC, sentiment.SPEC)
}
