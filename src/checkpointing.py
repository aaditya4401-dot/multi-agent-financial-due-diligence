"""How this project's own types survive a checkpoint.

Checkpointing is not optional any more. A human approval gate is implemented as
an interrupt, an interrupt is a durable write of the whole graph state, and that
state is full of project types — ``Claim``, ``ResearchPlan``, ``Gap`` and their
enums — rather than plain JSON.

LangGraph will deserialize unknown types today, but it warns that it will stop:
reconstructing arbitrary classes from a checkpoint store is a code-execution
path if anyone can write to that store. The permissive default is therefore
both a forward-compatibility problem and a security posture we should not want.

Declaring the allowlist explicitly fixes both. Anything not named here will not
be reconstructed, which is the correct default for data read back off disk.

**If you add a Pydantic model or enum to** :class:`~src.state.DueDiligenceState`
**, add it here too** — otherwise resuming a run silently loses it once strict
mode becomes the default.
"""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.claims.models import Claim, SourceTier
from src.claims.ontology import Predicate, Unit
from src.eval.models import CitationAudit, EvaluationReport, GroundednessCheck
from src.gaps.models import Gap, GapKind
from src.models.thesis import (
    InvestmentThesis,
    RedFlag,
    ScenarioCase,
    ThesisDriver,
)
from src.planning.models import (
    Classification,
    CompanyType,
    PlanDecision,
    ResearchPlan,
    ResearchTask,
)

#: Every project type that can appear in checkpointed graph state.
CHECKPOINTED_TYPES: tuple[type, ...] = (
    # claims
    Claim,
    SourceTier,
    Predicate,
    Unit,
    # planning
    ResearchPlan,
    ResearchTask,
    Classification,
    PlanDecision,
    CompanyType,
    # gaps
    Gap,
    GapKind,
    # thesis
    InvestmentThesis,
    ThesisDriver,
    ScenarioCase,
    RedFlag,
    # evaluation
    EvaluationReport,
    GroundednessCheck,
    CitationAudit,
)


def serializer() -> JsonPlusSerializer:
    """A serializer that will rebuild this project's types, and nothing else."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
