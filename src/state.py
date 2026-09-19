from typing import TypedDict, Annotated, NotRequired, Optional
from operator import add

# Imported eagerly, not under TYPE_CHECKING: LangGraph resolves the state
# schema's annotations at runtime via get_type_hints(). Safe from cycles —
# src.claims.models depends only on src.claims.ontology, and
# src.planning.models depends on nothing in this package.
from src.claims.models import Claim
from src.gaps.models import Gap
from src.eval.models import CitationAudit, EvaluationReport
from src.models.thesis import InvestmentThesis
from src.planning.models import ResearchPlan, ResearchTask


class Finding(TypedDict):
    claim: str
    source: str
    confidence: float           # 0.0 to 1.0
    source_quality: str         # "official_filing" | "news_article" | "social_media" | "forum"
    date_of_data: str


class AgentFindings(TypedDict):
    agent_name: str
    findings: list[Finding]
    summary: str
    data_sources_used: list[str]
    ok: bool                    # False when the agent errored — distinct from
                                # "ran fine but found nothing"
    caveat: NotRequired[str]    # Set from the planner's task. Labels evidence
                                # quality the confidence model cannot express,
                                # e.g. "these figures are third-party estimates".


class Conflict(TypedDict):
    type: str                   # "factual_contradiction" | "complementary_tension" | "stale_data"
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    resolution: str
    resolved_confidence: float


class DueDiligenceState(TypedDict):
    company: str
    plan: Optional[ResearchPlan]                # what the planner decided, and why
    human_review: bool                          # pause for approval before spending

    # What the reviewer actually did, as data rather than prose. The decision
    # was already recorded in `messages`, but only as a sentence written for a
    # human to read — parsing it back to answer "did a human intervene, and
    # where" would mean depending on log wording as if it were an API. Each
    # entry is {stage, action, dropped_agents}; reducer-backed and keyed by
    # stage because the graph tolerates more than one gate even though only
    # the plan gate exists today. Plain dicts, so the checkpoint serializer
    # allowlist needs no new entry.
    approvals: Annotated[list[dict], add]

    # One reducer-backed list rather than four Optional keys. Research tasks
    # are dispatched via Send, so their number is decided at runtime: a fixed
    # key per agent cannot express two tasks aimed at the same agent, and two
    # parallel writes to one key raise InvalidUpdateError. Each AgentFindings
    # already carries `agent_name`, so grouping is not lost.
    findings: Annotated[list[AgentFindings], add]

    claims: list[Claim]                         # scored evidence graph, built by evidence_node

    # NOT reducer-backed, unlike findings. Only evidence_node writes conflicts,
    # and it recomputes them wholly from the full claim set on every pass. With
    # a concatenating reducer the refinement loop would append a second copy of
    # every conflict on its second lap.
    conflicts: list[Conflict]

    # How many findings blocks have already been turned into claims. The loop
    # revisits evidence_node once per round, and re-extracting blocks that were
    # already processed would pay the same LLM cost twice for the same text.
    extracted_upto: int

    # --- refinement loop bookkeeping ---------------------------------------
    # These exist to make the cycle provably finite. See gap_analyzer.
    research_round: int                         # gap-analysis passes so far
    gaps: list[Gap]                             # open gaps, recomputed each pass
    refine_tasks: list[ResearchTask]            # what to re-research; empty means stop
    attempted_gaps: Annotated[list[str], add]   # gap ids already dispatched, never reissued

    # Why the loop stopped: "converged" | "round_ceiling". Recorded rather than
    # inferred, because research_round is incremented on both exits and so a
    # converged run is indistinguishable from an exhausted one afterwards.
    refine_stop_reason: NotRequired[str]

    thesis: Optional[InvestmentThesis]           # formed before the memo is written
    thesis_audit: Optional[CitationAudit]        # what citation checking discarded
    evaluation: Optional[EvaluationReport]       # groundedness of the finished report
    final_report: Optional[dict]
    messages: Annotated[list, add]              # reducer: concatenate across parallel writes


def findings_for(state: DueDiligenceState, agent: str) -> list[AgentFindings]:
    """Every findings block produced by *agent*, in dispatch order."""
    return [f for f in (state.get("findings") or []) if f.get("agent_name") == agent]
