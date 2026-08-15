"""Typed claims: the evidence layer underneath the agents.

Pipeline: agent findings → :func:`extract_claims` → :func:`detect` →
:func:`score_claims`. Contradiction detection is deterministic; the LLM is only
asked to adjudicate what the groupby already found.
"""

from src.claims.confidence import aggregate_confidence, score_claims
from src.claims.contradictions import Contradiction, DetectionResult, detect
from src.claims.extract import extract_claims, finding_to_claim
from src.claims.models import Claim, SourceTier, tier_for_source
from src.claims.ontology import Predicate, Unit, parse_value, resolve_predicate

__all__ = [
    "Claim",
    "Contradiction",
    "DetectionResult",
    "Predicate",
    "SourceTier",
    "Unit",
    "aggregate_confidence",
    "detect",
    "extract_claims",
    "finding_to_claim",
    "parse_value",
    "resolve_predicate",
    "score_claims",
    "tier_for_source",
]
