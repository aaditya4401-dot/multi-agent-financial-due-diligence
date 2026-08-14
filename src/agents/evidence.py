"""The evidence node: agent findings in, scored claim graph out.

This replaces the old conflict resolver's job of eyeballing four prose dumps
for numeric disagreements. The work now splits by what each tool is good at:

1. **Extraction** (FAST tier, per agent, in parallel) turns prose into typed claims.
2. **Detection** (pure Python) finds numeric contradictions by groupby.
3. **Scoring** (pure Python) propagates confidence through the evidence.
4. **Tension detection** (REASONING tier, once) is the only LLM judgement left,
   and it only looks at qualitative claims — the part arithmetic cannot settle.

Steps 2 and 3 cost nothing and cannot hallucinate. Step 4 sees a fraction of
the context the old resolver did.
"""

import asyncio
import logging

from src.agents.conflict_resolver import detect_tensions
from src.claims.confidence import score_claims
from src.claims.contradictions import Contradiction, detect
from src.claims.extract import extract_claims
from src.claims.models import Claim
from src.state import Conflict, DueDiligenceState

logger = logging.getLogger(__name__)

FINDING_KEYS = [
    "financial_findings",
    "market_findings",
    "risk_findings",
    "sentiment_findings",
]

#: Qualitative claims handed to the tension pass, best evidence first.
MAX_TENSION_CLAIMS = 40


def _to_conflict(contradiction: Contradiction) -> Conflict:
    """Render a detected contradiction in the report's Conflict shape."""
    winner = contradiction.winner
    losers = contradiction.losers()
    loser = losers[0] if losers else None

    return Conflict(
        type=contradiction.kind,
        agent_a=winner.extracted_by if winner else "unknown",
        agent_b=loser.extracted_by if loser else "unknown",
        claim_a=winner.describe() if winner else "",
        claim_b=loser.describe() if loser else "",
        resolution=contradiction.rationale,
        resolved_confidence=round(winner.confidence, 4) if winner else 0.0,
    )


async def evidence_node(state: DueDiligenceState) -> dict:
    """LangGraph node: build the scored claim graph and derive conflicts."""
    subject = state["company"].strip().lower()

    findings = [state.get(key) for key in FINDING_KEYS]
    extracted = await asyncio.gather(
        *(extract_claims(f, subject) for f in findings),
        return_exceptions=True,
    )

    claims: list[Claim] = []
    for key, result in zip(FINDING_KEYS, extracted):
        if isinstance(result, BaseException):
            logger.error("Claim extraction failed for %s: %s", key, result)
            continue
        claims.extend(result)

    if not claims:
        logger.warning("No claims extracted for %r", subject)
        return {"claims": [], "conflicts": []}

    detection = detect(claims)
    score_claims(claims, detection)

    conflicts: list[Conflict] = [_to_conflict(c) for c in detection.contradictions]

    # The only remaining LLM judgement: tensions between qualitative claims,
    # which no amount of arithmetic can settle.
    qualitative = sorted(
        (c for c in claims if not c.is_quantitative),
        key=lambda c: c.confidence,
        reverse=True,
    )[:MAX_TENSION_CLAIMS]

    if len(qualitative) >= 2:
        conflicts.extend(await detect_tensions(subject, qualitative))

    quantitative = sum(1 for c in claims if c.is_quantitative)
    logger.info(
        "Evidence: %d claim(s) (%d quantitative), %d conflict(s)",
        len(claims), quantitative, len(conflicts),
    )
    return {"claims": claims, "conflicts": conflicts}
