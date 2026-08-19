"""The judgement half of evaluation, kept offline and kept separate.

Groundedness answers "does the report only assert what we found?" — a lookup.
This answers "is the report any *good*?" — whether drivers are genuinely
falsifiable, whether the recommendation follows from them, whether detected
conflicts were engaged rather than ignored. That needs judgement, so it costs a
model call.

**It runs offline, over a fixture set, not in the pipeline.** Three reasons:

* it does not change the report, so paying for it on every run buys nothing;
* a judge in the hot path is a second thing that can fail and delay a result;
* the number is only meaningful as a *trend* across a fixed set of companies.
  A single score in isolation tells you nothing about whether the system got
  better or worse.

The judge is given the report only — never the claim graph. Groundedness has
already established whether the figures are real; asking one model to check
both facts and quality tends to produce a single vague impression instead of
two separable signals.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.llm import Tier, get_llm

logger = logging.getLogger(__name__)

Criterion = Literal[
    "thesis_falsifiable",
    "verdict_follows_from_evidence",
    "conflicts_addressed",
    "caveats_honoured",
    "uncertainty_acknowledged",
]

RUBRIC: dict[str, str] = {
    "thesis_falsifiable": (
        "Do the drivers state conditions that could actually be checked and "
        "found false? 5 = every driver names a specific, observable "
        "falsifier. 1 = drivers are restatements of observations."
    ),
    "verdict_follows_from_evidence": (
        "Does the recommendation follow from the drivers and red flags as "
        "stated? 5 = the rationale traces driver by driver. 1 = the verdict "
        "is asserted independently of the analysis above it."
    ),
    "conflicts_addressed": (
        "Where the report lists conflicting evidence, does the analysis engage "
        "with it? 5 = conflicts are resolved or explicitly carried as "
        "uncertainty. 1 = conflicts are listed and then ignored."
    ),
    "caveats_honoured": (
        "Where a section is flagged as estimates rather than reported figures, "
        "does the prose respect that? 5 = estimates are always described as "
        "such. 1 = estimates are written as audited fact."
    ),
    "uncertainty_acknowledged": (
        "Does the report distinguish what it does not know from what is bad "
        "news? 5 = missing evidence is reported as unknown. 1 = absent "
        "evidence is treated as a negative finding, or glossed over entirely."
    ),
}


class CriterionScore(BaseModel):
    criterion: Criterion
    score: int = Field(ge=1, le=5)
    justification: str = Field(
        description="Cite the specific passage that earned this score."
    )


class JudgeVerdict(BaseModel):
    scores: list[CriterionScore] = Field(default_factory=list)
    summary: str = ""

    @property
    def mean(self) -> float:
        if not self.scores:
            return 0.0
        return round(sum(s.score for s in self.scores) / len(self.scores), 2)

    @property
    def weakest(self) -> CriterionScore | None:
        return min(self.scores, key=lambda s: s.score, default=None)


SYSTEM_PROMPT = (
    "You are a demanding reviewer of due diligence memos. Score the memo you "
    "are given against each criterion from 1 to 5, and justify each score by "
    "quoting the passage that earned it.\n\n"
    "Score the memo as written. Do not reward length, confidence, or fluent "
    "prose. A short memo that is honest about thin evidence should score well "
    "on uncertainty; a polished one that asserts an unsupported verdict should "
    "score badly, however well written.\n\n"
    "Criteria:\n"
    + "\n".join(f"- `{name}`: {text}" for name, text in RUBRIC.items())
)


def _render(report: dict) -> str:
    """Flatten a report to the parts a reviewer would actually read."""
    thesis = report.get("thesis") or {}
    lines = [
        f"Company: {report.get('company_name')}",
        f"Verdict: {report.get('overall_verdict')} "
        f"(score {report.get('overall_score')}, risk {report.get('risk_level')}, "
        f"confidence {report.get('overall_confidence')})",
        f"Unavailable sections: {report.get('unavailable_sections') or 'none'}",
        f"Not researched: {report.get('skipped_sections') or 'none'}",
        "",
        f"Executive summary: {report.get('executive_summary', '')}",
        "",
        f"Thesis: {thesis.get('summary', '(none)')}",
        f"Recommendation: {thesis.get('recommendation', '(none)')} — "
        f"{thesis.get('recommendation_rationale', '')}",
    ]

    for driver in thesis.get("drivers") or []:
        lines.append(
            f"  DRIVER [{driver.get('direction')}] {driver.get('statement')}"
            f"\n    falsified if: {driver.get('what_would_falsify_this')}"
        )
    for flag in thesis.get("red_flags") or []:
        lines.append(f"  FLAG [{flag.get('severity')}] {flag.get('issue')}")
    for question in thesis.get("open_questions") or []:
        lines.append(f"  OPEN: {question}")

    for conflict in report.get("conflicts_detected") or []:
        lines.append(
            f"  CONFLICT [{conflict.get('type')}] {conflict.get('claim_a')} "
            f"vs {conflict.get('claim_b')} — {conflict.get('resolution')}"
        )

    for key in ("financial", "market", "risk", "sentiment"):
        section = report.get(f"{key}_section") or {}
        findings = section.get("findings") or []
        lines.append(
            f"\n{key.upper()} (available={section.get('available', True)}, "
            f"confidence={section.get('section_confidence')}):"
        )
        for finding in findings:
            lines.append(f"  - [{finding.get('severity')}] {finding.get('claim')}")

    return "\n".join(lines)


async def judge_report(report: dict) -> JudgeVerdict:
    """Score one finished report against the rubric. Never raises."""
    try:
        model = get_llm(Tier.SYNTHESIS).with_structured_output(JudgeVerdict)
        verdict: JudgeVerdict = await model.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _render(report)},
        ])
        return verdict
    except Exception as exc:
        logger.exception("Judge failed for %r", report.get("company_name"))
        return JudgeVerdict(scores=[], summary=f"judging failed: {exc}")
