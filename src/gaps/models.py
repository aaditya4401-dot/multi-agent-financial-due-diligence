"""What the evidence is still missing, as data.

A :class:`Gap` is a specific, answerable question — not "the financial section
feels thin" but "no revenue figure exists for FY2025". That specificity is what
makes the refinement loop converge: round two can only be worth running if it
asks something round one did not.

Every gap carries a **value of information**: how much resolving it would move
the verdict. Prioritising by that, rather than by "is anything unresolved",
keeps the loop from spending a second round chasing a detail that changes
nothing.
"""

import hashlib
from enum import Enum

from pydantic import BaseModel, Field


class GapKind(str, Enum):
    """Why the evidence is insufficient. Ordered by how much it should worry us."""

    #: Two sources disagree and neither clearly wins. The report would state a
    #: number we have active reason to doubt.
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"

    #: A metric the verdict leans on has no claim at all.
    MISSING_METRIC = "missing_metric"

    #: A section exists but rests on weak sources.
    LOW_CONFIDENCE = "low_confidence"

    #: The best claim for a metric is old enough that it may no longer hold.
    STALE_EVIDENCE = "stale_evidence"


class Gap(BaseModel):
    """One specific thing worth another round of research."""

    kind: GapKind
    agent: str                       # who should chase it
    subject: str
    predicate: str | None = None
    detail: str = ""                 # human-readable statement of the hole
    question: str = ""               # the narrow follow-up to actually ask
    value_of_information: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def id(self) -> str:
        """Stable identity, so a gap already attempted is never reissued.

        Deliberately excludes the value-of-information score and the prose:
        the *same hole* must hash the same across rounds even if its priority
        shifted, or the loop could chase it forever under a new id.
        """
        parts = [self.kind.value, self.agent, self.subject, self.predicate or ""]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def describe(self) -> str:
        return f"[{self.kind.value}] {self.detail} (VoI {self.value_of_information:.2f})"
