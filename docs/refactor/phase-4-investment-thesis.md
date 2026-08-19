# Phase 4 — The investment thesis

**Status:** complete — thesis node, citation grounding, 206 tests green

## Goal

The report already had a verdict, a score and a risk level. What it lacked was a
**thesis**: an explicit statement of the few things the case rests on, each tied
to evidence and each falsifiable.

```
gap_analyzer ──> thesis ──> synthesizer ──> END
```

The thesis is formed *before* the memo is written, so the executive summary
reflects a view rather than the two being composed independently and hoping they
agree.

## What this deliberately does not build

The original critique asked for SWOT and Porter's Five Forces. Both were
declined. They are teaching frameworks, not artefacts of real diligence — no
investment committee reads a Five Forces — and generating them would make the
output look *less* like the thing it imitates to anyone who does this for a
living.

What diligence memos actually contain, and what this builds instead: a thesis
with falsifiable drivers, bull/base/bear scenarios, red flags graded by whether
they end the conversation, open questions, and a recommendation whose rationale
traces back to the drivers.

## The grounding check — the actual engineering

A model asked for a thesis will produce a fluent one whether or not the research
supports it, and it will cheerfully cite claim ids that do not exist. So the
prompt is not the interesting part. This is:

1. **Schema invariants** (Pydantic). A driver must cite at least one claim id
   and must state what would falsify it. `evidence_claim_ids` has
   `min_length=1`, so an uncited driver fails validation outright.
2. **Run invariants** (`validate_citations`). Cited ids are checked against the
   *actual* claim graph. Invented ids are stripped; a driver left with no valid
   citations is dropped; if nothing survives, the recommendation is forced to
   `Insufficient evidence`.

That split is worth stating plainly: **Pydantic can enforce that a citation
exists; only code with access to the run can enforce that it is true.**

Observed, with a model deliberately citing one real and one fabricated id:

```
WARNING Thesis grounding: driver 'Take rate is expanding year over year' cited 1 unknown claim id(s)
WARNING Thesis grounding: dropped driver 'Take rate is expanding...': no valid citations

drivers surviving : 1
   ▲ Payment volume exceeds $1.4T annually
     cites ['55c98a365087bf01'] | falsified if: TPV growth turns negative
red flags kept    : [('Private company, unaudited figures', [])]
```

Red flags are treated differently from drivers on purpose: a flag with a bad
citation keeps the flag and loses the citation, because surfacing a possible
problem is worth something even unsourced, whereas a driver *is* the investment
case and an unsourced one is just an opinion.

**None of this is possible without the claim graph.** Groundedness is a set
membership test here — `cited_ids <= {c.id for c in claims}` — precisely because
claims are typed and carry stable ids. Against prose findings it would be
another LLM judging whether a sentence supports a sentence.

## Falsifiability

`what_would_falsify_this` is a required, non-blank field. "Revenue is growing"
is an observation; "payment volume growth stays above 25% through FY2026, which
fails if merchant concentration keeps rising" is a driver, because someone can
go and check it. Requiring the field is a cheap forcing function against the
model emitting the former and calling it a thesis.

## Open questions are generated, not invented

`open_questions` is overwritten from the gap analyzer's output rather than
authored by the model. The gap analyzer already knows exactly what it could not
establish and has ranked it by value of information — asking a model to guess
would be strictly worse information.

This is the second time Phase 2 pays off: its gaps drive both the refinement
loop and the memo's "further diligence required" section.

## Why `FinalReport` is a separate class

`DueDiligenceReport` remains the synthesizer's `with_structured_output` target;
`FinalReport` extends it with the thesis. The model is asked only for what it
should author. Putting the thesis in the same schema would invite the
synthesizer to write a second, ungrounded version of a view that was already
formed against the evidence.

## Degradation

Both failure paths return a usable artefact rather than raising:

| Situation | Result |
|---|---|
| No claims gathered | `insufficient_evidence_thesis`, no LLM call made |
| Thesis model call fails | `insufficient_evidence_thesis`, pipeline continues |
| All drivers ungrounded | Recommendation forced to `Insufficient evidence` |

`Insufficient evidence` is deliberately distinct from `Pass`. Declining to
invest is a conclusion; having no evidence is the absence of one, and conflating
them would let a failed pipeline read as a negative judgement about a company —
the same principle as `inconclusive_report`.
