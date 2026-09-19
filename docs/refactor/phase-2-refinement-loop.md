# Phase 2 — The refinement loop

**Status:** complete — cyclic graph, deterministic gap analysis, 168 tests green
**Verified against:** langgraph 1.2.11

## Goal

Stop the synthesizer from having to write up whatever the first pass happened to
find. If the evidence is thin in a specific, nameable way, buy one more round of
narrow research and try to close it.

```
research ──> evidence ──> gap_analyzer ─┬──> research     (refine)
    ▲                                    │
    └────────────────────────────────────┘
                                         └──> synthesizer (good enough)
```

## Why a separate node, and not the synthesizer

The obvious version of this feature is "let the synthesizer say the data is too
thin and ask for more". That conflates two decisions that fail differently.

*Is the evidence sufficient?* is a question about coverage, confidence, and
contradiction — answerable with arithmetic over the claim graph, which already
exists. *How do I word this memo?* is a language task. Giving the prose-writer
authority to spend another few dollars means the spend decision inherits every
failure mode of a language model, and cannot be tested without paying one to
have an opinion.

Splitting them means `tests/test_gaps.py` can assert "thin evidence buys another
round, complete evidence does not" in about a millisecond.

## What counts as a gap

All four detectors are pure functions over the scored claim graph. No LLM runs
in `src/gaps/`.

| Kind | Detected by | Why it matters |
|---|---|---|
| `UNRESOLVED_CONTRADICTION` | two claims disagree, winner's margin < 0.15 | the report will *state* one of two incompatible numbers |
| `MISSING_METRIC` | a core predicate has no quantitative claim | the verdict leans on something we never found |
| `LOW_CONFIDENCE` | section aggregate below 0.45 | the section exists but rests on weak sources |
| `STALE_EVIDENCE` | best claim's recency factor below 0.50 | the number may no longer hold |

Contradiction outranks a missing metric deliberately. A hole means the report
omits something; a contradiction means the report asserts something we have
active reason to doubt. **Being confidently wrong is worse than being silent.**

Core predicates are per company type, so a private company is never sent to
find its market cap — that would be a guaranteed wasted round.

## Value of information

Gaps are ranked, not just counted:

```
voi = kind_weight x importance x residual_uncertainty
```

So a contradicted revenue figure outranks a missing Glassdoor rating, and a
metric already known confidently scores near zero even though it is technically
improvable. Gaps below `MIN_VALUE_OF_INFORMATION` (0.5) do not buy a round at
all.

This is the difference between "is anything unresolved?" — which is always yes,
forever — and "would resolving this change the answer?", which is the question
worth spending money on.

## Why the loop terminates

Three independent guarantees, any one of which would suffice:

1. **Round ceiling.** `research_round` increments on every gap-analysis pass;
   refinement is refused past `MAX_REFINEMENT_ROUNDS` (default 1).
2. **Monotonic attempted set.** `attempted_gaps` only grows, and attempted gaps
   are filtered out of the candidate set, so the set strictly shrinks. Gaps are
   marked at *dispatch*, not on success — a gap we chased and failed to close
   must not be chased again.
3. **Finite gap space.** Bounded by the core predicate table times the number of
   sections.

`Gap.id` deliberately excludes the value-of-information score and the prose, so
the same hole hashes identically across rounds even if its priority moved.
Without that, guarantee 2 fails silently: every lap would see "new" gaps.

The graph's `recursion_limit` is a backstop, not the mechanism.

## Why it converges rather than repeating

Each dispatched task carries the gap's own question as its `focus`:

> *This is a targeted follow-up. Earlier research left a specific gap: Find
> stripe's valuation for the most recent reported period… Answer only that. Do
> not repeat the earlier broad survey.*

**A refinement loop that re-runs the original prompt is not a loop, it is a
retry** — same prompt, same tools, same answer, twice the cost. Round two is
only worth running if it asks something round one did not.

Follow-ups also inherit the plan's tool selection for that agent. Otherwise
refinement would hand a private company back the market-data tool the planner
deliberately removed in Phase 1, and the extra round would be spent re-making
the original mistake.

## Two changes the cycle forced

**`conflicts` lost its reducer.** It was `Annotated[list[Conflict], add]`, but
only `evidence_node` writes it and that node recomputes conflicts wholly from
the full claim set. On the loop's second lap a concatenating reducer would have
appended a second copy of every conflict.

**`evidence_node` became incremental.** It tracks `extracted_upto` and sends
only unseen findings blocks for extraction — the expensive step. Detection and
scoring still re-run over the *whole* claim set, because corroboration and
contradiction are relations between claims: one new claim can change the
confidence of one gathered a round earlier.

## Observed behaviour

A private-company run where round one found only revenue:

```
Round 0: 3 gap(s) worth chasing — [missing_metric] revenue_growth_yoy (VoI 0.81);
         [missing_metric] valuation (VoI 0.72); [missing_metric] funding_total (VoI 0.58)
  ↳ Find stripe's revenue growth yoy for the most recent reported period…
  ↳ Find stripe's valuation for the most recent reported period…
  ↳ Find stripe's funding total for the most recent reported period…
Evidence: 4 claim(s) (4 quantitative, 3 new), 0 conflict(s)
Round 1: proceeding to synthesis — refinement budget of 1 round(s) spent
```

7 research tasks total: 4 planned, 3 refinements.
