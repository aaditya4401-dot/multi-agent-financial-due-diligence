# Phase 5 — Evaluation

**Status:** complete — groundedness in-pipeline, rubric judge offline, 230 tests green

## Two layers, deliberately separate

| | Groundedness | Judge |
|---|---|---|
| Question | Does the report only assert what we found? | Is the report any good? |
| Method | Deterministic lookups over the claim graph | LLM against a rubric |
| Runs | Every run, as the last graph node | Offline, over a fixture set |
| Cost | Free | One model call per report |

**Groundedness first, and not merely in file order.** A judge scoring prose
without a groundedness check is theatre: a fluent report that cites nothing real
scores well on clarity and structure. Establish that the figures are true, then
ask whether the argument is good.

```
synthesizer ──> evaluation ──> END
```

The score is attached to the report, not left in a benchmark someone might run.
A groundedness number that only exists when you remember to measure it says
nothing about the report actually in front of the reader.

## The load-bearing check: unsupported figures

A diligence memo is mostly numbers, and a number in the prose matching no claim
is the most damaging thing this system could emit — it reads as researched fact
and is not. Detecting it is a parse and a comparison, both of which already
exist for the claim graph.

Demonstrated end to end, with a synthesizer that invents a cash balance:

```
claims gathered      : 1  (payment volume $1.4T)
exec summary asserts : $1.4T  and  $88B

FAIL  figures_traceable  0.50  1 unsupported of 2 distinct figure(s): $88B
```

**Only figures carrying a magnitude marker are checked** — a currency symbol, a
scale word, or a percent sign. Bare integers are too ambiguous: years, counts,
ordinals and list positions would all trip a naive check and bury real findings
in noise. `extract_figures("In 2024 revenue rose")` returns nothing, on purpose.
A narrow check with no false positives is worth more than a broad one nobody
trusts.

Percent figures are only compared against percent-valued claims, so `34%` never
matches a claim of 34 units.

## The other three checks

| Check | Fails when |
|---|---|
| `claims_have_sources` | claims carry no source URL |
| `thesis_drivers_grounded` | drivers were dropped in citation checking |
| `sections_backed_by_claims` | an available section rests on no extracted claim |

`thesis_drivers_grounded` needs `audit_citations` to run *before*
`validate_citations`, because validation is destructive by design — once a
fabricated driver is dropped, the evidence that it was ever proposed is gone.
"How much of this thesis was invented" is exactly what the metric reports.

## The judge

Five rubric criteria, scored 1-5 with a justification that must quote the
passage earning it: `thesis_falsifiable`, `verdict_follows_from_evidence`,
`conflicts_addressed`, `caveats_honoured`, `uncertainty_acknowledged`.

The judge sees the report only, never the claim graph. Groundedness has already
settled whether the figures are real; asking one model to check both facts and
quality produces a single vague impression instead of two separable signals.

It runs offline because it does not change the report, because a judge in the
hot path is another thing that can fail and delay a result, and because the
number is only meaningful as a trend across a fixed set of companies. A single
score in isolation tells you nothing about whether the system improved.

```
python -m src.graph Stripe --save reports/stripe.json
python scripts/evaluate.py reports/*.json --judge
```

## A bug this phase exposed

`src/tools/search_tools.py` calls `load_dotenv()` at import, so importing most
of this project pulls real API keys into the environment. Several tests are
written around "reaching a model would raise" — which is what makes them free,
fast and deterministic — and that assumption silently stopped holding whenever
an earlier test in the session imported the tools package.

The symptom was order-dependent: `tests/test_eval.py::TestJudge` passed alone
and failed after `tests/test_pipeline.py`, because by then a key was loaded and
the call that "must fail" succeeded against the real API.

**The cost of that bug was not a red test — a full-suite run quietly spent
money.** `tests/conftest.py` now scrubs model credentials for the session, which
makes the offline assumption true by construction rather than by luck. The suite
went from ~35s to ~16s, which is a measure of how much live calling was
happening.

`TAVILY_API_KEY` is deliberately *not* scrubbed: `tests/test_tools.py` exercises
the real search API on purpose and guards on the key's presence, as does the
yfinance path. Those are existing, intentional integration tests. Worth knowing
they cost money on every run, but that is a separate decision.
