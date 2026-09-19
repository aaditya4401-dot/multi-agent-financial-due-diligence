# Multi-Agent Due Diligence System

A multi-agent AI system built with LangGraph that performs automated due diligence
on companies. A planner decides what research to run; specialized agents cover
financial health, market position, risk and sentiment in parallel; their prose is
converted into a **typed claim graph** so contradictions are found by arithmetic
rather than by asking a model to eyeball two paragraphs. Thin evidence buys another
round of targeted research, an investment thesis is formed with every driver cited
to a claim id, and the finished memo is scored for groundedness before it is
returned.

The design bet, stated once: **use the LLM only where the work is genuinely
semantic.** Extraction and qualitative tension get a model. Detection,
adjudication, confidence, gap-finding and groundedness are pure Python — so they
can be unit tested instead of trusted.

Every run is traceable end to end. Optional **LangSmith** instrumentation records
cost and latency grouped by routing tier, how much work the refinement loop
actually did, and pushes the groundedness and citation-audit results onto the run
as feedback scores — so report quality is tracked across runs rather than
inspected one report at a time. It is off unless configured, and cannot fail a
run.

---

## At a glance

| | |
|---|---|
| **Orchestration** | LangGraph `StateGraph` — cyclic, with runtime fan-out via `Send` |
| **Agents** | 4 research agents (financial, market, risk, sentiment), running concurrently |
| **Planning** | Dynamic: classification decides how many agents run and which tools each may reach |
| **Refinement** | Evidence → gap analysis → targeted re-research, with provable termination |
| **Human in the loop** | `interrupt()`-based approval gate; resumable across processes via SQLite checkpoints |
| **Evidence model** | Pydantic v2 typed claim graph; contradictions found by arithmetic, not by an LLM |
| **Cost control** | 3-tier model routing (`fast` / `reasoning` / `synthesis`), cached clients, coalesced search |
| **Evaluation** | Deterministic groundedness on every run + citation audit; LLM rubric judge offline |
| **Observability** | LangSmith: per-tier cost and latency, run metadata, evaluator scores as feedback |
| **Tests** | 317, offline by construction — no keys, no network, no spend |
| **Stack** | Python 3.10+, LangGraph 1.x, LangChain 1.x, Pydantic v2, LangSmith, Tavily, yfinance, Streamlit |

---

## Quick Start

### 1. Clone & install

Requires **Python 3.10+** (the codebase uses PEP 604 `X | None` syntax).

```bash
git clone <your-repo-url>
cd multi-agent-financial-due-diligence

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the template and fill it in — `.env.example` documents every variable the
system reads, with placeholders. `.env` is gitignored; `.env.example` is not, and
must never hold a real key.

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...        # Required — powers all 4 agents + resolver + synthesizer
TAVILY_API_KEY=tvly-...      # Required — web search, news, Reddit, SEC filings

LANGSMITH_TRACING=true       # Optional — tracing is off unless this AND the key are set
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=multi-agent-due-diligence
```

### 3. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Type a company name (e.g. `Stripe`, `Tesla`, `Coinbase`) and click **Run Analysis**.

### 4. Or run from CLI

```bash
python -m src.graph Stripe
```

Prints the full JSON report to stdout.

### 5. Run tests

```bash
python -m pytest tests/ -q
```

**317 tests, and all but one module run offline** — no keys, no network, no spend.
`tests/test_tools.py` is the deliberate exception: it smoke-tests the tool layer
against live APIs when keys are present, and skips when they are not.

Offline is guaranteed by construction, not by convention. `tests/conftest.py`
scrubs model credentials *and* the LangSmith variables for the whole session,
because both had the same failure mode: a suite that quietly spends money — or
quietly ships traces to a real project — when a developer happens to have the
variable exported. The suite's environment is part of its contract.

---

## Configuration

Everything below is optional and has a working default.

| Variable | Default | What it controls |
|---|---|---|
| `DD_MODEL_FAST` | `gpt-4o-mini` | Cheap tier — mechanical, high-volume work |
| `DD_MODEL_REASONING` | `gpt-4o` | Research agents and conflict detection |
| `DD_MODEL_SYNTHESIS` | `gpt-4o` | Final report generation |
| `DD_AGENT_MAX_STEPS` | `20` | Per-agent tool-loop cap (~10 tool calls) |
| `DD_GRAPH_RECURSION_LIMIT` | `50` | Whole-graph step ceiling |
| `DD_LLM_TIMEOUT` | `90` | Per-request timeout, seconds |
| `DD_LLM_MAX_RETRIES` | `2` | Retries per LLM call |
| `SEARCH_CACHE_TTL` | `3600` | Search cache lifetime, seconds |
| `DD_CHECKPOINT_PATH` | unset | SQLite file enabling resumable runs |
| `DD_MAX_REFINEMENT_ROUNDS` | `1` | Hard ceiling on the research → evidence → gap cycle |
| `DD_MAX_REFINE_TASKS` | `3` | Follow-up tasks dispatched per round |
| `DD_MIN_GAP_VOI` | `0.5` | Value-of-information floor a gap must clear to be worth another round |

**Observability** — all optional, all inert when tracing is off.

| Variable | Default | What it controls |
|---|---|---|
| `LANGSMITH_TRACING` | unset | Master switch. Tracing needs this **and** the key |
| `LANGSMITH_API_KEY` | unset | Personal access token from smith.langchain.com |
| `LANGSMITH_PROJECT` | unset | Project runs are grouped under |
| `LANGSMITH_ENDPOINT` | unset | Self-hosted or EU only; leave unset for default |
| `DD_TRACE_FINALIZE_TIMEOUT` | `10` | Wall clock on end-of-run telemetry, seconds |
| `DD_TRACE_CONNECT_TIMEOUT_MS` | `2000` | Connect timeout for the tracing client |
| `DD_TRACE_READ_TIMEOUT_MS` | `5000` | Read timeout for the tracing client |
| `DD_TRACE_RETRIES` | `1` | Retries per tracing request — deliberately near-zero |

Resumable runs need both a checkpoint path and a thread id:

```python
await run("Stripe", thread_id="stripe-2026-08", checkpoint_path="checkpoints.sqlite")
```

A crashed run resumes from the last completed node instead of re-paying for
every agent.

---

## Demo: Analyzing Stripe

When you analyze **Stripe**, the system produces:

| Metric | Value |
|--------|-------|
| **Overall Score** | 85/100 |
| **Verdict** | Favorable |
| **Risk Level** | Moderate |
| **Confidence** | 85% |

**What the agents found:**

- **Financial Agent** — Stripe's payment volume reached $1.9T (34% YoY growth), Revenue suite ARR hitting $1B, valuation at $159B after latest tender offer. yfinance returns nothing (private company), so the agent automatically fell back to Tavily web search.
- **Market Agent** — 17% global payment processing share, $1T processing volume is <2% of the total digital payments TAM, significant room for growth.
- **Risk Agent** — Compliance challenges with KYC/identity verification, restrictions on high-risk industries (gambling, crypto). No major litigation flagged.
- **Sentiment Agent** — Neutral news trajectory, positive developer sentiment on Reddit, mixed employee reviews on Glassdoor (3.4/5 compensation rating).

**3 conflicts detected and resolved:**

1. **Factual Contradiction** — Financial agent reported $1.9T payment volume vs Market agent's $1.4T. Resolved by preferring the higher-confidence source.
2. **Stale Data** — $159B valuation (recent) vs $50B Series I valuation (older). Resolved by accepting the more recent figure.
3. **Complementary Tension** — Risk agent flagged compliance challenges while Sentiment agent found neutral media coverage. Both true — compliance issues haven't yet impacted public perception.

The full analysis runs in ~60 seconds with all 4 agents executing concurrently.

**What it actually costs** — measured across **5 traced runs**, not estimated
from one. Median, with the full range across runs in brackets. These figures are
not copied by hand; they are read back off the LangSmith API, and anyone with
access to the project can regenerate them:

```bash
python scripts/trace_stats.py --company Stripe --markdown
```

| | |
|---|---|
| Wall clock | 57s (44s–140s) |
| Total tokens | 55,033 (51,486–61,070) |
| Claims extracted | 31 (30–41) |
| Conflicts detected | 3 (3–4) |
| Planning iterations | 1 (1–2) |
| `groundedness` | 0.856 (0.826–0.929) |
| `groundedness:claims_have_sources` | 0.854 (0.767–1.000) |
| `groundedness:figures_traceable` | 0.579 (0.538–0.731) |
| `groundedness:thesis_drivers_grounded` | 1.000 |
| `groundedness:sections_backed_by_claims` | 1.000 |
| `citation_grounding_rate` | 1.000 |
| `invented_citations` | 0 |

Cost by routing tier — the number the tiering exists to produce, and the reason
it is worth grouping by tier rather than reporting one total:

| Tier | Model | Calls | Tokens | Share |
|---|---|---|---|---|
| `fast` | gpt-4o-mini | 5 (5–6) | 15,412 (14,808–17,311) | 28% |
| `reasoning` | gpt-4o | 9 (9–12) | 30,360 (28,103–34,290) | 55% |
| `synthesis` | gpt-4o | 2 | 8,931 (8,093–9,469) | 16% |

`reasoning` dominates because it carries the four research agents and their tool
loops — which makes it the obvious target if cost matters, and that is a
conclusion the system can now support with a number instead of an intuition.
The per-tier totals sum exactly to LangSmith's independently computed total for
each run, which is the cheapest available check that the accounting is right.

**Read the ranges, not just the medians.** Live search returns different results
every run, so the inputs genuinely differ: `reasoning` spends 9–12 model calls
depending on how many tool calls each agent needs, and one run took 140s against
a 44s best case. `figures_traceable` is the widest quality spread (0.538–0.731)
because it depends on which figures the model happened to quote from whatever
search returned — which is exactly what that check exists to catch.

Two things the range makes visible that a single run hides: the refinement loop
*does* fire (`planning_iterations` reaching 2) on runs where a gap clears the
value-of-information bar, and the invariants hold on every run — zero invented
citations, every thesis driver grounded, every section backed by claims.

> Numbers from a single run are an anecdote. Two runs happened to agree closely
> enough here to suggest cost was near-deterministic, which five runs disproved.
> The script exists so the figures are cheap to re-derive rather than quietly
> stale.

---

## Architecture Overview

```
User Input: "Analyze Stripe"
        │
        ▼
┌─────────────────────┐
│      Planner        │  ← classifies public/private,
│  (classify + route) │     emits a ResearchPlan
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Approval Gate     │  ← optional human review, before
│  (interrupt/resume) │     a single dollar is spent
└────┬───┬───┬───┬────┘
     │   │   │   │        ← Fan-out: one Send per planned
     ▼   ▼   ▼   ▼          task, width decided at runtime
   ┌───┐┌───┐┌───┐┌───┐
   │ F ││ M ││ R ││ S │  ← N invocations of one research
   │ i ││ a ││ i ││ e │     node, each with the tools the
   │ n ││ r ││ s ││ n │     plan selected for it
   └─┬─┘└─┬─┘└─┬─┘└─┬─┘
     │   │   │   │        ← Fan-in: one edge, fires once
     ▼   ▼   ▼   ▼          however many tasks ran
┌─────────────────────┐
│      Evidence       │  ← Extracts typed claims, then
│ extract→detect→score│     detects contradictions by
└──────────┬──────────┘     groupby — no LLM guesswork
           ▼
┌─────────────────────┐
│    Gap Analyzer     │  ← Is this good enough? Ranks
│  coverage/conflict/ │     gaps by value of information
│  confidence/recency │
└──────┬───────┬──────┘
       │       └──────────────────┐
       │ good enough              │ thin — one more round of
       ▼                          │ narrow, targeted research
┌─────────────────────┐           │
│      Thesis         │           └──> back to research ──┐
│ (grounded in claims)│  ← falsifiable drivers, each      │
└──────────┬──────────┘     citing claim ids; invented    │
           ▼                citations stripped            │
┌─────────────────────┐                                    │
│   Synthesizer       │                                    │
│   (final report)    │                                    │
└──────────┬──────────┘  <─────────────────────────────────┘
           ▼
┌─────────────────────┐
│    Evaluation       │  ← scores the report against the
│   (groundedness)    │     evidence: are its figures real?
└──────────┬──────────┘
           ▼
   Due Diligence Report
   (derived confidence +
    groundedness score)
```

### How the graph is wired

```python
# Plan, then a review gate that is a no-op unless human_review is set.
workflow.set_entry_point("planner")
workflow.add_edge("planner", "approval")

# Fan-out: one Send per planned task. The number of tasks is a runtime
# decision, so this cannot be a fixed set of edges.
workflow.add_conditional_edges("approval", dispatch_research, ["research", "thesis"])

# Fan-in: a single edge, which fires once after every dispatched task
# finishes — whatever N was.
workflow.add_edge("research", "evidence")

# The cycle: thin evidence buys another round of narrow research. Termination
# is enforced inside gap_analyzer, not by this wiring.
workflow.add_edge("evidence", "gap_analyzer")
workflow.add_conditional_edges("gap_analyzer", route_after_gaps, ["research", "thesis"])

# A view is formed before the memo is written, and the finished memo is scored
# against the evidence behind it before it leaves the graph.
workflow.add_edge("thesis", "synthesizer")
workflow.add_edge("synthesizer", "evaluation")
workflow.add_edge("evaluation", END)
```

**Why `Send` rather than conditional edges to four fixed nodes.** A conditional
edge chooses a *path* among known nodes; `Send` chooses a *population*. Only the
latter lets the planner decide how many research tasks exist at runtime.

**Why the fan-in changed.** The old fan-in named all four agents in one
`add_edge([...], ...)` call, which is a join: it fires only when every named
node has run. That barrier can never tolerate a variable-width fan-out — skip
one agent and the graph stalls. A single `research → evidence` edge fires once
regardless of how many tasks were dispatched.

**Why findings are one reducer-backed list.** Two parallel writes to the same
state key raise `InvalidUpdateError`, so a fixed key per agent cannot express
two tasks aimed at the same agent. Tasks append to
`findings: Annotated[list[AgentFindings], add]` instead, and each block carries
its own `agent_name`. `conflicts` and `messages` use the same reducer pattern.

### What the planner actually decides

Classification is deliberately split in two: the model supplies **recall** (it
knows Block, Inc. trades as a ticker and Stripe does not trade at all), and a
market-data lookup supplies **verification**. A proposed ticker that fails to
resolve downgrades the company to `unknown`, never to `private` — no ticker is
not proof of privateness, and the costs are asymmetric.

The plan changes *tools and focus*, not which agents run:

| | Public | Private |
|---|---|---|
| financial tools | market data + web | funding rounds + web |
| dropped, and why | funding history is noise for a listed company | the ticker lookup is a guaranteed miss |
| caveat | — | figures are third-party estimates |

Skipping a whole agent is not justifiable on listing status alone, so Phase 1
does not do it. The skip machinery exists for the human approval gate, where a
reviewer can deselect an agent before it spends — and a skipped section is
reported separately from a failed one, because only one of them is an evidence
gap.

### Evaluation

Two layers, answering different questions. **Groundedness** — does the report
only assert what we found? — is deterministic, free, and runs as the last node
on every run, so the score is attached to the report rather than living in a
benchmark someone might remember to run. The **judge** — is the report any
good? — is an LLM against a five-criterion rubric, and runs offline over a
fixture set.

Groundedness first, and not merely in file order: a judge scoring prose without
a groundedness check is theatre, because a fluent report that cites nothing real
scores well on clarity and structure.

The load-bearing check is **unsupported figures**. A memo is mostly numbers, and
a number matching no claim reads as researched fact while being invented. Only
figures with a magnitude marker are checked — `$1.4T`, `38%`, `100 million` —
because bare integers like years and counts would produce false positives that
bury the real findings.

```
python -m src.graph Stripe --save reports/stripe.json
python scripts/evaluate.py reports/*.json --judge
```

### Observability

The pipeline used to run blind. Seven model call sites across three tiers, a
refinement loop of variable length, and a groundedness score computed on every
run — none of it visible afterwards. There was no token accounting anywhere, so
the tiering that exists specifically to control cost could not be shown to work.

`src/observability.py` is the whole integration. The rest of the pipeline gains
134 lines across six files — 59 of them comments — because the goal was to keep
tracing concentrated in one module rather than smeared through every node. It
answers three questions — *what did this cost, per tier*,
*how much work did the loop do*, *was the output grounded* — and records nothing
that does not answer one of them.

**Off by default, and inert when off.** Tracing requires `LANGSMITH_TRACING` and
`LANGSMITH_API_KEY` both set. With either missing no client is constructed, no
callbacks are attached, and the invoke config is byte-for-byte what it was
before. Requiring the key as well as the switch means a half-configured
environment degrades to *off* rather than to a run that errors on every call.

**It cannot fail a run, and it cannot stall one.** Every entry point is wrapped
in a single `fail_open` primitive — one audited decorator rather than try/except
scattered across a dozen call sites. But catching exceptions is only half of
failing open: an unreachable endpoint does not raise, it *stalls*, and with the
client's default retry policy a single feedback call blocks for minutes. So the
client is built with short timeouts and near-zero retries, and finalisation runs
under a wall clock on a daemon thread. Unreachable costs seconds, not minutes.

#### Where the instrumentation attaches, and why

Three placement decisions, each of which has a wrong version that looks fine
until it doesn't.

**Tier is stamped on the cached client, not at the seven call sites.** `get_llm`
is `lru_cache`d on `(tier, temperature)`, so tier is a property of *the client* —
a given client always has exactly one. Stamping it at construction covers every
call site in four lines, including the nested ReAct loop, whose model calls
inherit it without `research_node` knowing tracing exists.

The same reasoning inverted is why per-run callbacks must **never** go there: the
cache outlives the run, so a usage collector attached to the client would fire
for every later run in the process. Run-scoped things travel in `config` at
invoke time. Ask of every field: *is this a property of the client, or of the
run?* — the answers differ inside one file.

**The pipeline opens its own root run.** LangSmith rejects a second update to a
finished run (`409 Duplicate run update requests`), because the tracer already
sent one when the run closed. A fact learned only at the end therefore has two
possible homes: something still open when you learn it, or a separate resource
with its own lifecycle. So `trace_run` owns the root, `record()` writes the
end-of-run metadata while it is still alive, and feedback — a separate resource
by design — is pushed afterwards.

**`gap_analyzer` records why the loop stopped instead of leaving it to be
inferred.** The round counter is incremented on *both* exits, so a run that
converged on its first pass finishes sitting exactly on the ceiling and is
indistinguishable from one that ran out of budget. Inference got this wrong in
the flattering direction — reporting "budget spent" for a loop that had simply
found nothing worth chasing. The node knew; it now says so.

> The general rule: instrument at the decision point. Downstream reconstruction
> is lossy exactly where the paths converge, which is exactly where you care.

#### What lands on a run

Naming note: `routing_tier`, never bare `tier` — `SourceTier` in the claim graph
is an unrelated evidence-quality grade, and one ambiguous column in a trace UI is
permanent. Metadata keys are a schema.

- **Trace tree** — one root run per execution; a child per graph node; research
  spans named per agent (`agent:financial`, `agent:risk`, …) because four tasks
  fan out via `Send` and would otherwise all read `research`; one
  `planning_iteration_N` span per lap of the refinement cycle.
- **Every model call** carries `routing_tier` and a `tier:*` tag, so cost and
  latency group three ways in the UI.
- **Run metadata** — planning iterations, why the loop stopped, whether the
  approval gate fired and at which stage, per-tier token and call counts, claim
  and conflict counts, and whether the run completed or is paused at a gate.
- **Feedback scores** — 7 per run: aggregate `groundedness`, one key per named
  groundedness check, `citation_grounding_rate` and `invented_citations`.

The evaluators are **wired through, not reimplemented**. `EvaluationReport` and
`CitationAudit` already carry exactly the numbers the feedback API wants, so this
is transport. Per-check keys are generated by iterating the evaluator's output,
so a fifth check added to `groundedness.py` appears in LangSmith with no change
to the tracing code.

Token counts come from a callback rather than return values, because
`.with_structured_output()` returns the parsed model and discards `usage_metadata`
at six of the seven call sites — the numbers are simply not in the return value.
A `BaseCallbackHandler` sees them anyway, joining `on_chat_model_start` (which
carries the tier) to `on_llm_end` (which carries the usage) on `run_id`.

#### Human review spans more than one run

A reviewed execution is `run()` plus one `resume()` per gate — separate root runs,
potentially in different processes days apart. There is no honest way to make
those one run: the parent would have to survive a boundary it cannot, and an
abandoned review would leave it open forever. So each segment is its own root run,
grouped by thread id, and feedback lands on the segment that actually reached
`evaluation` and produced a score.

### The investment thesis

The memo's analytical core: a one-line thesis, 3-5 **falsifiable drivers**,
bull/base/bear cases, red flags graded by whether they end the conversation, and
a recommendation whose rationale traces back to the drivers.

Every driver must cite `evidence_claim_ids` and state `what_would_falsify_this`.
Both are schema-enforced — an uncited driver fails validation, and "revenue is
growing" is an observation, not a driver.

Citations are then checked against the **actual claim graph**: invented ids are
stripped, a driver left with no valid citation is dropped, and if nothing
survives the recommendation is forced to `Insufficient evidence`. Pydantic can
enforce that a citation exists; only code with access to the run can enforce
that it is true.

This is the concrete payoff for typed claims. Groundedness is a set membership
test — `cited_ids <= {c.id for c in claims}` — because claims carry stable ids.
Against prose findings it would be another LLM judging whether one sentence
supports another.

`open_questions` is generated from the gap analyzer rather than authored by the
model: the system already knows precisely what it could not establish.

Deliberately not built: SWOT and Porter's Five Forces. They are teaching
frameworks, not artefacts of real diligence.

### Human in the loop

`run(company, human_review=True)` pauses after planning and before any agent
runs. The reviewer can approve, drop agents, or cancel; a dropped agent is
reported as *not researched* rather than as a failure, and their note lands in
the report's audit trail.

The gate is its own node containing nothing but the `interrupt()` call, because
**an interrupting node re-executes from the top on resume** — anything above
that line runs twice. Putting the gate at the end of the planner would have paid
for classification twice per approval, silently.

An interrupt is a checkpoint, so human review makes checkpointing mandatory
rather than optional, and `src/checkpointing.py` declares an explicit allowlist
of the project types that travel in that state.

### The refinement loop

`gap_analyzer` sits between evidence and synthesis and answers one question: is
this worth writing up, or is there a specific thing worth paying for another
round of research to find?

Gap detection is **deterministic** — four pure functions over the claim graph
(missing core metrics, unresolved contradictions, weak sections, stale figures).
No model decides whether the evidence is thin, so that decision can be unit
tested rather than trusted.

Gaps are ranked by **value of information**, `kind_weight x importance x
residual_uncertainty`, and gaps below a threshold buy nothing. "Is anything
unresolved?" is always yes, forever. "Would resolving this change the verdict?"
is the question worth money.

**The loop provably terminates** — a round ceiling, an attempted-gap set that
only grows (so the candidate set strictly shrinks), and a finite gap space. The
recursion limit is a backstop, not the mechanism.

**And it converges rather than repeating.** Each follow-up carries the gap's own
question as its focus and is told not to redo the broad survey. A loop that
re-runs the original prompt is not a loop, it is a retry: same prompt, same
tools, same answer, twice the cost.

---

## Key Concepts

### Multi-Agent vs Single Agent

Single agent = **one brain with many tools**.
This project = **4 agents with their own brains** — each has its own LLM call, system prompt, personality, and tool set.

Key implications:
- Each agent can **fail independently** (financial API down, but others succeed)
- Agents can **disagree** (different data sources → different conclusions)
- You need an **orchestrator** to coordinate them and a **resolver** to handle disagreements

### The claim graph

Agent findings are not passed around as prose. The `evidence` node converts
them into typed **claims**, and everything downstream reasons over those.

A claim is either **quantitative** — a number pinned to
`(subject, predicate, period, unit)` — or **qualitative**, a sourced assertion
with no number. The split matters: you can subtract two numbers, you cannot
subtract two sentences, and pretending otherwise is what made prose-based
conflict detection unreliable.

The predicate vocabulary is what makes it work. "TPV", "payment volume" and
"total payments processed" all resolve to one predicate, and `$1.9T`,
`1.9 trillion` and `1,900,000,000,000` all parse to the same float — so two
agents disagreeing becomes a `groupby`, not a judgement call.

```
agent findings
   → extract   (FAST tier; LLM maps semantics, Python does all arithmetic)
   → detect    (pure Python — no LLM, cannot hallucinate)
   → score     (pure Python — confidence propagation)
   → tensions  (REASONING tier; qualitative claims only)
```

### Conflict Resolution

| Type | Example | How it's found |
|------|---------|----------------|
| **Factual contradiction** | Agent A: TPV $1.9T, Agent B: $1.4T | Deterministic — same `(subject, predicate, period)`, values beyond the predicate's tolerance |
| **Stale data** | 2023 valuation vs 2026 valuation | Deterministic — divergence across a gap longer than the predicate's half-life |
| **Complementary tension** | Financial: growing fast, Risk: burning cash | LLM — genuinely qualitative, no arithmetic can settle it |

Only the third needs a model, and it sees only qualitative claims ranked by
confidence rather than every finding from every agent.

Adjudication is deterministic too: better source tier wins, ties break on
recency, and for stale-data conflicts recency leads instead. Sources that
*agree* with the winner are not demoted alongside the one that disagreed.

### Confidence

Confidence used to be a constant at the call site — `0.85` for anything
yfinance returned, `0.5` for anything from the web — and overall confidence was
invented by the LLM. It is now derived:

```
confidence = prior x recency x corroboration x contradiction_penalty
```

- **prior** — source tier (SEC filing 0.95 → forum 0.35), inferred from domain
- **recency** — exponential decay on the predicate's half-life (market cap
  30 days, TAM 365), floored so old evidence is discounted, not deleted
- **corroboration** — noisy-OR across *independent registrable domains*,
  discounted and capped, so syndicated copies of one wire story count once and
  a pile of forum posts cannot outweigh a primary filing
- **contradiction** — conflicting claims split confidence in proportion to
  their standing, so the loser is visibly demoted

Section confidence is the aggregate of that section's claims, and overall
confidence is the mean across available sections. Both **overwrite** whatever
the model guessed.

### Guarding against bad extraction

Every predicate carries a plausible magnitude range. A model asked for "the
valuation" will occasionally return a share price or a stray number from the
same sentence — a live run produced `valuation = $49` and `payment_volume = 40`.
Such values are demoted to qualitative claims: the text survives, the false
precision does not, and it can never manufacture a contradiction or skew a
confidence score.

### Graceful Degradation — without smoothing over failures

Every agent and tool handles failures without crashing the pipeline. If
yfinance returns nothing (private company), the financial agent falls back to
web search. If Tavily is down, tools return empty lists.

The important distinction is between **"ran fine, found nothing"** and
**"failed to run"**. Failed agents are marked `ok=False`, and that propagates:

```python
def error_findings(agent_name: str, exc: Exception) -> AgentFindings:
    return AgentFindings(
        agent_name=agent_name,
        findings=[],        # never fabricate findings for a failed agent
        summary=f"{agent_name.title()} agent encountered an error: {exc}",
        data_sources_used=[],
        ok=False,
    )
```

Consequences:

- The conflict resolver only compares agents that actually produced findings,
  and skips the LLM call entirely when fewer than two did.
- The synthesizer marks failed sections `available: false`, lists them in
  `unavailable_sections`, and lowers **confidence** rather than **score**.
- If every agent fails, the report is `Inconclusive` / `Unknown` — not a
  0/100 `Unfavorable` verdict. A pipeline failure is an absence of evidence,
  not evidence of a bad investment.
- The UI renders unavailable sections in amber as "treat as unknown, not as a
  clean result" instead of silently showing "No findings available."

### Efficiency

- **One search client per event loop**, not one per call, so connections pool
  across the run.
- **Cached and coalesced search.** All four agents research the same company,
  so queries overlap heavily. Identical concurrent queries collapse into a
  single upstream request; repeats inside the TTL are free.
- **Cached LLM clients per tier**, instead of constructing a new `ChatOpenAI`
  inside every node on every invocation.
- **Model tiering** so cheap work can run on a cheap model (see Configuration).
- **Agents built once**, lazily, rather than rebuilt per invocation.
- **Bounded tool loops** — each agent runs under a step cap and degrades to
  partial findings instead of running away.

### Testing

**317 tests, offline by construction.** The design bet pays off here: because
detection, adjudication, confidence, gap-finding and groundedness are pure
Python, the parts that decide things can be tested rather than trusted. The LLM's
share is small enough to stub.

Three principles the suite is built on:

**Hermeticity is enforced, not assumed.** `conftest.py` scrubs model credentials
for the whole session because the original failure was order-dependent — a test
asserting "reaching a model raises" passed alone and failed after any test that
imported the tools package and pulled real keys into the environment. The cost of
that bug was never a red test; it was a full-suite run quietly spending money.
The LangSmith variables are scrubbed for the identical reason one step over: a
developer with tracing exported would otherwise ship traces of every stubbed run
to a real project.

**Assert on the real artifact, one layer below the UI.** The run-tree tests
record LangChain's actual callback stream — the exact thing LangSmith renders
into a trace — so "one root run, a child per node, one span per planning
iteration" is asserted against genuine framework output rather than against a
mock of our own code. Mock the network, not the framework.

**A green suite proves your code matches your beliefs, not reality.** Two real
bugs shipped past all 317 tests: LangSmith's refusal of a second run update
(409), and the convergence mislabel above. Both were contract mismatches, and in
both cases the test double happily agreed with the wrong assumption — a mock
encodes a belief about a dependency, so when the belief is wrong the mock is
wrong the same way, now with a passing test defending it.

The remedy is one cheap real-contract check rather than more mocks: stubbed LLMs
against live LangSmith costs nothing and exercises the boundary. It found both
bugs in a single run. A partial double is worse than none, incidentally — the
recording client initially lacked `create_run`, so the tracer silently fell back
to a *real* client and the offline suite started making live calls.

---

## Project Structure

```
multi-agent-due-diligence/
├── README.md
├── CLAUDE.md                     # AI coding assistant instructions
├── requirements.txt
├── .env                          # Real keys — gitignored, never committed
├── .env.example                  # Every variable the system reads, with placeholders
├── app.py                        # Streamlit frontend (dark fintech theme)
├── scripts/
│   ├── evaluate.py               # Offline judge run over saved report fixtures
│   └── trace_stats.py            # Aggregate cost/quality across traced runs (read-only)
├── src/
│   ├── __init__.py
│   ├── state.py                  # DueDiligenceState, Finding, AgentFindings, Conflict
│   ├── llm.py                    # Tiered, cached LLM clients (FAST/REASONING/SYNTHESIS)
│   ├── observability.py          # LangSmith tracing, per-tier cost, evaluator feedback
│   ├── checkpointing.py          # Serializer allowlist for interrupt/resume state
│   ├── graph.py                  # LangGraph StateGraph wiring + CLI entry point
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── classify.py           # Public/private classification (recall + ticker verification)
│   │   ├── plan.py               # Company type → ResearchPlan (tasks, tools, caveats)
│   │   └── models.py             # ResearchPlan, ResearchTask, CompanyType
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # research_node: runs whichever ResearchTask it's handed
│   │   ├── registry.py           # AGENT_SPECS — per-agent prompt/tools/tier config
│   │   ├── spec.py               # AgentSpec: tool resolution, task composition
│   │   ├── utils.py              # Shared: parse_react_output, error_findings
│   │   ├── financial.py          # Financial agent config (yfinance + Tavily)
│   │   ├── market.py             # Market agent config (Tavily)
│   │   ├── risk.py               # Risk agent config (SEC + Tavily)
│   │   ├── sentiment.py          # Sentiment agent config (news + Reddit + Tavily)
│   │   ├── evidence.py           # extract → detect → score → conflicts
│   │   ├── conflict_resolver.py  # Qualitative tension detection (the LLM's share)
│   │   ├── gap_analyzer.py       # Refinement loop: is this good enough, or one more round?
│   │   ├── thesis.py             # Investment thesis, citation-audited against claims
│   │   ├── synthesizer.py        # Final report generation (structured output)
│   │   ├── evaluation.py         # Groundedness scoring node, last before END
│   │   └── approval.py           # Human-in-the-loop plan review gate (interrupt/resume)
│   ├── claims/
│   │   ├── __init__.py
│   │   ├── models.py             # Claim, SourceTier, domain-based tiering
│   │   ├── ontology.py           # Predicates, aliases, units, half-lives, bounds
│   │   ├── extract.py            # Findings → typed claims (FAST tier)
│   │   ├── contradictions.py     # Deterministic detection + adjudication
│   │   └── confidence.py         # Confidence propagation
│   ├── gaps/
│   │   ├── __init__.py
│   │   ├── models.py             # Gap, GapKind
│   │   └── detect.py             # Four deterministic gap-finding passes, ranked by VoI
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── models.py             # GroundednessCheck, EvaluationReport, CitationAudit
│   │   ├── groundedness.py       # Deterministic: does the report assert only what we found?
│   │   └── judge.py              # LLM-judged rubric score, run offline over fixtures
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py       # Tavily client pool + TTL cache + request coalescing
│   │   ├── financial_tools.py    # yfinance + Tavily fallback for financials
│   │   ├── news_tools.py         # News search + Reddit sentiment via Tavily
│   │   └── sec_tools.py          # SEC EDGAR filing search via Tavily
│   └── models/
│       ├── __init__.py
│       ├── schemas.py            # Pydantic v2 report models (single source of truth)
│       └── thesis.py             # InvestmentThesis, Driver, RedFlag models
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_tools.py             # Smoke tests for all tool modules (hits network)
    ├── test_pipeline.py          # Caching, failure semantics, graph wiring (offline)
    ├── test_claims.py            # Normalization, detection, confidence (offline)
    ├── test_planning.py          # Classification + plan generation (offline)
    ├── test_gaps.py              # Gap detection + value of information (offline)
    ├── test_thesis.py            # Citation audit + validation (offline)
    ├── test_approval.py          # Human-in-the-loop interrupt/resume (offline)
    ├── test_checkpointing.py     # Serializer allowlist (offline)
    ├── test_eval.py              # Groundedness checks (offline)
    └── test_observability.py     # Run tree, tier metadata, feedback, fail-open (offline)
```

The four research agents are configuration only — prompt, tools, and target
state key. All shared machinery lives in `agents/base.py`.

---

## Tools & APIs

| Tool | Used By | What It Does |
|------|---------|-------------|
| **yfinance** | Financial Agent | Stock prices, revenue, margins, cash flow for public companies |
| **Tavily Search** | All Agents | Web search (general, news, Reddit, SEC). Single API key powers everything |
| **Tavily (news topic)** | Sentiment Agent | `search_depth="advanced"` + `topic="news"` for recent articles |
| **Tavily (reddit scope)** | Sentiment Agent | `include_domains=["reddit.com"]` for developer/user discussions |
| **Tavily (sec.gov scope)** | Risk Agent | `include_domains=["sec.gov"]` for SEC EDGAR filings |
| **LangSmith** | Whole pipeline | Optional tracing: per-tier cost/latency, run metadata, evaluator feedback scores |

All tools return empty results on failure instead of raising exceptions, so the pipeline always completes.
LangSmith follows the same rule one step further: it fails open *and* fails fast,
so neither an outage nor an unreachable endpoint can fail or delay a run.

---

## Output Structure

The synthesizer produces a structured investment memo via `with_structured_output`:

```python
class DueDiligenceReport(BaseModel):
    company_name: str
    report_date: str
    overall_score: int                    # 0-100
    overall_verdict: Literal["Favorable", "Cautious", "Unfavorable", "Inconclusive"]
    risk_level: Literal["Low", "Moderate", "High", "Unknown"]
    overall_confidence: float             # 0.0-1.0

    financial_section: ReportSection      # + available: bool
    market_section: ReportSection
    risk_section: ReportSection
    sentiment_section: SentimentSection   # + news_trajectory, developer/employee sentiment

    conflicts_detected: list[ConflictModel]
    executive_summary: str
    unavailable_sections: list[str]       # agents that failed — unknown, not negative
```

This model is the `with_structured_output` target *and* what the UI renders,
so the requested shape and the consumed shape cannot drift apart.

---

## Requirements

See [`requirements.txt`](requirements.txt). Core stack: LangGraph 1.x,
LangChain 1.x, Pydantic v2, LangSmith, Tavily, yfinance, Streamlit.

> **Note on `langsmith`:** it already arrives as a `langchain-core` dependency,
> but is pinned explicitly because `src/observability.py` imports it directly —
> an implicit transitive dependency is not a contract. It stays optional at
> runtime regardless.

> **Note on LangGraph 1.x:** agents are built with
> `langchain.agents.create_agent(system_prompt=...)`. The older
> `langgraph.prebuilt.create_react_agent(state_modifier=...)` is removed —
> passing `state_modifier` raises `TypeError` on LangGraph 1.x.
