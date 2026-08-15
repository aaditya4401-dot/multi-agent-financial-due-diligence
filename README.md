# Multi-Agent Due Diligence System

A multi-agent AI system built with LangGraph that performs automated due diligence on companies by deploying 4 specialized agents in parallel to analyze financial health, market position, risk factors, and public sentiment — then resolves conflicts between agents and synthesizes findings into a structured investment memo with confidence scoring.

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

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...        # Required — powers all 4 agents + resolver + synthesizer
TAVILY_API_KEY=tvly-...      # Required — web search, news, Reddit, SEC filings
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
python -m pytest tests/ -v
```

`tests/test_tools.py` smoke-tests the tool layer (hits the network when keys are
set). `tests/test_pipeline.py` covers caching, failure semantics, and graph
wiring entirely offline — no keys needed.

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

---

## Architecture Overview

```
User Input: "Analyze Stripe"
        │
        ▼
┌─────────────────────┐
│   Orchestrator      │  ← LangGraph StateGraph
│   (routes, manages) │
└────┬───┬───┬───┬────┘
     │   │   │   │        ← Fan-out (parallel execution)
     ▼   ▼   ▼   ▼
   ┌───┐┌───┐┌───┐┌───┐
   │ F ││ M ││ R ││ S │  ← 4 independent agents
   │ i ││ a ││ i ││ e │     each with own LLM + tools
   │ n ││ r ││ s ││ n │
   └─┬─┘└─┬─┘└─┬─┘└─┬─┘
     │   │   │   │        ← Fan-in (merge results)
     ▼   ▼   ▼   ▼
┌─────────────────────┐
│      Evidence       │  ← Extracts typed claims, then
│ extract→detect→score│     detects contradictions by
└──────────┬──────────┘     groupby — no LLM guesswork
           ▼
┌─────────────────────┐
│   Synthesizer       │  ← Produces structured
│   (final report)    │     investment memo
└──────────┬──────────┘
           ▼
   Due Diligence Report
   (with derived confidence)
```

### How the graph is wired

```python
# Fan-out: orchestrator → all 4 agents in parallel
workflow.add_edge("orchestrator", "financial")
workflow.add_edge("orchestrator", "market")
workflow.add_edge("orchestrator", "risk")
workflow.add_edge("orchestrator", "sentiment")

# Fan-in: all 4 → conflict resolver
workflow.add_edge(["financial", "market", "risk", "sentiment"], "conflict_resolver")

# Linear: resolver → synthesizer → END
workflow.add_edge("conflict_resolver", "synthesizer")
workflow.add_edge("synthesizer", END)
```

LangGraph runs all four agent nodes concurrently via asyncio. Each node writes to its own state field (`financial_findings`, `market_findings`, etc.) — no write conflicts. Shared fields (`conflicts`, `messages`) use `Annotated[list, add]` reducers that concatenate results.

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

---

## Project Structure

```
multi-agent-due-diligence/
├── README.md
├── CLAUDE.md                     # AI coding assistant instructions
├── requirements.txt
├── .env                          # API keys (OPENAI_API_KEY, TAVILY_API_KEY)
├── app.py                        # Streamlit frontend (dark fintech theme)
├── src/
│   ├── __init__.py
│   ├── state.py                  # DueDiligenceState, Finding, AgentFindings, Conflict
│   ├── llm.py                    # Tiered, cached LLM clients (FAST/REASONING/SYNTHESIS)
│   ├── graph.py                  # LangGraph StateGraph wiring + CLI entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # make_research_agent factory (shared agent machinery)
│   │   ├── utils.py              # Shared: parse_react_output, error_findings
│   │   ├── financial.py          # Financial analysis agent (yfinance + Tavily)
│   │   ├── market.py             # Market research agent (Tavily)
│   │   ├── risk.py               # Risk assessment agent (SEC + Tavily)
│   │   ├── sentiment.py          # Sentiment analysis agent (news + Reddit + Tavily)
│   │   ├── evidence.py           # extract → detect → score → conflicts
│   │   ├── conflict_resolver.py  # Qualitative tension detection (the LLM's share)
│   │   └── synthesizer.py        # Final report generation (structured output)
│   ├── claims/
│   │   ├── __init__.py
│   │   ├── models.py             # Claim, SourceTier, domain-based tiering
│   │   ├── ontology.py           # Predicates, aliases, units, half-lives, bounds
│   │   ├── extract.py            # Findings → typed claims (FAST tier)
│   │   ├── contradictions.py     # Deterministic detection + adjudication
│   │   └── confidence.py         # Confidence propagation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py       # Tavily client pool + TTL cache + request coalescing
│   │   ├── financial_tools.py    # yfinance + Tavily fallback for financials
│   │   ├── news_tools.py         # News search + Reddit sentiment via Tavily
│   │   └── sec_tools.py          # SEC EDGAR filing search via Tavily
│   └── models/
│       ├── __init__.py
│       └── schemas.py            # Pydantic v2 report models (single source of truth)
└── tests/
    ├── __init__.py
    ├── test_tools.py             # Smoke tests for all tool modules
    ├── test_pipeline.py          # Caching, failure semantics, graph wiring (offline)
    └── test_claims.py            # Normalization, detection, confidence (offline)
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

All tools return empty results on failure instead of raising exceptions, so the pipeline always completes.

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
LangChain 1.x, Pydantic v2, Tavily, yfinance, Streamlit.

> **Note on LangGraph 1.x:** agents are built with
> `langchain.agents.create_agent(system_prompt=...)`. The older
> `langgraph.prebuilt.create_react_agent(state_modifier=...)` is removed —
> passing `state_modifier` raises `TypeError` on LangGraph 1.x.
