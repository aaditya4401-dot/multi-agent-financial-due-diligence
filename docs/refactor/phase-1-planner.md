# Phase 1 — Orchestrator becomes a Planner

**Status:** complete — planner, Send dispatch, and state migration built; 145 tests green
**Verified against:** langgraph 1.2.11, langchain 1.3.15, pydantic 2.13.4

## Goal

Replace the pass-through `orchestrator_node` with a planner that classifies the
target company and dispatches a *variable* set of research tasks, so the graph
stops being a straight line and the four agents stop being a fixed constant.

## What forces the shape

```python
workflow.add_edge(["financial", "market", "risk", "sentiment"], "evidence")
```

`src/graph.py:64` is a **join**: `evidence` fires only when all four named nodes
have run. Skip one agent and the barrier never satisfies. Conditional routing is
therefore impossible without also reworking the fan-in — they are one change.

### Chosen: one generic `research` node, N invocations via `Send`

```python
workflow.add_conditional_edges("planner", dispatch, ["research"])
workflow.add_edge("research", "evidence")
```

Rejected alternative: keep four named nodes and return a subset of their names
from a conditional edge. That works, but caps width at four forever and relies
on subtler scheduling semantics. `Send` is the documented map-reduce idiom and
gives genuinely dynamic width.

Cost, stated honestly: `draw_mermaid()` collapses to `planner → research →
evidence`. The four-way picture is lost in the diagram but not in the Streamlit
UI, which reads the distinct state keys.

## Runtime facts established by probe

Three things were measured against langgraph 1.2.11 rather than assumed:

1. **A `Send` payload is the node's entire input.** It does *not* merge into
   graph state. A node dispatched with `Send("research", {"task": ...})` cannot
   read `state["company"]`. Payloads must be self-contained.
2. **Two `Send`s writing the same state key raise `InvalidUpdateError`**
   ("Can receive only one value per step") unless that key has a reducer.
3. **The fan-in edge fires exactly once** for N = 1, 2, 4, 6, including when
   the dispatched tasks have uneven durations.

Fact 1 changes the node signature. Fact 2 is the open decision below.

## Decided: reducer-backed findings (Option 2)

> **Resolved.** Option 2 was chosen, and it simplifies further than
> anticipated: once findings are reducer-backed, the four `*_findings`
> keys collapse into a single `findings: Annotated[list[AgentFindings],
> add]`, because each `AgentFindings` already carries `agent_name`. That
> removes `state_key` from `ResearchTask` entirely. Original reasoning
> below.


Fact 2 means the "dispatch two financial tasks with different focuses" benefit
of `Send` is **not** available while `financial_findings` is a plain
`Optional[AgentFindings]`.

**Option 1 — one task per agent (minimal).** State shape untouched. Phase 1 stays
small. But Phase 2's refinement loop writes follow-up findings into
already-populated keys, so it hits the same wall and forces the change then.

**Option 2 — make findings reducer-backed now (recommended).**
`Annotated[list[AgentFindings], add]`, keyed by agent. Pay the rework once
instead of twice: `evidence.py`'s `FINDING_KEYS` loop and `synthesizer.py`'s
`SECTIONS` loop get touched a single time.

Recommendation: **Option 2.** Phase 2 forces it regardless, and doing it under
Phase 1's smaller blast radius is cheaper than doing it mid-loop-refactor.
Larger diff now (`state.py`, `evidence.py`, `synthesizer.py`, `app.py`, tests),
but no second migration.

## Classification: the model proposes, yfinance disposes

One FAST-tier structured call returns `{company_type, ticker, aliases, sector,
rationale}`. The ticker is then **verified** against yfinance using the guard
already living at `src/tools/financial_tools.py:57` (`info.get("quoteType") is
None`), extracted into a shared `resolve_ticker()`.

| Situation | Result |
|---|---|
| Ticker verifies | `PUBLIC`, ticker recorded |
| Model says public, ticker fails to verify | `UNKNOWN` — *not* `PRIVATE` |
| Classification call fails entirely | `UNKNOWN`, full four-task plan |

The middle row matters: no ticker is not proof of privateness (foreign listing,
flaky API). Collapsing it to `PRIVATE` would assert a fact we did not establish
— the same philosophy as `inconclusive_report`, where a pipeline failure is an
absence of evidence rather than evidence of a bad investment.

> **The defensible line:** the model is used for *recall* ("what is the ticker
> for Block Inc?"), the API for *verification*. A model asserting something a
> cheap deterministic check could settle is a design smell.

## What the plan changes per company type

**Public** — financial drops `tool_get_funding_rounds` (a public company's
Series C is noise); focus moves to 10-K/10-Q line items and margin trend. Risk
keeps the SEC tool and targets the filing's risk-factors section.

**Private** — financial drops `tool_get_company_financials` entirely. Today that
is a yfinance lookup that returns `[]` for Stripe and falls through to web
anyway (`financial_tools.py:38-42`) — pure wasted latency. It gets
`[funding_rounds, web]` plus an instruction to attribute every number to whoever
estimated it. Risk keeps SEC search (private companies still appear in
enforcement actions and S-1s) but deprioritises the EDGAR half.

**Unknown** — today's behaviour, unrestricted.

## Honesty mechanism, and what was rejected

Rejected: an `evidence_ceiling` float per task capping how much confidence a
private-company financial claim may earn. There is already a confidence model
driven by source tier — a web estimate lands at `NEWS` (0.55) against a filing's
0.95. A second knob means two mechanisms that can disagree, with no clean answer
for which governs. **Do not build two confidence systems.**

Adopted instead: a `caveat` string on the task, carried into `AgentFindings` and
rendered into the synthesizer prompt for that section. *"No audited public
financials; all figures are third-party estimates."* Labelling, not scoring.

Related bug fixed in passing: `financial_tools.py:64-70` stamps
`source_quality="official_filing"` (prior 0.95) on everything yfinance returns.
Fair for a verified ticker, indefensible for a misresolved one. Gate it on the
planner's verification.

## Correction from implementation: Phase 1 plans do not skip agents

The design above implies the planner drops whole agents by company type. Building
the routing rules did not bear that out. All four agents have something to say
about both public and private companies — there is no listing status that makes
market research or sentiment analysis worthless. The honest differentiation is
**tools and focus**, not membership:

| | Public | Private |
|---|---|---|
| financial tools | `financials`, `web` | `funding_rounds`, `web` |
| dropped, and why | funding history is noise for a listed company | the ticker lookup is a guaranteed miss |

`plan.skipped()` and the `skipped_sections` machinery below are therefore *not*
exercised by Phase 1. They are kept because Phase 3 supplies the genuine skip:
a human at the approval gate deselecting an agent before it spends. That is a
real skip and it must not be recorded as a failure.

Cutting an agent purely to look dynamic would have been architecture theatre.
`tests/test_planning.py::test_every_plan_dispatches_all_four_agents` pins this
decision so it is a deliberate choice on the record rather than an oversight.

## Third state: skipped is not failed

`synthesizer.py:72-79` treats a missing findings key as **failed**:

```python
if not findings or not findings.get("ok", True):
    failed.append(label.lower())
```

Once the planner can skip an agent, a deliberately-skipped agent is reported as
a failure and lands in `unavailable_sections`, dragging down
`overall_confidence`. But "we chose not to run this because it is irrelevant for
a public company" is a third state, and it carries no penalty.

The plan is in state, so this is derivable: an agent absent from `plan.tasks`
was skipped by design; an agent present in `plan.tasks` with missing or not-ok
findings genuinely failed. Adds `skipped_sections` to the report schema
alongside `unavailable_sections`.

## Shape

```python
class CompanyType(str, Enum):
    PUBLIC, PRIVATE, UNKNOWN = "public", "private", "unknown"

class ResearchTask(BaseModel):
    agent: str          # "financial" | "market" | "risk" | "sentiment"
    state_key: str      # where this task's findings land
    focus: str          # appended to the agent's base task prompt
    tools: list[str]    # tool names, resolved through a registry
    caveat: str = ""    # propagates into the report section

class ResearchPlan(BaseModel):
    company: str
    company_type: CompanyType
    ticker: str | None
    rationale: str      # why these tasks — shown to the human in Phase 3
    tasks: list[ResearchTask]
```

`rationale` exists now because Phase 3 needs something to show a human at the
approval gate.

## Files that move

| File | Change |
|---|---|
| `src/planning/` | New: `models.py`, `classify.py` (ticker resolution), `plan.py` (routing rules) |
| `src/state.py` | `+ plan` field; findings keys reducer-backed if Option 2 |
| `src/graph.py` | orchestrator → planner, conditional dispatch, fan-in rework |
| `src/agents/base.py` | Payload-driven node signature (probe fact 1); single-slot `compiled` cache at line 58 becomes a dict keyed by toolset, since tools now vary per run |
| `src/agents/{financial,market,risk,sentiment}.py` | Become *specs* (prompt + tool registry) rather than prebuilt nodes |
| `src/models/schemas.py` | `+ skipped_sections` |
| `tests/test_pipeline.py:270-283` | `TestGraphTopology` asserts today's exact edges — fails by construction, rewritten to the new contract |

## Why this is defensible

Routing rules live in pure functions — `plan_for(company_type, ticker) ->
ResearchPlan`. No LLM, no network. "Does a private company skip yfinance?"
becomes a millisecond unit test. Only `classify_company` touches the outside
world, and it sits behind a deterministic verifier.

> The orchestrator's **judgment** is deterministic and tested. Only its
> **perception** is probabilistic.
