# Phase 3 — Human in the loop

**Status:** complete — approval gate, resumable runs, 189 tests green
**Verified against:** langgraph 1.2.11

## Goal

Let a person see the research plan before the machine spends money on it. Due
diligence is the right shape of task for this: the work is expensive, the target
may simply be the wrong company, and a reviewer often knows at a glance that a
whole line of enquiry is pointless.

```
planner ──> approval ──┬──> research ×N ──> evidence ──> gap_analyzer ──> …
                        └──> synthesizer        (cancelled: no tasks)
```

## The measurement that shaped the design

An interrupting node does **not** resume after the `interrupt()` call. It
re-executes from the top, and `interrupt()` returns the reviewer's answer the
second time through. Probed directly:

```
after first pass : ['EXPENSIVE CALL']
after resume     : ['EXPENSIVE CALL', 'EXPENSIVE CALL']
=> code before interrupt() runs 2 time(s)
```

So the gate is its own node containing nothing but the interrupt. Putting it at
the end of `planner_node` would have paid for classification twice per approval,
silently. `test_planner_is_not_paid_for_twice` pins this: it counts
classification calls across a full approve cycle and asserts exactly one.

This is also the property LangGraph actually requires — an interrupting node
must be safe to re-run.

## What the reviewer can do

| Action | Effect |
|---|---|
| `approve` | run the plan as proposed |
| `revise` + `drop_agents` | remove agents before dispatch |
| `cancel` | run nothing, report inconclusive |

`cancel` yields a plan with **no tasks** rather than a null plan, so it flows
through the same empty-plan path the graph already had from Phase 1 — no
special case needed downstream.

Dropping an agent is the genuine skip that Phase 1 built `skipped_sections` for
and deliberately did not exercise. A deselected agent is reported as *not
researched*, not as a failure, and does not reduce `overall_confidence`. The
reviewer's note is appended to `plan.rationale`, so the report carries an audit
trail of who decided what.

Decision parsing is deliberately forgiving — a `PlanDecision`, a dict, or a bare
string all work, and anything unintelligible is treated as approval. The value
crosses a process boundary, and a malformed answer is not a reason to discard a
run the reviewer already looked at.

## Checkpointing stopped being optional

An interrupt *is* a checkpoint: the graph durably writes its whole state and
stops, and resuming reads that state back — possibly in another process, or the
next day. With no checkpointer there is nothing to resume from, so
`run(human_review=True)` always compiles one, falling back to in-memory rather
than to none.

Streamlit forced the file-backed path. It reruns its entire script on every
interaction, so a paused run cannot live in process memory: the app keeps the
thread id in `st.session_state` and the state in a SQLite file.

## The serialization trap

Checkpointed state is full of project types — `Claim`, `ResearchPlan`, `Gap` and
their enums — not plain JSON. LangGraph rebuilds unregistered types today but
warns it will stop, because reconstructing arbitrary classes from a checkpoint
store is a code-execution path if anyone can write to that store.

`src/checkpointing.py` declares the allowlist explicitly, which fixes the
forward-compatibility problem and tightens the security posture at the same
time. Verified by resuming a run with `LANGGRAPH_STRICT_MSGPACK=true`:

```
resumed OK under strict msgpack
  claims rebuilt as: Claim
  gaps rebuilt as  : Gap
  plan rebuilt as  : ResearchPlan
```

The failure mode here is silent and only appears on *resume*, so
`tests/test_checkpointing.py` walks the state schema structurally and fails if
any reachable model or enum is missing from the allowlist. That test was
verified non-vacuous by removing `Gap` and watching it fail.

## API

```python
state = await run("Stripe", human_review=True)
payload = pending_review(state)          # None when the run finished
if payload:
    state = await resume({"action": "revise", "drop_agents": ["sentiment"]},
                         state["__thread_id__"])
```

CLI: `python -m src.graph Stripe --review` prompts on stdout.
