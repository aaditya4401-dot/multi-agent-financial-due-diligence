"""LangSmith tracing for a due diligence run.

The pipeline used to run blind. It makes model calls at seven sites across three
routing tiers, loops a refinement cycle a variable number of times, and scores
every report for groundedness — and none of that was visible once the run
finished. In particular there was no token accounting anywhere, so the tiered
routing that exists precisely to control cost could not be shown to work.

This module is the whole integration. Three things it is careful about:

**It is off unless asked for.** ``tracing_enabled()`` requires both
``LANGSMITH_TRACING`` and ``LANGSMITH_API_KEY``. With either absent no client is
built, no callbacks are attached, and the pipeline's config is exactly what it
was before this file existed. Requiring the key too — rather than the switch
alone — means "switch on, key forgotten" degrades to *off* rather than to a run
that emits errors on every model call.

**It fails open, structurally, and it also fails *fast*.** Every entry point
below is wrapped in :func:`fail_open`, which swallows everything and returns a
neutral value — one audited primitive rather than try/except scattered at call
sites, because it is far easier to verify that a decorator is on every public
function than that a dozen handlers are each correct.

But swallowing exceptions only covers a LangSmith that *errors*. One that
simply does not answer never raises: with the client's default retry policy a
single ``create_feedback`` against an unreachable endpoint blocks for minutes
(measured, not assumed), which would mean a finished report sitting behind
telemetry that is still politely retrying. So the client is built with short
timeouts and near-zero retries, and all of finalisation runs under a wall clock
in :func:`_bounded`. Unreachable therefore costs seconds, not minutes, and the
pipeline's own result is never held hostage.

Observability is not worth losing — or delaying — a report over, which is the
rule ``evaluation_node`` and ``judge_report`` already follow, and the one
``CLAUDE.md`` states outright.

**It reads results, it does not compute them.** The groundedness evaluator and
the citation audit already run in the graph and leave their results in state.
This module transports those numbers to LangSmith's feedback API and nothing
more — no evaluation logic lives here, and none should.

Two facts about *where* things attach, both of which cost real debugging to
learn and are easy to undo by accident:

* Run-scoped things (callbacks, run ids, per-run metadata) attach to ``config``
  at invoke time, never to the client in ``src/llm.py`` — that client is
  ``lru_cache``d and outlives the run, so a callback left on it would fire for
  every subsequent run in the process.
* Tier is the exception, and goes on the client, because tier *is* a property
  of the client. See :func:`src.llm.get_llm`.
"""

import functools
import logging
import os
import threading
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

#: How long finalisation may spend talking to LangSmith before the run gives up
#: and returns. See :func:`_bounded`.
FINALIZE_TIMEOUT_SECONDS = float(os.getenv("DD_TRACE_FINALIZE_TIMEOUT", "10"))

#: Per-request connect and read timeouts, in milliseconds.
_REQUEST_TIMEOUT_MS = (
    int(os.getenv("DD_TRACE_CONNECT_TIMEOUT_MS", "2000")),
    int(os.getenv("DD_TRACE_READ_TIMEOUT_MS", "5000")),
)

#: Retries per request. Deliberately far below the client's own default: this
#: is best-effort telemetry at the end of a finished run, not something worth
#: an exponential backoff.
_REQUEST_RETRIES = int(os.getenv("DD_TRACE_RETRIES", "1"))

#: Tier recorded for a model call whose client carried no tier metadata. Should
#: only ever appear for a model built outside get_llm(); if this shows up in the
#: UI it means a call site bypassed the factory.
UNKNOWN_TIER = "unknown"


# ---------------------------------------------------------------------------
# Fail-open primitive
# ---------------------------------------------------------------------------

def _bounded(work: Callable[[], Any], seconds: float,
             label: str = "work") -> Any:
    """Run *work* on a daemon thread, abandoning it after *seconds*.

    The second half of failing open, and the half that is easy to miss:
    :func:`fail_open` handles a LangSmith that *errors*, but not one that
    simply never answers. A dead endpoint does not raise, it stalls — and
    without a wall clock the pipeline would hand back its finished report only
    after the telemetry gave up.

    The thread is a daemon and is never joined past the budget, so an abandoned
    request cannot keep the process alive at exit either. Its result is
    discarded; it was best-effort telemetry about a run that has already
    finished.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = work()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.debug("tracing: %s failed, ignoring", label, exc_info=True)

    thread = threading.Thread(target=target, daemon=True,
                              name=f"dd-trace-{label}")
    thread.start()
    thread.join(seconds)

    if thread.is_alive():
        logger.debug("tracing: %s exceeded its %.1fs budget — abandoned",
                     label, seconds)
    return box.get("value")


def fail_open(default: Any = None) -> Callable:
    """Never let an observability failure reach the pipeline.

    Catches ``BaseException`` deliberately, not ``Exception``. A tracing call
    that raised something outside the ``Exception`` hierarchy would otherwise
    take down a run that was, by then, probably complete and correct. The two
    exits that must stay uncaught are re-raised explicitly.

    Logged at ``debug``: a broken tracer is not the operator's problem in the
    middle of a run, and at ``warning`` a LangSmith outage would bury the run's
    real output under one line per model call.

    Args:
        default: Returned on failure. Pass a zero-argument callable when the
            neutral value is mutable, so callers cannot share one instance.
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                logger.debug("tracing: %s failed, ignoring", fn.__name__,
                             exc_info=True)
                return default() if callable(default) else default
        return wrapper
    return decorate


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------

def tracing_enabled() -> bool:
    """Whether to trace this run.

    Read from the environment on every call rather than frozen into a module
    constant at import. The rest of this project freezes env at import
    (``src/llm.py:26``), which is fine for a timeout but would make this
    untestable: the test suite would have to control import order to exercise
    both the on and off paths.
    """
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() not in _TRUTHY:
        return False
    return bool(os.getenv("LANGSMITH_API_KEY", "").strip())


_client_cache: Any = None
_client_attempted = False


@fail_open()
def get_client() -> Any:
    """The LangSmith client, or ``None`` if one cannot be built.

    The failure is cached alongside the success. A missing or malformed key
    fails identically every time, and retrying per feedback call would turn one
    bad config into a burst of pointless network timeouts at the end of a run.
    """
    global _client_cache, _client_attempted

    if _client_attempted:
        return _client_cache

    _client_attempted = True
    if not tracing_enabled():
        return None

    from langsmith import Client

    # Short timeouts and almost no retries, which is not the client's default.
    # Out of the box it retries with backoff, and against an unreachable
    # endpoint a single create_feedback call blocks for minutes — measured, not
    # assumed. That is telemetry holding a finished report hostage.
    try:
        from urllib3.util import Retry

        retry_config = Retry(total=_REQUEST_RETRIES, backoff_factor=0.1,
                             allowed_methods=None, status_forcelist=[])
    except BaseException:
        retry_config = None

    _client_cache = Client(timeout_ms=_REQUEST_TIMEOUT_MS,
                           retry_config=retry_config)
    return _client_cache


def reset_client() -> None:
    """Drop the cached client. For tests, which change the env between cases."""
    global _client_cache, _client_attempted
    _client_cache = None
    _client_attempted = False


# ---------------------------------------------------------------------------
# Per-tier token accounting
# ---------------------------------------------------------------------------

class UsageByTierCallback(BaseCallbackHandler):
    """Accumulate token usage, split by routing tier.

    This handler exists because the obvious approach does not work.
    ``.with_structured_output()`` returns the parsed Pydantic object and throws
    the ``AIMessage`` away, so at six of the seven call sites the usage numbers
    are simply not in the return value and no amount of reading call sites will
    recover them. A callback sees the raw response regardless of what the chain
    did with it afterwards.

    The tier is joined on in two steps because no single callback carries both
    halves: ``on_chat_model_start`` receives the client's ``metadata`` (which
    holds ``routing_tier``) but no usage, and ``on_llm_end`` receives usage but
    no metadata. ``run_id`` is the key common to both.

    Instantiate one per run and pass it via ``config["callbacks"]``. Never
    attach it to the cached client — see this module's docstring.
    """

    #: Tell LangChain not to re-raise out of this handler. Every method is
    #: already guarded; this is the framework-level backstop behind that.
    raise_error = False

    def __init__(self) -> None:
        self._by_tier: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "total": 0}
        )
        self._tier_by_run: dict[UUID, str] = {}
        self._calls: dict[str, int] = defaultdict(int)

    # -- collection --------------------------------------------------------

    def _record_tier(self, run_id: UUID, metadata: dict | None,
                     tags: list[str] | None) -> None:
        tier = (metadata or {}).get("routing_tier")
        if not tier:
            # Fall back to the tag, in case a call site passed tags but no
            # metadata through config.
            for tag in tags or []:
                if tag.startswith("tier:"):
                    tier = tag.split(":", 1)[1]
                    break
        self._tier_by_run[run_id] = tier or UNKNOWN_TIER

    def on_chat_model_start(self, serialized: dict, messages: list, *,
                            run_id: UUID, parent_run_id: UUID | None = None,
                            tags: list[str] | None = None,
                            metadata: dict | None = None, **kwargs: Any) -> None:
        try:
            self._record_tier(run_id, metadata, tags)
        except BaseException:
            logger.debug("tracing: could not record tier", exc_info=True)

    def on_llm_start(self, serialized: dict, prompts: list, *,
                     run_id: UUID, parent_run_id: UUID | None = None,
                     tags: list[str] | None = None,
                     metadata: dict | None = None, **kwargs: Any) -> None:
        # Completion-style models. Nothing in this project uses one today, but
        # an unhandled start would silently attribute its tokens to "unknown".
        try:
            self._record_tier(run_id, metadata, tags)
        except BaseException:
            logger.debug("tracing: could not record tier", exc_info=True)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            tier = self._tier_by_run.pop(run_id, UNKNOWN_TIER)
            counts = self._usage_from(response)
            if not counts:
                return
            bucket = self._by_tier[tier]
            for key in ("input", "output", "total"):
                bucket[key] += counts.get(key, 0)
            self._calls[tier] += 1
        except BaseException:
            logger.debug("tracing: could not record usage", exc_info=True)

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     **kwargs: Any) -> None:
        # A failed call reports no usage; drop the mapping so a long run does
        # not accumulate one dead entry per retry.
        self._tier_by_run.pop(run_id, None)

    @staticmethod
    def _usage_from(response: Any) -> dict[str, int]:
        """Pull token counts out of an ``LLMResult``, whichever shape it has."""
        # Preferred: the provider-agnostic field on the message itself.
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None),
                                "usage_metadata", None)
                if usage:
                    return {
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    }

        # Fallback: the OpenAI-shaped aggregate.
        llm_output = getattr(response, "llm_output", None) or {}
        raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if raw:
            prompt = raw.get("prompt_tokens", 0)
            completion = raw.get("completion_tokens", 0)
            return {
                "input": prompt,
                "output": completion,
                "total": raw.get("total_tokens", prompt + completion),
            }
        return {}

    # -- reporting ---------------------------------------------------------

    def totals(self) -> dict[str, dict[str, int]]:
        """Per-tier counts, as a plain dict safe to serialise as metadata."""
        return {tier: dict(counts) for tier, counts in self._by_tier.items()}

    def summary(self) -> dict[str, Any]:
        """Per-tier counts plus call counts and a grand total."""
        totals = self.totals()
        return {
            "by_tier": totals,
            "calls_by_tier": dict(self._calls),
            "total_tokens": sum(c.get("total", 0) for c in totals.values()),
        }


# ---------------------------------------------------------------------------
# Trace handle
# ---------------------------------------------------------------------------

@dataclass
class TraceHandle:
    """What a traced run needs to carry from start to finish.

    A disabled handle (the default) is a working no-op: ``apply_to`` leaves the
    config untouched and ``record``/``finalize`` do nothing. That is what keeps
    the tracing-off path in ``src/graph.py`` free of conditionals.
    """

    enabled: bool = False
    run_id: UUID | None = None
    usage: UsageByTierCallback | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_tree: Any = None

    def record(self, state: dict) -> dict:
        """Write the end-of-run facts onto the root run, before it closes.

        Timing is the whole point, and it was learned the hard way. The
        obvious design — let the run finish, then PATCH the metadata on — does
        not work: LangSmith answers a second update to the same run with
        ``409 Duplicate run update requests for the same run are not
        supported``, because the tracer already sent one when the run ended.
        Waiting for the tracer first, as an earlier version did, only
        guaranteed we would always be the duplicate.

        So the root run is opened by us (see :func:`trace_run`) and this
        mutates it while it is still open. The single update the tracer sends
        on close then carries these fields, and there is no second write to
        conflict with.
        """
        # Guarded as a whole, not just around the run-tree write. This runs in
        # a `finally` on the pipeline's own path, so anything escaping it —
        # including from run_metadata — would surface as a pipeline failure on
        # a run that had already succeeded.
        try:
            metadata = run_metadata(state, self.usage)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.debug("tracing: could not derive run metadata", exc_info=True)
            return {}

        if not self.enabled or self.run_tree is None:
            return metadata

        try:
            self.run_tree.extra = self.run_tree.extra or {}
            self.run_tree.extra.setdefault("metadata", {}).update(metadata)

            # A resumed segment is opened before its company is known, so the
            # name is corrected here rather than by a second update.
            company = state.get("company")
            if company and self.metadata.get("dd.stage") == "resume":
                self.run_tree.name = f"due-diligence (resumed): {company}"
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.debug("tracing: could not record run metadata", exc_info=True)

        return metadata

    def apply_to(self, config: dict) -> dict:
        """Merge tracing keys into an existing invoke config, in place.

        Hand-guarded rather than wrapped in :func:`fail_open`, because the
        neutral value here is the caller's own config — returning a fresh empty
        dict on failure would strip the recursion limit and thread id the
        pipeline actually needs. A half-applied config is harmless; a lost one
        is not.
        """
        if not self.enabled:
            return config

        try:
            # Deliberately no `run_id`: the root run belongs to `trace_run`,
            # and the graph's own run nests underneath it.
            config["metadata"] = {**config.get("metadata", {}), **self.metadata}
            config["tags"] = [*config.get("tags", []), "due-diligence"]
            config["callbacks"] = [*(config.get("callbacks") or []), self.usage]
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.debug("tracing: could not apply trace config", exc_info=True)
        return config


@fail_open(default=TraceHandle)
def start_trace(company: str, thread_id: str | None = None,
                stage: str = "run") -> TraceHandle:
    """Open a trace for one due diligence execution.

    Args:
        company: The subject, used to name the run.
        thread_id: The checkpoint thread. Doubles as the LangSmith thread key,
            which is what stitches a human-reviewed run back together — see
            below.
        stage: ``"run"`` for a fresh execution, ``"resume"`` for a continuation
            after human review.

    Returns a disabled handle when tracing is off, so callers never branch.

    **On human review and run identity.** A reviewed execution is not one
    ``ainvoke`` — it is ``run()`` plus one ``resume()`` per gate, each a
    separate root run, and potentially in different processes days apart. There
    is no honest way to make those one run: the parent would have to stay open
    across a boundary it cannot survive, and an abandoned review would leave it
    open forever. So each segment is its own root run and they are grouped by
    thread, which is LangSmith's own answer to exactly this shape. Feedback
    lands on the final segment — the one whose graph actually reached
    ``evaluation`` and produced a score.
    """
    if not tracing_enabled():
        return TraceHandle()

    metadata: dict[str, Any] = {
        "dd.company": company,
        "dd.stage": stage,
        "dd.run_name": f"due-diligence: {company}",
    }
    if thread_id:
        # All three spellings: LangSmith's thread grouping has accepted
        # different keys across versions, and they are cheap to write.
        metadata["session_id"] = thread_id
        metadata["thread_id"] = thread_id
        metadata["dd.thread_id"] = thread_id

    return TraceHandle(
        enabled=True,
        run_id=uuid4(),
        usage=UsageByTierCallback(),
        metadata=metadata,
    )


@contextmanager
def trace_run(company: str, thread_id: str | None = None,
              stage: str = "run"):
    """Open the root run for one due diligence execution.

    We open this run ourselves rather than letting LangGraph's own run be the
    root, for one concrete reason: a run we own is still open when the graph
    returns, so the end-of-run facts can be written into it before it closes.
    LangSmith refuses a second update to a finished run (409), so anything
    learned only at the end has to go in while the run is alive. See
    :meth:`TraceHandle.record`.

    The graph's run nests underneath, so the tree gains one honest level: the
    execution, then the graph, then the nodes.

    Yields a :class:`TraceHandle`, and yields a working one even if the run
    cannot be opened — a broken tracer must not stop the caller's ``with``
    block from running the pipeline.
    """
    handle = start_trace(company, thread_id, stage)
    if not handle.enabled:
        yield handle
        return

    with ExitStack() as stack:
        try:
            from langsmith.run_helpers import trace, tracing_context

            # `tracing_context` rather than relying on the env var, so the
            # decision stays with tracing_enabled() and stays testable.
            #
            # The client is passed explicitly to both, and that matters twice
            # over: in production it is the one built with short timeouts and
            # almost no retries, and in tests it is a stub — without it, the
            # nested LangChain tracer falls back to its own default client and
            # starts posting to the real LangSmith from the unit suite.
            client = get_client()
            stack.enter_context(tracing_context(enabled=True, client=client))
            run_tree = stack.enter_context(trace(
                name=handle.metadata.get("dd.run_name", "due-diligence"),
                run_type="chain",
                inputs={"company": company},
                metadata=dict(handle.metadata),
                client=client,
            ))
            handle.run_tree = run_tree
            handle.run_id = getattr(run_tree, "id", handle.run_id)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.debug("tracing: could not open the root run", exc_info=True)

        yield handle


# ---------------------------------------------------------------------------
# Annotating spans from inside the graph
# ---------------------------------------------------------------------------

@fail_open()
def annotate_current_run(name: str | None = None, **metadata: Any) -> None:
    """Rename and/or add metadata to the span a node is currently running in.

    LangGraph already emits a child run per node, and re-entering a node on the
    next lap of the refinement cycle already produces a *separate* child run —
    so the tree shape this project needs exists without any help. What it lacks
    is labels: four parallel ``research`` runs are indistinguishable, and one
    ``gap_analyzer`` run looks like the next.

    So this stamps the existing span rather than nesting a new one. Wrapping
    node bodies in ``@traceable`` would have produced the same labels plus a
    redundant layer of nesting, and would have meant restructuring working
    nodes to serve tracing — the tail wagging the dog.

    A no-op when tracing is off, or when called outside a run context.
    """
    if not tracing_enabled():
        return

    from langsmith.run_helpers import get_current_run_tree

    run = get_current_run_tree()
    if run is None:
        return

    if name:
        run.name = name
    if metadata:
        run.extra = run.extra or {}
        run.extra.setdefault("metadata", {}).update(metadata)


# ---------------------------------------------------------------------------
# Run metadata, derived from final state
# ---------------------------------------------------------------------------

@fail_open(default=dict)
def run_metadata(state: dict, usage: UsageByTierCallback | None = None) -> dict:
    """Summarise a finished run: loop work, approvals, and cost by tier.

    All of this is only knowable after the graph returns, which is why it is
    PATCHed onto the run rather than passed in at invoke time.
    """
    metadata: dict[str, Any] = {}

    # -- refinement loop ---------------------------------------------------
    rounds = int(state.get("research_round") or 0)
    metadata["dd.planning_iterations"] = rounds
    metadata["dd.gaps_attempted"] = len(state.get("attempted_gaps") or [])

    # A segment that ends at an interrupt has not finished, it is waiting for a
    # human. Worth stating outright: without it a run paused at the approval
    # gate reports zero rounds and would be labelled "converged" below, which
    # is exactly backwards — it has not started.
    paused = bool(state.get("__interrupt__"))
    metadata["dd.status"] = "paused" if paused else "complete"

    # Why the loop stopped. "Converged" and "ran out of budget" are very
    # different facts about a run, and the round count alone conflates them:
    # a run that stopped at round 1 because nothing was worth researching
    # looks identical to one that stopped at round 1 because that is the cap.
    from src.agents.gap_analyzer import MAX_REFINEMENT_ROUNDS

    if paused:
        reason = "paused"
    elif state.get("refine_stop_reason"):
        # Reported by gap_analyzer, which is the only thing that knows. Do not
        # try to infer it from the round count: that counter is incremented on
        # both exits, so a converged run finishes sitting exactly on the
        # ceiling and infers as "round_ceiling" — wrong, and wrong in the
        # flattering direction, since it hides that the loop never fires.
        reason = state["refine_stop_reason"]
    elif state.get("refine_tasks"):
        reason = "in_progress"
    else:
        reason = "unknown"
    metadata["dd.convergence"] = reason
    metadata["dd.round_ceiling"] = MAX_REFINEMENT_ROUNDS

    # -- human approval gate ----------------------------------------------
    approvals = state.get("approvals") or []
    metadata["dd.approval_triggered"] = bool(approvals)
    metadata["dd.approval_enabled"] = bool(state.get("human_review"))
    if approvals:
        metadata["dd.approval_stages"] = [a.get("stage") for a in approvals]
        # Singular too: the graph has exactly one gate today, and a scalar is
        # what the LangSmith UI can group and filter on.
        metadata["dd.approval_stage"] = approvals[0].get("stage")
        metadata["dd.approval_actions"] = [a.get("action") for a in approvals]
        dropped = [a for entry in approvals
                   for a in (entry.get("dropped_agents") or [])]
        metadata["dd.agents_dropped_by_reviewer"] = dropped

    # -- cost --------------------------------------------------------------
    if usage is not None:
        summary = usage.summary()
        metadata["dd.tokens_by_tier"] = summary["by_tier"]
        metadata["dd.calls_by_tier"] = summary["calls_by_tier"]
        metadata["dd.total_tokens"] = summary["total_tokens"]

    # -- shape of the work -------------------------------------------------
    metadata["dd.findings_blocks"] = len(state.get("findings") or [])
    metadata["dd.claims"] = len(state.get("claims") or [])
    metadata["dd.conflicts"] = len(state.get("conflicts") or [])

    return metadata


# ---------------------------------------------------------------------------
# Feedback, from the evaluators that already ran
# ---------------------------------------------------------------------------

def _feedback_items(state: dict) -> list[dict]:
    """Translate the evaluators' existing output into feedback payloads.

    Note what this does *not* do: it does not evaluate anything. The
    groundedness evaluator ran in ``evaluation_node`` and the citation audit
    ran in ``thesis_node``; both left typed results in state, already carrying
    exactly the numbers the feedback API wants. This is transport.

    Kept separate from the pushing so the mapping can be tested without a
    client.
    """
    items: list[dict] = []

    evaluation = state.get("evaluation")
    if evaluation is not None:
        unsupported = list(getattr(evaluation, "unsupported_figures", []) or [])
        comment = getattr(evaluation, "describe", lambda: "")()
        if unsupported:
            # The diagnostic belongs with the score it explains — a 0.5 with no
            # indication of which figures went unsupported sends you back to
            # the report to find out.
            comment += f" | unsupported figures: {', '.join(unsupported[:10])}"
        items.append({
            "key": "groundedness",
            "score": float(getattr(evaluation, "score", 0.0)),
            "comment": comment,
        })

        # One key per named check. Generated by iteration rather than listed,
        # so a fifth check added to src/eval/groundedness.py appears here with
        # no change to this file.
        for check in getattr(evaluation, "checks", []) or []:
            items.append({
                "key": f"groundedness:{check.name}",
                "score": float(check.score),
                "comment": check.detail or "",
            })

    # Prefer the audit the evaluator embedded, falling back to state: a run
    # that failed during synthesis has the audit but no evaluation report.
    audit = getattr(evaluation, "citation_audit", None) if evaluation else None
    if audit is None:
        audit = state.get("thesis_audit")

    if audit is not None:
        items.append({
            "key": "citation_grounding_rate",
            "score": float(audit.grounding_rate),
            "comment": (f"{audit.drivers_kept}/{audit.drivers_proposed} "
                        f"thesis drivers were evidence-backed"),
        })
        items.append({
            "key": "invented_citations",
            "score": float(audit.invented_citations),
            "comment": "claim ids cited by the thesis that no claim matched",
        })

    return items


@fail_open(default=0)
def push_feedback(run_id: UUID | None, state: dict) -> int:
    """Attach the evaluators' scores to *run_id*. Returns how many landed.

    Each score is pushed independently: one malformed value should cost its own
    key, not the whole set.
    """
    if run_id is None:
        return 0

    client = get_client()
    if client is None:
        return 0

    pushed = 0
    for item in _feedback_items(state):
        try:
            client.create_feedback(
                run_id,
                key=item["key"],
                score=item["score"],
                comment=item.get("comment") or None,
                source_info={"evaluator": "deterministic", "origin": "pipeline"},
            )
            pushed += 1
        except BaseException:
            logger.debug("tracing: feedback %r failed", item["key"],
                         exc_info=True)
    return pushed


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

@fail_open()
def _flush_tracers() -> None:
    """Wait for the background tracer to finish POSTing this run.

    Necessary, not defensive. LangChain's tracer PATCHes the root run's
    completion from a background thread, and ``update_run`` below races it: if
    our PATCH lands first, the tracer's own flush can overwrite the metadata we
    just wrote. Feedback has the same ordering problem — it can be rejected
    outright if it arrives before the run it references exists.

    Bounded, and skipped entirely when tracing is off.
    """
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()

    # The langsmith client batches its own sends on a background thread, and
    # feedback that arrives before the run it references can be rejected.
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except BaseException:
            logger.debug("tracing: client flush failed", exc_info=True)


@fail_open(default=dict)
def finalize(handle: TraceHandle, state: dict) -> dict:
    """Close out a traced run: metadata, then feedback.

    Safe to call on a disabled handle, on a partial state, or after the
    pipeline itself raised — which is exactly when it is called from
    ``graph.py``'s ``finally``. Returns the metadata written, for tests and
    for callers that want the token counts locally.
    """
    if not handle.enabled:
        return {}

    # Metadata is not written here. It was written into the root run by
    # `TraceHandle.record` while that run was still open, because LangSmith
    # rejects a second update to a closed run. This function therefore only
    # pushes feedback, which is a separate resource and has no such limit.
    metadata = run_metadata(state, handle.usage)

    def publish() -> None:
        # Everything that touches the network, under one wall clock.
        _flush_tracers()

        pushed = push_feedback(handle.run_id, state)
        logger.debug("tracing: finalised run %s (%d feedback scores)",
                     handle.run_id, pushed)

    _bounded(publish, FINALIZE_TIMEOUT_SECONDS, label="finalize")
    return metadata
