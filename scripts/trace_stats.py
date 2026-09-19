"""Aggregate the cost and quality numbers across traced runs.

The README used to quote figures copied by hand from one run. That is an
anecdote pretending to be a benchmark: live search returns different results
every time and the model is not fully deterministic even at temperature 0, so a
single run's token count and groundedness score are a sample, not a measurement.
Worse, hand-copied numbers rot silently — nothing tells you they no longer match
the code.

This reads the runs back out of LangSmith instead, so the figures are
reproducible by anyone with access to the project and improve as runs
accumulate. It reports a median and the full range, because the range is the
honest part: it says how much a single run can be trusted.

    python scripts/trace_stats.py                    # default project, last 50
    python scripts/trace_stats.py --company Stripe   # one subject only
    python scripts/trace_stats.py --markdown         # paste-ready README table

Reads only. Costs nothing, runs no models, and never writes to LangSmith.
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

TIERS = ("fast", "reasoning", "synthesis")


def _median(values):
    return statistics.median(values) if values else 0


def _fmt_range(values, fmt="{:.0f}"):
    """Median, with the spread beside it — or just the value when N is 1."""
    if not values:
        return "—"
    med = _median(values)
    if len(values) == 1:
        return fmt.format(med)
    lo, hi = min(values), max(values)
    if lo == hi:
        return fmt.format(med)
    return f"{fmt.format(med)} ({fmt.format(lo)}–{fmt.format(hi)})"


def collect(client, project, company=None, limit=50):
    """Root runs carrying our metadata, newest first.

    Runs without ``dd.tokens_by_tier`` are skipped rather than counted as zero:
    they predate the metadata landing correctly, and averaging them in would
    quietly halve every figure.
    """
    runs = []
    for run in client.list_runs(project_name=project, is_root=True, limit=limit):
        meta = (run.extra or {}).get("metadata", {})
        if "dd.tokens_by_tier" not in meta:
            continue
        if company and meta.get("dd.company", "").lower() != company.lower():
            continue
        if run.status != "success":
            continue
        runs.append((run, meta))
    return runs


def summarise(client, runs):
    """Fold the runs into one table's worth of numbers."""
    out = {
        "n": len(runs),
        "latency": [],
        "total_tokens": [],
        "claims": [],
        "conflicts": [],
        "iterations": [],
        "tier_tokens": defaultdict(list),
        "tier_calls": defaultdict(list),
        "feedback": defaultdict(list),
        "convergence": defaultdict(int),
    }

    for run, meta in runs:
        if run.end_time and run.start_time:
            out["latency"].append((run.end_time - run.start_time).total_seconds())
        out["total_tokens"].append(meta.get("dd.total_tokens") or run.total_tokens or 0)
        out["claims"].append(meta.get("dd.claims", 0))
        out["conflicts"].append(meta.get("dd.conflicts", 0))
        out["iterations"].append(meta.get("dd.planning_iterations", 0))
        out["convergence"][meta.get("dd.convergence", "unknown")] += 1

        by_tier = meta.get("dd.tokens_by_tier") or {}
        calls = meta.get("dd.calls_by_tier") or {}
        for tier in TIERS:
            out["tier_tokens"][tier].append((by_tier.get(tier) or {}).get("total", 0))
            out["tier_calls"][tier].append(calls.get(tier, 0))

    # Feedback is a separate resource, so it is a separate fetch.
    ids = [run.id for run, _ in runs]
    if ids:
        for fb in client.list_feedback(run_ids=ids):
            if fb.score is not None:
                out["feedback"][fb.key].append(fb.score)

    return out


def render(stats, project, company, markdown=False):
    n = stats["n"]
    if n == 0:
        print("No traced runs carrying dd.* metadata were found.")
        print(f"  project: {project}" + (f" | company: {company}" if company else ""))
        print("\nRun the pipeline with LANGSMITH_TRACING=true first.")
        return

    subject = company or "all companies"
    bar = "-" * 66

    print(bar)
    print(f"  {n} successful traced run(s) — {subject} — project {project!r}")
    if n < 3:
        print("  WARNING: N is too small to quote as a benchmark. Treat as")
        print("           illustrative of scale until N >= 5.")
    print(bar)

    print("\n  Per run (median, range across runs)\n")
    rows = [
        ("Wall clock", _fmt_range(stats["latency"], "{:.0f}s")),
        ("Total tokens", _fmt_range(stats["total_tokens"], "{:,.0f}")),
        ("Claims extracted", _fmt_range(stats["claims"])),
        ("Conflicts detected", _fmt_range(stats["conflicts"])),
        ("Planning iterations", _fmt_range(stats["iterations"])),
    ]
    for label, value in rows:
        print(f"    {label:<24} {value}")

    total_median = _median(stats["total_tokens"]) or 1
    print("\n  Cost by routing tier\n")
    print(f"    {'Tier':<12} {'Calls':<14} {'Tokens':<22} Share")
    for tier in TIERS:
        tokens = stats["tier_tokens"][tier]
        share = _median(tokens) / total_median * 100 if tokens else 0
        print(f"    {tier:<12} {_fmt_range(stats['tier_calls'][tier]):<14} "
              f"{_fmt_range(tokens, '{:,.0f}'):<22} {share:.0f}%")

    if stats["feedback"]:
        print("\n  Quality (deterministic evaluators, pushed as feedback)\n")
        for key in sorted(stats["feedback"]):
            print(f"    {key:<42} {_fmt_range(stats['feedback'][key], '{:.3f}')}")

    print("\n  Why the refinement loop stopped\n")
    for reason, count in sorted(stats["convergence"].items(), key=lambda x: -x[1]):
        print(f"    {reason:<24} {count}/{n} run(s)")

    if markdown:
        print("\n" + bar)
        print("  Paste-ready for the README")
        print(bar + "\n")
        print(f"Measured over **{n} traced run(s)**"
              + (f" of {company}" if company else "")
              + ". Median, with the range across runs in brackets;")
        print("regenerate with `python scripts/trace_stats.py --markdown`.\n")
        print("| | |")
        print("|---|---|")
        for label, value in rows:
            print(f"| {label} | {value} |")
        for key in sorted(stats["feedback"]):
            print(f"| `{key}` | {_fmt_range(stats['feedback'][key], '{:.3f}')} |")
        print("\n| Tier | Calls | Tokens | Share |")
        print("|---|---|---|---|")
        for tier in TIERS:
            tokens = stats["tier_tokens"][tier]
            share = _median(tokens) / total_median * 100 if tokens else 0
            print(f"| `{tier}` | {_fmt_range(stats['tier_calls'][tier])} "
                  f"| {_fmt_range(tokens, '{:,.0f}')} | {share:.0f}% |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT"),
                        help="LangSmith project (default: $LANGSMITH_PROJECT)")
    parser.add_argument("--company", help="Only runs for this subject")
    parser.add_argument("--limit", type=int, default=50,
                        help="How many recent root runs to scan")
    parser.add_argument("--markdown", action="store_true",
                        help="Also emit a paste-ready README table")
    args = parser.parse_args()

    if not args.project:
        print("No project. Set LANGSMITH_PROJECT or pass --project.", file=sys.stderr)
        return 2
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set.", file=sys.stderr)
        return 2

    from langsmith import Client

    client = Client()
    runs = collect(client, args.project, args.company, args.limit)
    render(summarise(client, runs), args.project, args.company, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
