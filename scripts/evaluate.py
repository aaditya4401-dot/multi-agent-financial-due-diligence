"""Evaluate saved due diligence reports.

Groundedness is already embedded in each report by the pipeline's evaluation
node, so this reads it rather than recomputing it — the claim graph it was
measured against is not in the saved artefact, and recomputing from prose alone
would be a different, weaker check pretending to be the same one.

The rubric judge is opt-in because it costs a model call per report.

    python scripts/evaluate.py reports/*.json
    python scripts/evaluate.py reports/*.json --judge

Produce reports to evaluate with::

    python -m src.graph Stripe --save reports/stripe.json
"""

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.eval.judge import judge_report  # noqa: E402


def _load(paths: list[str]) -> list[tuple[str, dict]]:
    reports = []
    for path in paths:
        try:
            reports.append((path, json.loads(pathlib.Path(path).read_text())))
        except Exception as exc:
            print(f"  ! skipping {path}: {exc}")
    return reports


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Saved report JSON files.")
    parser.add_argument("--judge", action="store_true",
                        help="Also run the rubric judge (costs one call each).")
    args = parser.parse_args()

    reports = _load(args.reports)
    if not reports:
        print("No readable reports.")
        return 1

    print(f"\n{'report':<28} {'verdict':<22} {'grounded':>9} {'judge':>7}")
    print("-" * 70)

    grounded_scores, judge_scores, failures = [], [], []

    for path, report in reports:
        name = pathlib.Path(path).name[:27]
        evaluation = report.get("evaluation") or {}
        grounded = evaluation.get("score")
        thesis = report.get("thesis") or {}
        verdict = thesis.get("recommendation") or report.get("overall_verdict", "?")

        judged = ""
        if args.judge:
            verdict_obj = await judge_report(report)
            if verdict_obj.scores:
                judge_scores.append(verdict_obj.mean)
                weakest = verdict_obj.weakest
                judged = f"{verdict_obj.mean:.1f}"
                if weakest and weakest.score <= 2:
                    failures.append(
                        f"{name}: {weakest.criterion} scored {weakest.score} — "
                        f"{weakest.justification[:100]}")

        if grounded is not None:
            grounded_scores.append(grounded)
        shown = f"{grounded:.0%}" if grounded is not None else "n/a"
        print(f"{name:<28} {verdict[:21]:<22} {shown:>9} {judged:>7}")

        for check in evaluation.get("checks", []):
            if not check.get("passed"):
                failures.append(f"{name}: {check['name']} — {check['detail']}")

    print("-" * 70)
    if grounded_scores:
        print(f"mean groundedness : {sum(grounded_scores)/len(grounded_scores):.0%} "
              f"across {len(grounded_scores)} report(s)")
    if judge_scores:
        print(f"mean judge score  : {sum(judge_scores)/len(judge_scores):.2f} / 5")

    if failures:
        print(f"\n{len(failures)} issue(s):")
        for failure in failures:
            print(f"  - {failure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
