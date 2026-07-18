#!/usr/bin/env python3
"""Compute exact repeated-evaluation metrics from a verl validation JSONL.

For N responses per question, this script reports both common meanings of
"best@N":

* best@N: the best full-dataset accuracy among the N response slots.  This is
  the convention used by the repeated-run tables in this project.
* pass@N: the fraction of questions with at least one correct response.

verl's built-in ``best@N/mean`` is a bootstrap statistic; this script does not
use it for either exact metric above.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _binary_score(row: dict[str, Any], line_number: int) -> float:
    value = row.get("acc", row.get("score"))
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"line {line_number}: missing/non-numeric acc or score: {value!r}")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"line {line_number}: expected score in [0, 1], got {value}")
    return float(value >= 0.5)


def _group_key(row: dict[str, Any], line_number: int) -> tuple[str, str]:
    sample_id = row.get("sample_id")
    if sample_id is not None and str(sample_id):
        return "sample_id", str(sample_id)

    prompt = row.get("input")
    ground_truth = row.get("gts")
    if prompt is None or ground_truth is None:
        raise ValueError(
            f"line {line_number}: need sample_id, or both input and gts, to group responses"
        )
    return "prompt+gt", json.dumps([prompt, ground_truth], ensure_ascii=False, sort_keys=True)


def summarize_rows(
    rows: list[dict[str, Any]],
    n_responses: int,
    expected_questions: int | None = None,
) -> dict[str, Any]:
    if n_responses <= 0:
        raise ValueError(f"n_responses must be positive, got {n_responses}")

    grouped: OrderedDict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = OrderedDict()
    for line_number, row in enumerate(rows, start=1):
        key = _group_key(row, line_number)
        grouped.setdefault(key, []).append((_binary_score(row, line_number), row))

    if not grouped:
        raise ValueError("validation JSONL contains no rows")
    if expected_questions is not None and len(grouped) != expected_questions:
        raise ValueError(
            f"expected {expected_questions} questions, found {len(grouped)}; "
            "refusing to report a partial evaluation"
        )

    bad_counts = {key[1]: len(items) for key, items in grouped.items() if len(items) != n_responses}
    if bad_counts:
        preview = ", ".join(f"{key}={count}" for key, count in list(bad_counts.items())[:8])
        raise ValueError(
            f"every question must have exactly {n_responses} responses; mismatches: {preview}"
        )

    score_matrix = [[score for score, _ in items] for items in grouped.values()]
    question_count = len(score_matrix)
    run_accuracies = [
        sum(question_scores[slot] for question_scores in score_matrix) / question_count
        for slot in range(n_responses)
    ]

    avg_at_n = sum(run_accuracies) / n_responses
    best_at_n = max(run_accuracies)
    worst_at_n = min(run_accuracies)
    pass_at_n = sum(max(question_scores) for question_scores in score_matrix) / question_count

    flat_rows = [row for items in grouped.values() for _, row in items]
    token_counts = [
        float(row["response_tokens"])
        for row in flat_rows
        if isinstance(row.get("response_tokens"), (int, float))
    ]
    format_scores = [
        float(row["format_score"])
        for row in flat_rows
        if isinstance(row.get("format_score"), (int, float, bool))
    ]

    return {
        "dataset": "GPQA",
        "questions": question_count,
        "responses_per_question": n_responses,
        "total_responses": len(rows),
        "avg_at_n": avg_at_n,
        "best_at_n": best_at_n,
        "worst_at_n": worst_at_n,
        "pass_at_n": pass_at_n,
        "run_accuracies": run_accuracies,
        "mean_response_tokens": (sum(token_counts) / len(token_counts)) if token_counts else None,
        "boxed_format_rate": (sum(format_scores) / len(format_scores)) if format_scores else None,
        "definitions": {
            "avg_at_n": "mean accuracy across the N full-dataset response slots",
            "best_at_n": "maximum full-dataset accuracy among the N response slots",
            "worst_at_n": "minimum full-dataset accuracy among the N response slots",
            "pass_at_n": "fraction of questions with at least one correct response among N",
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="verl validation 0.jsonl")
    parser.add_argument("--n", type=int, default=16, help="responses per question")
    parser.add_argument(
        "--expected-questions",
        type=int,
        default=198,
        help="fail if the evaluation is incomplete; use 0 to disable",
    )
    parser.add_argument("--output", type=Path, help="optional summary JSON path")
    args = parser.parse_args()

    summary = summarize_rows(
        load_jsonl(args.input),
        n_responses=args.n,
        expected_questions=args.expected_questions or None,
    )

    print("=" * 64)
    print(f"GPQA exact repeated evaluation ({summary['questions']} questions)")
    print(f"avg@{args.n}:  {_percent(summary['avg_at_n'])}")
    print(f"best@{args.n}: {_percent(summary['best_at_n'])}  (best full-dataset run)")
    print(f"worst@{args.n}: {_percent(summary['worst_at_n'])}")
    print(f"pass@{args.n}: {_percent(summary['pass_at_n'])}  (any correct per question)")
    if summary["mean_response_tokens"] is not None:
        print(f"mean response tokens: {summary['mean_response_tokens']:.2f}")
    if summary["boxed_format_rate"] is not None:
        print(f"boxed format rate: {_percent(summary['boxed_format_rate'])}")
    print("=" * 64)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Summary JSON: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
