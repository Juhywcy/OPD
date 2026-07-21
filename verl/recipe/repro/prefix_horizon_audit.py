#!/usr/bin/env python3
"""Run the matched-control prefix-horizon analysis on POV audit output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from recipe.repro.pov_audit_utils import cluster_bootstrap_mean_ci, match_prefix_controls, mean_or_none


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-records", required=True, help="batches.jsonl produced by pov_gradient_audit.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length-caliper", type=float, default=0.20, help="Maximum relative response-length gap.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_batches(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pair_side_mean(pairs: list[dict[str, Any]], group: str, side: str, metric: str) -> float | None:
    return mean_or_none(pair[f"{group}_{side}"][metric] for pair in pairs)


def main() -> None:
    args = build_parser().parse_args()
    if args.length_caliper < 0:
        raise ValueError("--length-caliper must be non-negative")

    batches = _read_batches(Path(args.audit_records))
    valid_batches = [batch for batch in batches if bool(batch.get("reference_valid"))]
    all_trajectories = [trajectory for batch in batches for trajectory in batch.get("trajectories", [])]
    valid_trajectories = [trajectory for batch in valid_batches for trajectory in batch.get("trajectories", [])]
    pairs = match_prefix_controls(valid_trajectories, length_caliper=args.length_caliper)

    conflict_by_prompt: dict[str, list[float]] = defaultdict(list)
    cosine_by_prompt: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        if pair["conflict_did"] is not None:
            conflict_by_prompt[str(pair["prompt_id"])].append(float(pair["conflict_did"]))
        if pair["cosine_did"] is not None:
            cosine_by_prompt[str(pair["prompt_id"])].append(float(pair["cosine_did"]))

    triggered_count = sum(bool(trajectory.get("triggered")) for trajectory in all_trajectories)
    eligible_triggered_count = sum(bool(trajectory.get("triggered")) for trajectory in valid_trajectories)
    summary = {
        "audit_batches": len(batches),
        "reference_valid_batches": len(valid_batches),
        "trajectory_count": len(all_trajectories),
        "reference_valid_trajectory_count": len(valid_trajectories),
        "triggered_trajectory_count": triggered_count,
        "trigger_rate": triggered_count / len(all_trajectories) if all_trajectories else None,
        "reference_valid_triggered_trajectory_count": eligible_triggered_count,
        "matched_pair_count": len(pairs),
        "matched_rate": len(pairs) / eligible_triggered_count if eligible_triggered_count else None,
        "length_caliper": args.length_caliper,
        "triggered": {
            "conflict_rate_pre": _pair_side_mean(pairs, "triggered", "pre", "conflict_rate"),
            "conflict_rate_post": _pair_side_mean(pairs, "triggered", "post", "conflict_rate"),
            "mean_cosine_pre": _pair_side_mean(pairs, "triggered", "pre", "mean_cosine"),
            "mean_cosine_post": _pair_side_mean(pairs, "triggered", "post", "mean_cosine"),
        },
        "control": {
            "conflict_rate_pre": _pair_side_mean(pairs, "control", "pre", "conflict_rate"),
            "conflict_rate_post": _pair_side_mean(pairs, "control", "post", "conflict_rate"),
            "mean_cosine_pre": _pair_side_mean(pairs, "control", "pre", "mean_cosine"),
            "mean_cosine_post": _pair_side_mean(pairs, "control", "post", "mean_cosine"),
        },
        "conflict_did": cluster_bootstrap_mean_ci(
            conflict_by_prompt, samples=args.bootstrap_samples, seed=args.seed
        ),
        "cosine_did": cluster_bootstrap_mean_ci(
            cosine_by_prompt, samples=args.bootstrap_samples, seed=args.seed + 1
        ),
        "bootstrap_unit": "prompt_id",
        "bootstrap_samples": args.bootstrap_samples,
    }
    for group in ("triggered", "control"):
        pre = summary[group]["conflict_rate_pre"]
        post = summary[group]["conflict_rate_post"]
        summary[group]["conflict_rate_post_minus_pre"] = post - pre if pre is not None and post is not None else None
        pre_cosine = summary[group]["mean_cosine_pre"]
        post_cosine = summary[group]["mean_cosine_post"]
        summary[group]["mean_cosine_post_minus_pre"] = (
            post_cosine - pre_cosine if pre_cosine is not None and post_cosine is not None else None
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "matched_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    (output_dir / "prefix_horizon_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
