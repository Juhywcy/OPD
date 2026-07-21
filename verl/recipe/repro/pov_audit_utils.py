"""Small, dependency-free statistics helpers for the POV audit scripts."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: Iterable[float | None],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    result: dict[str, float | int | None] = {
        "mean": statistics.fmean(clean) if clean else None,
        "ci_low": None,
        "ci_high": None,
        "n": len(clean),
    }
    if not clean or samples <= 0:
        return result
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choices(clean, k=len(clean))) for _ in range(samples)]
    result["ci_low"] = percentile(draws, 0.025)
    result["ci_high"] = percentile(draws, 0.975)
    return result


def cluster_bootstrap_mean_ci(
    values_by_cluster: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    clusters = {
        str(cluster): [float(value) for value in values if math.isfinite(float(value))]
        for cluster, values in values_by_cluster.items()
    }
    clusters = {cluster: values for cluster, values in clusters.items() if values}
    observed = [value for values in clusters.values() for value in values]
    result: dict[str, float | int | None] = {
        "mean": statistics.fmean(observed) if observed else None,
        "ci_low": None,
        "ci_high": None,
        "n": len(observed),
        "clusters": len(clusters),
    }
    if not clusters or samples <= 0:
        return result

    cluster_ids = list(clusters)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled_ids = rng.choices(cluster_ids, k=len(cluster_ids))
        sampled_values = [value for cluster_id in sampled_ids for value in clusters[cluster_id]]
        draws.append(statistics.fmean(sampled_values))
    result["ci_low"] = percentile(draws, 0.025)
    result["ci_high"] = percentile(draws, 0.975)
    return result


def _window_sides(
    trajectory: dict[str, Any],
    *,
    boundary: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = trajectory.get("windows", [])
    # A pseudo-boundary can fall inside a control window. Treat that containing
    # window as post, matching the protocol's treatment of the trigger window.
    pre = [window for window in windows if int(window["end"]) <= boundary]
    post = [window for window in windows if int(window["end"]) > boundary]
    return pre, post


def _side_metrics(windows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    cosine_values = [window.get("cosine") for window in windows]
    return {
        "window_count": len(windows),
        "conflict_rate": mean_or_none(float(bool(window.get("conflict"))) for window in windows),
        "mean_cosine": mean_or_none(cosine_values),
        "valid_cosine_windows": sum(value is not None for value in cosine_values),
    }


def match_prefix_controls(
    trajectories: list[dict[str, Any]],
    *,
    length_caliper: float,
) -> list[dict[str, Any]]:
    """Greedily match triggered trajectories to controls without replacement."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[(str(trajectory["prompt_id"]), int(trajectory["outcome_class"]))].append(trajectory)

    pairs: list[dict[str, Any]] = []
    for (prompt_id, outcome_class), group in grouped.items():
        triggered = [item for item in group if bool(item.get("triggered"))]
        controls = [item for item in group if not bool(item.get("triggered"))]
        available = {str(item["trajectory_id"]): item for item in controls}

        # Hard-to-match trajectories go first so an easy match cannot consume
        # their only valid control.
        def eligible_count(item: dict[str, Any]) -> int:
            length = max(1, int(item["response_length"]))
            return sum(
                abs(int(control["response_length"]) - length) / length <= length_caliper
                for control in controls
            )

        for treatment in sorted(triggered, key=lambda item: (eligible_count(item), str(item["trajectory_id"]))):
            treatment_length = max(1, int(treatment["response_length"]))
            candidates = [
                control
                for control in available.values()
                if abs(int(control["response_length"]) - treatment_length) / treatment_length <= length_caliper
            ]
            if not candidates:
                continue
            control = min(
                candidates,
                key=lambda item: (abs(int(item["response_length"]) - treatment_length), str(item["trajectory_id"])),
            )
            del available[str(control["trajectory_id"])]

            treatment_boundary = int(treatment["horizon_token"])
            normalized_boundary = treatment_boundary / treatment_length
            control_boundary = int(round(normalized_boundary * max(1, int(control["response_length"]))))
            treatment_pre, treatment_post = _window_sides(treatment, boundary=treatment_boundary)
            control_pre, control_post = _window_sides(control, boundary=control_boundary)
            if not treatment_pre or not treatment_post or not control_pre or not control_post:
                continue

            treatment_pre_metrics = _side_metrics(treatment_pre)
            treatment_post_metrics = _side_metrics(treatment_post)
            control_pre_metrics = _side_metrics(control_pre)
            control_post_metrics = _side_metrics(control_post)

            conflict_values = [
                treatment_pre_metrics["conflict_rate"],
                treatment_post_metrics["conflict_rate"],
                control_pre_metrics["conflict_rate"],
                control_post_metrics["conflict_rate"],
            ]
            conflict_did = None
            if all(value is not None for value in conflict_values):
                conflict_did = (
                    float(treatment_post_metrics["conflict_rate"])
                    - float(treatment_pre_metrics["conflict_rate"])
                    - float(control_post_metrics["conflict_rate"])
                    + float(control_pre_metrics["conflict_rate"])
                )

            cosine_values = [
                treatment_pre_metrics["mean_cosine"],
                treatment_post_metrics["mean_cosine"],
                control_pre_metrics["mean_cosine"],
                control_post_metrics["mean_cosine"],
            ]
            cosine_did = None
            if all(value is not None for value in cosine_values):
                cosine_did = (
                    float(treatment_post_metrics["mean_cosine"])
                    - float(treatment_pre_metrics["mean_cosine"])
                    - float(control_post_metrics["mean_cosine"])
                    + float(control_pre_metrics["mean_cosine"])
                )

            pairs.append(
                {
                    "prompt_id": prompt_id,
                    "outcome_class": outcome_class,
                    "triggered_trajectory_id": treatment["trajectory_id"],
                    "control_trajectory_id": control["trajectory_id"],
                    "triggered_length": treatment_length,
                    "control_length": int(control["response_length"]),
                    "normalized_boundary": normalized_boundary,
                    "triggered_boundary": treatment_boundary,
                    "control_boundary": control_boundary,
                    "triggered_pre": treatment_pre_metrics,
                    "triggered_post": treatment_post_metrics,
                    "control_pre": control_pre_metrics,
                    "control_post": control_post_metrics,
                    "conflict_did": conflict_did,
                    "cosine_did": cosine_did,
                }
            )
    return pairs
