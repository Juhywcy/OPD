"""Route verl validation samples to the math or GPQA-Diamond verifier.

The OPD training run evaluates mathematical free-response datasets and
GPQA-Diamond in one validation pass.  GPQA stores an A/B/C/D ground truth and
must not be sent through the symbolic-math equivalence checker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable


def is_gpqa_source(data_source: Any) -> bool:
    """Return whether ``data_source`` identifies a GPQA split."""

    return "gpqa" in str(data_source).strip().lower()


def _fallback_math_reward(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score math answers without optional symbolic-LaTeX dependencies.

    The training prompts request a final ``\\boxed{...}`` answer, so an
    available final box is authoritative.  The Minerva-style ``Answer:``
    parser is retained for datasets or generations that do not use boxes.
    """

    del data_source, extra_info, kwargs

    from verl.utils.reward_score.math_dapo import compute_score as dapo_compute_score
    from verl.utils.reward_score.math_reward import (
        compute_score as boxed_compute_score,
        last_boxed_only_string,
        remove_boxed,
    )

    solution = str(solution_str)
    truth = str(ground_truth)
    boxed_truth = last_boxed_only_string(truth)
    if boxed_truth is not None:
        truth = remove_boxed(boxed_truth)

    boxed_prediction = last_boxed_only_string(solution)
    if boxed_prediction is not None:
        correct = bool(boxed_compute_score(solution, truth))
        return {
            "score": float(correct),
            "acc": correct,
            "pred": remove_boxed(boxed_prediction),
        }

    result = dapo_compute_score(solution, truth)
    correct = bool(result.get("acc", False))
    result["score"] = float(correct)
    return result


def _math_reward() -> Callable[..., Any]:
    try:
        from verl.utils.reward_score.ttrl_math import reward_func

        return reward_func
    except ModuleNotFoundError as exc:
        if exc.name != "latex2sympy2_extended":
            raise

    # Keep Raw OPD runnable on lean training images without changing the
    # answer format expected by the original verifier.
    return _fallback_math_reward


def _gpqa_reward() -> Callable[..., Any]:
    module_path = Path(__file__).with_name("gpqa_reward.py")
    spec = importlib.util.spec_from_file_location("_pov_gpqa_reward", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GPQA reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_func


def _normalize_reward_result(result: Any, solution_str: Any) -> dict[str, Any]:
    """Return the same metric schema for every validation dataset.

    verl concatenates reward metadata across all validation loaders before it
    groups samples by dataset.  Dataset-specific keys therefore create ragged
    columns and fail validation.  Keep only fields that can be populated for
    both free-response math and GPQA.
    """

    if isinstance(result, dict):
        score = float(result.get("score", 0.0))
        acc = float(result.get("acc", score > 0.0))
        pred = result.get("pred", "")
        format_score = result.get("format_score", r"\boxed" in str(solution_str))
    else:
        score = float(result)
        acc = float(score > 0.0)
        pred = ""
        format_score = float(r"\boxed" in str(solution_str))

    return {
        "score": score,
        "acc": acc,
        "format_score": float(format_score),
        "pred": "" if pred is None else str(pred),
    }


def reward_func(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> Any:
    """Score one rollout with the verifier appropriate to its dataset."""

    scorer = _gpqa_reward() if is_gpqa_source(data_source) else _math_reward()
    result = scorer(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    return _normalize_reward_result(result, solution_str)


__all__ = ["is_gpqa_source", "reward_func"]
