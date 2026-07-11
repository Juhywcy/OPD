# Copyright 2026
#
# Standalone reproduction of DeepScaler's stage-1 answer verifier.

"""Original DeepScaler answer-verification semantics used by L2S stage 1.

The original verifier evaluates the *complete* prompt-plus-response text.  It
requires a ``<think>...</think>`` block, extracts the last ``\\boxed{}``
answer after that block, and checks it with the MathD and SymPy graders.
"""

from collections.abc import Sequence
from typing import Any

from verl.utils.reward_score.ttrl_math.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)


THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def _extract_answer(passage: str) -> str | None:
    """Match DeepScaler's ``extract_answer`` helper."""
    if "\\boxed" not in passage:
        return None
    return extract_boxed_answer(passage)


def deepscaler_reward_fn(solution_str: str, ground_truth: Any) -> dict[str, Any]:
    """Reproduce ``deepscaler.rewards.l2s_reward.deepscaler_reward_fn``.

    Returns the original binary score (``1.0`` or ``-1.0``), correctness, the
    extracted prediction, and a reason suitable for reward debugging.
    """
    if THOUGHT_DELIMITER_START not in solution_str or THOUGHT_DELIMITER_END not in solution_str:
        # Validation aggregates every numeric extra and treats strings as
        # categorical fields.  Use an empty string rather than ``None`` so
        # malformed responses cannot make metric aggregation crash.
        return {"score": -1.0, "acc": False, "pred": "", "reason": "missing_think_delimiters"}

    model_solution = solution_str.split(THOUGHT_DELIMITER_END, 1)[1]
    model_answer = _extract_answer(model_solution)
    if model_answer is None:
        return {"score": -1.0, "acc": False, "pred": "", "reason": "missing_boxed_answer"}

    ground_truths: Sequence[Any]
    if isinstance(ground_truth, (str, int, float)):
        ground_truths = [ground_truth]
    elif isinstance(ground_truth, Sequence):
        ground_truths = ground_truth
    else:
        return {"score": -1.0, "acc": False, "pred": model_answer, "reason": "invalid_ground_truth"}

    processed_ground_truths: list[str] = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            truth = _extract_answer(truth)
        if truth is not None:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return {"score": -1.0, "acc": False, "pred": model_answer, "reason": "invalid_ground_truth"}

    for truth in processed_ground_truths:
        if grade_answer_mathd(model_answer, truth) or grade_answer_sympy(model_answer, truth):
            return {"score": 1.0, "acc": True, "pred": model_answer, "reason": "correct"}

    return {"score": -1.0, "acc": False, "pred": model_answer, "reason": "incorrect"}
