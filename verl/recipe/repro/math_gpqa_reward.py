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


def _math_reward() -> Callable[..., Any]:
    from verl.utils.reward_score.ttrl_math import reward_func

    return reward_func


def _gpqa_reward() -> Callable[..., Any]:
    module_path = Path(__file__).with_name("gpqa_reward.py")
    spec = importlib.util.spec_from_file_location("_pov_gpqa_reward", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GPQA reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_func


def reward_func(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> Any:
    """Score one rollout with the verifier appropriate to its dataset."""

    scorer = _gpqa_reward() if is_gpqa_source(data_source) else _math_reward()
    return scorer(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


__all__ = ["is_gpqa_source", "reward_func"]
