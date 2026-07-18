"""Rule-based GPQA-Diamond scorer for verl validation.

The local GPQA parquet uses ``data_source=GPQA`` and stores an A/B/C/D letter
in ``reward_model.ground_truth``.  This scorer deliberately does not use the
generic mathematical equivalence checker: it extracts an explicit final
multiple-choice answer and compares the normalized letter exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_CHOICES = frozenset({"A", "B", "C", "D"})
_BOX_START_RE = re.compile(r"\\boxed\s*\{")
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:therefore\s*,?\s*)?(?:the\s+)?(?:final\s+)?answer\s*"
    r"(?:is\s*|[:=]\s*)[*_`]*[\(\[]?([A-D])[\)\]]?[*_`]*[.!]?\s*$"
)
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textrm|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}",
    flags=re.IGNORECASE,
)


def _boxed_contents(text: str) -> list[str]:
    """Return every balanced ``\\boxed{...}`` payload in source order."""

    contents: list[str] = []
    for match in _BOX_START_RE.finditer(text):
        depth = 1
        cursor = match.end()
        start = cursor
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            contents.append(text[start : cursor - 1])
    return contents


def _normalize_choice(candidate: Any) -> str:
    """Normalize one candidate to A/B/C/D, or return an empty string."""

    text = str(candidate).strip()
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_WRAPPER_RE.sub(r"\1", text)

    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("$", "").strip()
    match = re.fullmatch(
        r"(?ix)\s*(?:(?:final\s+)?(?:answer|option|choice)\s*(?:is\s*)?(?::|=)?\s*)?"
        r"[\(\[]?([A-D])[\)\]]?[\s.,;:!?]*",
        text,
    )
    return match.group(1).upper() if match else ""


def extract_gpqa_choice(response: Any) -> tuple[str, str]:
    """Extract the last explicit answer and its source (``boxed``/``explicit``)."""

    text = str(response)

    # The prompt explicitly requests \boxed{}, so prefer the last valid box.
    for payload in reversed(_boxed_contents(text)):
        choice = _normalize_choice(payload)
        if choice:
            return choice, "boxed"

    # Be robust to checkpoints whose chat template produces a dedicated
    # ``Answer: D`` line.  Do not accept a bare "option D" from ordinary CoT.
    explicit_matches = list(_EXPLICIT_ANSWER_RE.finditer(text))
    if explicit_matches:
        return explicit_matches[-1].group(1).upper(), "explicit"

    return "", "none"


def _sample_id(extra_info: Any) -> str:
    if isinstance(extra_info, Mapping):
        for key in ("index", "id", "sample_id"):
            value = extra_info.get(key)
            if value is not None and str(value):
                return str(value)
    return ""


def reward_func(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **_: Any,
) -> dict[str, Any]:
    """verl custom reward entry point.

    All returned metric fields are non-null.  String fields are retained in
    the validation JSONL but skipped by verl's numeric metric aggregator.
    """

    prediction, extraction = extract_gpqa_choice(solution_str)
    truth = _normalize_choice(ground_truth)
    correct = float(bool(truth) and prediction == truth and prediction in _CHOICES)

    return {
        "score": correct,
        "acc": correct,
        "format_score": float(extraction == "boxed"),
        "pred": prediction,
        "extracted_gt": truth,
        "extraction": extraction,
        "sample_id": _sample_id(extra_info),
        "scored_source": str(data_source),
    }


__all__ = ["extract_gpqa_choice", "reward_func"]
