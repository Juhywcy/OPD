"""Tests for the mixed math/GPQA reward router."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_gpqa_reward import _fallback_math_reward, is_gpqa_source, reward_func


class MathGpqaRewardRouterTest(unittest.TestCase):
    def test_source_detection_is_case_insensitive(self):
        self.assertTrue(is_gpqa_source("GPQA"))
        self.assertTrue(is_gpqa_source("Idavidrein/GPQA-Diamond"))
        self.assertFalse(is_gpqa_source("AMC23"))

    def test_routes_gpqa(self):
        with patch("math_gpqa_reward._gpqa_reward") as get_gpqa:
            get_gpqa.return_value.return_value = {"score": 1.0, "acc": True, "pred": "B"}
            result = reward_func("GPQA", r"\\boxed{B}", "B")
        self.assertEqual(
            result,
            {"score": 1.0, "acc": 1.0},
        )
        get_gpqa.return_value.assert_called_once()

    def test_routes_math(self):
        with patch("math_gpqa_reward._math_reward") as get_math:
            get_math.return_value.return_value = {"score": 0.0, "acc": False, "pred": "24"}
            result = reward_func("AIME24", "answer", "25")
        self.assertEqual(
            result,
            {"score": 0.0, "acc": 0.0},
        )
        get_math.return_value.assert_called_once()

    def test_fallback_accepts_last_boxed_answer(self):
        result = _fallback_math_reward("AIME24", r"Therefore, \\boxed{25}.", "25")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["acc"])
        self.assertEqual(result["pred"], "25")

    def test_fallback_uses_the_last_box(self):
        result = _fallback_math_reward("AIME24", r"First \\boxed{24}, finally \\boxed{25}.", "25")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["pred"], "25")

    def test_fallback_rejects_wrong_box_even_if_answer_line_is_correct(self):
        result = _fallback_math_reward("AIME24", "Answer: 25\n" + r"Finally, \\boxed{24}.", "25")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["acc"])

    def test_fallback_supports_unboxed_answer_line(self):
        result = _fallback_math_reward("MATH", "Reasoning\nAnswer: 25", "25")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["acc"])


if __name__ == "__main__":
    unittest.main()
