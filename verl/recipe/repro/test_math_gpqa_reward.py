"""Tests for the mixed math/GPQA reward router."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_gpqa_reward import is_gpqa_source, reward_func


class MathGpqaRewardRouterTest(unittest.TestCase):
    def test_source_detection_is_case_insensitive(self):
        self.assertTrue(is_gpqa_source("GPQA"))
        self.assertTrue(is_gpqa_source("Idavidrein/GPQA-Diamond"))
        self.assertFalse(is_gpqa_source("AMC23"))

    def test_routes_gpqa(self):
        with patch("math_gpqa_reward._gpqa_reward") as get_gpqa:
            get_gpqa.return_value.return_value = {"score": 1.0}
            result = reward_func("GPQA", "answer", "B")
        self.assertEqual(result, {"score": 1.0})
        get_gpqa.return_value.assert_called_once()

    def test_routes_math(self):
        with patch("math_gpqa_reward._math_reward") as get_math:
            get_math.return_value.return_value = {"score": 0.0}
            result = reward_func("AIME24", "answer", "25")
        self.assertEqual(result, {"score": 0.0})
        get_math.return_value.assert_called_once()


if __name__ == "__main__":
    unittest.main()
