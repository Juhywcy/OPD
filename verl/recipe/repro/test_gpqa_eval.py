#!/usr/bin/env python3
"""Unit tests for the standalone GPQA validation helpers."""

from __future__ import annotations

import unittest

from gpqa_reward import extract_gpqa_choice, reward_func
from summarize_gpqa_avg16_best16 import summarize_rows


class GPQARewardTest(unittest.TestCase):
    def test_boxed_letter_and_latex_wrapper(self):
        self.assertEqual(extract_gpqa_choice(r"Reasoning. \boxed{D}"), ("D", "boxed"))
        self.assertEqual(extract_gpqa_choice(r"Reasoning. \boxed{\text{c}}"), ("C", "boxed"))

    def test_last_valid_box_wins(self):
        self.assertEqual(extract_gpqa_choice(r"Try \boxed{A}; final \boxed{B}."), ("B", "boxed"))

    def test_explicit_answer_fallback(self):
        self.assertEqual(extract_gpqa_choice("Therefore, the final answer is (b)."), ("B", "explicit"))

    def test_ordinary_option_discussion_is_not_an_answer(self):
        self.assertEqual(extract_gpqa_choice("Option D is inconsistent with the premise."), ("", "none"))

    def test_reward_fields_are_complete(self):
        result = reward_func(
            data_source="GPQA",
            solution_str=r"Conclusion: \boxed{A}",
            ground_truth="A",
            extra_info={"index": "GPQA-7"},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format_score"], 1.0)
        self.assertEqual(result["pred"], "A")
        self.assertEqual(result["sample_id"], "GPQA-7")
        self.assertTrue(all(value is not None for value in result.values()))


class GPQASummaryTest(unittest.TestCase):
    def test_run_best_and_question_pass_are_both_reported(self):
        rows = [
            {"sample_id": "q1", "acc": 1.0, "response_tokens": 10},
            {"sample_id": "q1", "acc": 0.0, "response_tokens": 20},
            {"sample_id": "q2", "acc": 1.0, "response_tokens": 30},
            {"sample_id": "q2", "acc": 0.0, "response_tokens": 40},
        ]
        summary = summarize_rows(rows, n_responses=2, expected_questions=2)
        self.assertEqual(summary["run_accuracies"], [1.0, 0.0])
        self.assertEqual(summary["avg_at_n"], 0.5)
        self.assertEqual(summary["best_at_n"], 1.0)
        self.assertEqual(summary["worst_at_n"], 0.0)
        self.assertEqual(summary["pass_at_n"], 1.0)
        self.assertEqual(summary["mean_response_tokens"], 25.0)

    def test_incomplete_question_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly 2 responses"):
            summarize_rows([{"sample_id": "q1", "score": 1.0}], n_responses=2)


if __name__ == "__main__":
    unittest.main()
