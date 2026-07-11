# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import math
import os
import subprocess
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("l2s_stage1.py")
LAUNCH_SCRIPT = Path(__file__).with_name("deepseek_r1_distill_llama_8b_stage1.sh")
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def load_l2s_stage1_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"missing L2S stage-1 implementation: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("l2s_stage1_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestL2SStage1(unittest.TestCase):
    def setUp(self):
        self.l2s = load_l2s_stage1_module()

    def test_correct_reward_matches_original_formula(self):
        reward = self.l2s.compute_length_decayed_reward(is_correct=True, response_length=4096)
        self.assertTrue(math.isclose(reward, 0.2, rel_tol=1e-12, abs_tol=1e-12))

    def test_correct_reward_at_zero_length_is_beta(self):
        reward = self.l2s.compute_length_decayed_reward(is_correct=True, response_length=0)
        self.assertEqual(reward, 2.0)

    def test_incorrect_reward_is_zero(self):
        reward = self.l2s.compute_length_decayed_reward(is_correct=False, response_length=1)
        self.assertEqual(reward, 0.0)

    def test_negative_response_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.l2s.compute_length_decayed_reward(is_correct=True, response_length=-1)

    def test_current_dataset_sources_map_to_math_verifier_names(self):
        self.assertEqual(self.l2s.normalize_math_data_source("DeepScaler"), "math_dapo")
        self.assertEqual(self.l2s.normalize_math_data_source("AIME24"), "aime24")

    def test_known_data_source_is_preserved(self):
        self.assertEqual(self.l2s.normalize_math_data_source("HuggingFaceH4/MATH-500"), "HuggingFaceH4/MATH-500")

    def test_custom_scorer_uses_original_deepscaler_verifier(self):
        """The custom scorer must use the original prompt-plus-response verifier."""
        bootstrap_path = Path(__file__).with_name("register_l2s_stage1.py")
        source = bootstrap_path.read_text(encoding="utf-8")
        self.assertIn("deepscaler_reward_fn", source)
        self.assertNotIn("strict_box_verify", source)

    def test_terminal_reward_is_placed_only_at_last_valid_token(self):
        row = [0.0] * 6
        self.l2s.place_terminal_reward(row, valid_response_length=4, reward=0.75)
        self.assertEqual(row, [0.0, 0.0, 0.0, 0.75, 0.0, 0.0])

    def test_empty_response_keeps_reward_row_zero(self):
        row = [0.0] * 4
        self.l2s.place_terminal_reward(row, valid_response_length=0, reward=2.0)
        self.assertEqual(row, [0.0, 0.0, 0.0, 0.0])

    def test_reward_as_advantage_broadcasts_over_valid_tokens(self):
        token_rewards = [[0.0, 0.0, 0.5, 0.0], [0.0, 1.25, 0.0, 0.0]]
        response_mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        advantages = self.l2s.broadcast_sequence_rewards(token_rewards, response_mask)
        self.assertEqual(advantages, [[0.5, 0.5, 0.5, 0.0], [1.25, 1.25, 0.0, 0.0]])

    def test_deepseek_launch_uses_only_l2s_stage1_components(self):
        environment = os.environ.copy()
        environment["DRY_RUN"] = "1"
        rendered_command = subprocess.check_output(
            ["bash", str(LAUNCH_SCRIPT)],
            env=environment,
            text=True,
        )

        self.assertIn("python3 -m recipe.repro.main_ppo", rendered_command)
        self.assertIn("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", rendered_command)
        self.assertIn(str(WORKSPACE_ROOT / "datasets/test_data/DeepScaler/train.parquet"), rendered_command)
        self.assertIn(str(WORKSPACE_ROOT / "datasets/test_data/AIME24/test.parquet"), rendered_command)
        self.assertIn("algorithm.adv_estimator=l2s_stage1_reward_as_advantage", rendered_command)
        self.assertIn("reward_model.enable=False", rendered_command)
        self.assertIn("reward_model.reward_manager=l2s_stage1", rendered_command)
        self.assertIn("custom_reward_function.path=", rendered_command)
        self.assertIn("register_l2s_stage1.py", rendered_command)
        self.assertNotIn("prefix_outcome", rendered_command)
        self.assertNotIn("token_reward_direct", rendered_command)
        self.assertNotIn("reward_manager=opd", rendered_command.lower())
        self.assertNotIn("opd_alpha", rendered_command.lower())


if __name__ == "__main__":
    unittest.main()
