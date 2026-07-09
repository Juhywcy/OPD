import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("eval_amc23_olympiad.sh")


class EvalLauncherConfigTest(unittest.TestCase):
    def test_ppo_mini_batch_is_not_dummy_train_batch(self):
        text = SCRIPT.read_text()

        self.assertIn("DP_SIZE=$((N_GPUS / PARALLEL_SIZE))", text)
        self.assertRegex(text, r"PPO_MINI_BATCH_SIZE=\$\{PPO_MINI_BATCH_SIZE:-\$\{DP_SIZE\}\}")
        self.assertIn('actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"', text)
        self.assertNotIn('actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}"', text)

    def test_default_math_keeps_normalized_ppo_mini_batch_positive_on_four_gpus(self):
        text = SCRIPT.read_text()
        n_gpus = 4
        parallel_size = 1
        rollout_n = int(re.search(r"actor_rollout_ref\.rollout\.n=(\d+)", text).group(1))

        dp_size = n_gpus // parallel_size
        ppo_mini_batch_size = dp_size
        normalized = ppo_mini_batch_size * rollout_n // dp_size

        self.assertGreater(normalized, 0)


if __name__ == "__main__":
    unittest.main()
