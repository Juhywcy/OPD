import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("make_pov_comparison.py")
spec = importlib.util.spec_from_file_location("pov_figure", MODULE_PATH)
figure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(figure)


class FigureLayoutTest(unittest.TestCase):
    def test_final_text_is_at_least_7_5pt(self):
        figure.FONT_SIZES_USED.clear()
        figure.build_content()
        self.assertTrue(figure.FONT_SIZES_USED)
        self.assertGreaterEqual(
            min(map(figure.effective_font_size, figure.FONT_SIZES_USED)),
            7.5,
        )

    def test_figure_uses_only_compact_copy(self):
        content = figure.build_content()
        required = [
            "Sparse outcome reward",
            "Dense probability reward",
            "Dense, aligned, prefix-safe",
        ]
        removed = [
            "Reward Estimators for On-Policy Reasoning Distillation",
            "POV advantage:",
            "prefix drift may be rewarded",
        ]
        self.assertTrue(all(label in content for label in required))
        self.assertTrue(all(label not in content for label in removed))

    def test_reward_elements_do_not_collide(self):
        self.assertGreater(figure.GRPO_ARROW_END_X, figure.GRPO_REWARD_X)
        self.assertLess(
            figure.GRPO_ARROW_END_X,
            figure.GRPO_REWARD_X + figure.GRPO_REWARD_W,
        )
        self.assertLess(
            figure.OPD_BAR_BASE + max(figure.OPD_BAR_VALUES),
            figure.OPD_SCORE_BOX_BOTTOM,
        )


if __name__ == "__main__":
    unittest.main()
