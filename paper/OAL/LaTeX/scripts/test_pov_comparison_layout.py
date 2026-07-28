import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "pov_comparison_source.tex"


class FigureLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SOURCE.read_text(encoding="utf-8")

    def test_smallest_explicit_font_remains_legible(self):
        sizes = [float(value) for value in re.findall(r"fontsize\{([0-9.]+)\}", self.content)]
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 6.3)

    def test_grpo_and_opd_baselines_are_explicit(self):
        required = [
            "Shared rollout",
            "A^{\\mathrm{GRPO}}",
            "use peer responses",
            "D_t=\\ell^T_t-\\ell^\\theta_t",
            "no outcome or prefix validity check",
        ]
        self.assertTrue(all(label in self.content for label in required))

    def test_complete_pov_computation_is_present(self):
        required = [
            "s_t=z_iD_t",
            "w^{\\mathrm{out}}",
            "excess surprisal\\\\$u_t$",
            "window average\\\\$\\bar u_b$",
            "CUSUM\\\\$C_b$",
            "after $b^\\star$",
            "w^{\\mathrm{pre}}",
            "p^{\\mathrm{sup}}",
            "w^{\\mathrm{sup}}",
            "A^{\\mathrm{POV}}_t=D_t",
            "never flips",
        ]
        self.assertTrue(all(label in self.content for label in required))

    def test_worked_token_example_covers_all_cases(self):
        required = [
            "aligned + drift",
            "aligned + later",
            "\\beta q_4D_4",
            "\\gamma_5D_5b_5",
            "\\gamma_6D_6b_6",
            "sampled-token example",
        ]
        self.assertTrue(all(label in self.content for label in required))


if __name__ == "__main__":
    unittest.main()
