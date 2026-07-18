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
            "D_{t,k}=\\rho_{t,k}\\delta_{t,k}",
            "no outcome or prefix validity check",
        ]
        self.assertTrue(all(label in self.content for label in required))

    def test_complete_pov_computation_is_present(self):
        required = [
            "s_{t,k}=z_i\\delta_{t,k}",
            "w^{\\mathrm{out}}",
            "excess surprisal $u_t$",
            "window mean $\\bar u_b$",
            "CUSUM drift $C_b$",
            "$b\\ge b^\\star$: monotone decay",
            "w^{\\mathrm{pre}}",
            "p^{\\mathrm{sup}}",
            "w^{\\mathrm{sup}}",
            "A^{\\mathrm{POV}}_{t,k}=D_{t,k}",
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
