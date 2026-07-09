# POV-OPD Comparison Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a compact three-card comparison figure whose text remains legible and non-overlapping at the final AAAI two-column page width.

**Architecture:** Keep the existing dependency-free vector-PDF generator, but simplify its visual hierarchy and expose the text sizes used during rendering. A small Python regression test will enforce the minimum effective font size and required/removed labels; full-manuscript compilation and page rendering will remain the visual acceptance test.

**Tech Stack:** Python 3 standard library, hand-authored PDF drawing commands, LaTeX/AAAI template, Ghostscript page rendering.

## Global Constraints

- Use one shared student trajectory followed by three equal method cards: GRPO, OPD, and POV-OPD.
- Remove the internal figure title, subtitle, long diagnostic sentences, and footer formula strip.
- Keep each method card to one reward diagram and one short takeaway.
- Give POV-OPD a restrained green border and an `OURS` tag.
- Render figure text at no less than approximately 7.5 pt after LaTeX scaling.
- Move detailed explanation into the LaTeX caption.

---

### Task 1: Add publication-scale typography regression checks

**Files:**
- Create: `paper/OAL/LaTeX/scripts/test_pov_comparison_layout.py`
- Modify: `paper/OAL/LaTeX/scripts/make_pov_comparison.py`

**Interfaces:**
- Consumes: `build_content() -> str` from the figure generator.
- Produces: `FONT_SIZES_USED: list[float]` and `effective_font_size(size: float) -> float` for layout validation.

- [ ] **Step 1: Write the failing regression test**

```python
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("make_pov_comparison.py")
spec = importlib.util.spec_from_file_location("pov_figure", MODULE_PATH)
figure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(figure)


def test_final_text_is_at_least_7_5pt():
    figure.FONT_SIZES_USED.clear()
    figure.build_content()
    assert figure.FONT_SIZES_USED
    assert min(map(figure.effective_font_size, figure.FONT_SIZES_USED)) >= 7.5


def test_figure_uses_only_compact_copy():
    content = figure.build_content()
    required = ["Sparse outcome reward", "Dense probability reward", "Dense, aligned, prefix-safe"]
    removed = ["Reward Estimators for On-Policy Reasoning Distillation", "POV advantage:", "prefix drift may be rewarded"]
    assert all(label in content for label in required)
    assert all(label not in content for label in removed)
```

- [ ] **Step 2: Run the test and verify that the old layout fails**

Run: `python3 -m unittest scripts/test_pov_comparison_layout.py -v`

Expected: FAIL because the current generator does not expose typography tracking and contains source font sizes that render below 7.5 pt.

- [ ] **Step 3: Add typography instrumentation without changing the old layout**

```python
FINAL_FIGURE_WIDTH_PT = 504.0
MIN_EFFECTIVE_FONT_PT = 7.5
FONT_SIZES_USED: list[float] = []


def effective_font_size(size: float) -> float:
    return size * FINAL_FIGURE_WIDTH_PT / PAGE_W
```

Append every size passed to `t_left()` and `t_center()` to `FONT_SIZES_USED`. This makes the existing small labels observable to the regression test.

- [ ] **Step 4: Re-run the test and verify that it fails for the intended size and copy assertions**

Run: `python3 -m unittest scripts/test_pov_comparison_layout.py -v`

Expected: two assertion failures: effective text below 7.5 pt and old verbose copy still present.

### Task 2: Rebuild the figure as a compact three-card comparison

**Files:**
- Modify: `paper/OAL/LaTeX/scripts/make_pov_comparison.py`
- Modify: `paper/OAL/LaTeX/sections/figure_comparison.tex`
- Regenerate: `paper/OAL/LaTeX/figures/pov_comparison.pdf`

**Interfaces:**
- Consumes: existing drawing helpers `rect`, `line`, `arrow`, `t_left`, and `t_center`.
- Produces: a 760 pt-wide vector PDF with a compact vertical footprint and source font sizes of at least 11.5 pt.

- [ ] **Step 1: Replace the content layout**

Set the canvas height to approximately 250 pt. Draw one shared trajectory row and three cards. Use source font sizes between 11.5 and 14 pt, with these exact takeaways:

```text
Sparse outcome reward
Dense probability reward
Dense, aligned, prefix-safe
```

The POV-OPD card composes three boxes labeled `Dense reward`, `Outcome gate`, and `Prefix gate` into `Validated token reward`. Remove all long prose from the graphic.

- [ ] **Step 2: Shorten the caption while preserving the estimator comparison**

Use a concise caption that explains GRPO's terminal 0/1 signal, OPD's dense probability signal, and POV-OPD's outcome and prefix gates without repeating every graphical label.

- [ ] **Step 3: Run the regression tests**

Run: `python3 -m unittest scripts/test_pov_comparison_layout.py -v`

Expected: `Ran 2 tests ... OK`.

- [ ] **Step 4: Regenerate the vector figure**

Run: `python3 scripts/make_pov_comparison.py`

Expected: prints the absolute path of `figures/pov_comparison.pdf` and exits with status 0.

### Task 3: Compile and visually verify the manuscript

**Files:**
- Regenerate: `paper/OAL/LaTeX/anonymous-submission-latex-2026.pdf`

**Interfaces:**
- Consumes: the regenerated vector figure and `sections/figure_comparison.tex`.
- Produces: the compiled AAAI manuscript PDF and a temporary PNG of the figure page.

- [ ] **Step 1: Compile with the existing TeX Live workflow**

Run from the LaTeX plugin root:

```bash
python3 scripts/compile_latex.py /Users/bytedance/Library/CloudStorage/OneDrive-个人/drpaper/OPD/OPD/paper/OAL/LaTeX/anonymous-submission-latex-2026.tex --compiler texlive
```

Expected: exit status 0 and an updated manuscript PDF.

- [ ] **Step 2: Scan the log for layout and reference failures**

Run:

```bash
rg -n "LaTeX Warning|Citation .*undefined|Reference .*undefined|Overfull|Undefined control sequence|Emergency stop|Fatal error" anonymous-submission-latex-2026.log
```

Expected: no matches related to the figure or manuscript build.

- [ ] **Step 3: Render and inspect the final manuscript page**

Render the page containing Figure 1 with Ghostscript at 220 dpi, then inspect the PNG at original resolution. Acceptance requires no overlapping or clipped text, readable labels at page scale, balanced whitespace, and clear visual emphasis on POV-OPD.

- [ ] **Step 4: Check the final diff**

Run: `git diff --check` and inspect `git status --short` to ensure only the intended figure, source, caption, tests, and generated manuscript artifacts changed.
