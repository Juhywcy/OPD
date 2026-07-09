# POV-OPD Comparison Figure Redesign

## Goal

Replace the current dense comparison graphic with a publication-scale figure that remains legible when placed at full two-column width in the AAAI template. The figure must communicate the progression from sparse outcome reward to dense probability reward and then to outcome- and prefix-validated dense reward.

## Layout

- Use one shared student trajectory followed by three equal method cards: GRPO, OPD, and POV-OPD.
- Remove the internal figure title, subtitle, long diagnostic sentences, and footer formula strip.
- Keep each method card to one reward diagram and one short takeaway.
- Give POV-OPD a restrained green border and an `OURS` tag; use color and structure together so the comparison remains understandable without relying on color alone.

## Content

- GRPO: show a single terminal `0/1` reward and label it `Sparse outcome reward`.
- OPD: show token-level probability differences and label them `Dense probability reward`.
- POV-OPD: show `Dense reward`, `Outcome gate`, and `Prefix-window gate` composing into a validated token reward; label it `Dense, aligned, prefix-safe`.
- Move all detailed explanation into the LaTeX caption.

## Typography and Spacing

- Design for the final AAAI page size, not the standalone PDF size.
- Use source-font sizes that render at no less than approximately 7.5 pt after LaTeX scaling.
- Avoid multi-line prose inside cards and leave explicit vertical padding around every label.
- Use measured text widths or generous fixed boxes; do not place text by approximate character-count centering when boxes are tight.

## Verification

- Regenerate the vector PDF and compile the full manuscript.
- Render the manuscript page containing the figure at high resolution.
- Inspect the final page for text overlap, clipping, font legibility, balanced card spacing, and caption separation.
- Scan the LaTeX log for overfull boxes, undefined references, and fatal errors.
