#!/usr/bin/env python3
"""Generate the POV-OPD method-comparison figure as a small vector PDF.

The script intentionally avoids matplotlib/TikZ dependencies so the AAAI build
remains portable.  The design follows a common conference-paper pattern:
three compact estimator cards fed by the same on-policy trajectory.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "pov_comparison.pdf"

PAGE_W = 760
PAGE_H = 250
FINAL_FIGURE_WIDTH_PT = 504.0
MIN_EFFECTIVE_FONT_PT = 7.5
FONT_SIZES_USED: list[float] = []

GRPO_REWARD_X = 62
GRPO_REWARD_W = 100
GRPO_ARROW_END_X = 154
OPD_BAR_BASE = 42
OPD_BAR_VALUES = [8, 16, 11, -7, 18, 10, 15]
OPD_SCORE_BOX_BOTTOM = 63

Color = tuple[float, float, float]

INK: Color = (0.09, 0.11, 0.16)
MUTED: Color = (0.42, 0.46, 0.54)
GRID: Color = (0.82, 0.85, 0.90)
PAPER: Color = (1.00, 1.00, 1.00)
SOFT_GRAY: Color = (0.96, 0.97, 0.985)
CARD: Color = (0.995, 0.997, 1.00)

ORANGE: Color = (0.92, 0.39, 0.08)
ORANGE_SOFT: Color = (1.00, 0.94, 0.86)
BLUE: Color = (0.12, 0.36, 0.86)
BLUE_SOFT: Color = (0.90, 0.95, 1.00)
RED: Color = (0.86, 0.18, 0.18)
GREEN: Color = (0.03, 0.56, 0.36)
GREEN_DARK: Color = (0.02, 0.38, 0.27)
GREEN_SOFT: Color = (0.89, 0.98, 0.93)
PURPLE: Color = (0.42, 0.27, 0.82)
PURPLE_SOFT: Color = (0.94, 0.91, 1.00)
WHITE: Color = (1.00, 1.00, 1.00)


def effective_font_size(size: float) -> float:
    return size * FINAL_FIGURE_WIDTH_PT / PAGE_W


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(color: Color, op: str) -> str:
    r, g, b = color
    return f"{r:.3f} {g:.3f} {b:.3f} {op}"


def mix(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))  # type: ignore[return-value]


def t_left(
    x: float,
    y: float,
    text: str,
    size: float = 9,
    bold: bool = False,
    leading: float = 10,
    color: Color = INK,
) -> list[str]:
    FONT_SIZES_USED.append(size)
    font = "F2" if bold else "F1"
    out = []
    for i, line in enumerate(text.split("\n")):
        out.append(
            f"BT /{font} {size:.1f} Tf {rgb(color, 'rg')} "
            f"{x:.2f} {y - i * leading:.2f} Td ({esc(line)}) Tj ET"
        )
    return out


def t_center(
    x: float,
    y: float,
    text: str,
    size: float = 9,
    bold: bool = False,
    leading: float = 10,
    color: Color = INK,
) -> list[str]:
    FONT_SIZES_USED.append(size)
    font = "F2" if bold else "F1"
    lines = text.split("\n")
    out = []
    start_y = y + (len(lines) - 1) * leading / 2
    for i, line in enumerate(lines):
        half_width = len(line) * size * 0.255
        out.append(
            f"BT /{font} {size:.1f} Tf {rgb(color, 'rg')} "
            f"{x - half_width:.2f} {start_y - i * leading:.2f} Td ({esc(line)}) Tj ET"
        )
    return out


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: Color,
    stroke: Color | None = GRID,
    lw: float = 0.8,
) -> list[str]:
    if stroke is None:
        return [f"{rgb(fill, 'rg')} {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f"]
    return [
        f"{rgb(fill, 'rg')} {rgb(stroke, 'RG')} {lw:.2f} w "
        f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re B"
    ]


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Color = MUTED,
    lw: float = 0.8,
    dash: bool = False,
) -> list[str]:
    dash_cmd = "[4 3] 0 d" if dash else "[] 0 d"
    return [
        f"{dash_cmd} {rgb(color, 'RG')} {lw:.2f} w "
        f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S [] 0 d"
    ]


def arrow(x1: float, y1: float, x2: float, y2: float, color: Color = MUTED, lw: float = 0.9) -> list[str]:
    cmds = line(x1, y1, x2, y2, color=color, lw=lw)
    if abs(x2 - x1) >= abs(y2 - y1):
        sgn = 1 if x2 >= x1 else -1
        cmds.append(
            f"{rgb(color, 'rg')} {x2:.2f} {y2:.2f} m "
            f"{x2 - 7 * sgn:.2f} {y2 + 3.4:.2f} l "
            f"{x2 - 7 * sgn:.2f} {y2 - 3.4:.2f} l h f"
        )
    else:
        sgn = 1 if y2 >= y1 else -1
        cmds.append(
            f"{rgb(color, 'rg')} {x2:.2f} {y2:.2f} m "
            f"{x2 - 3.4:.2f} {y2 - 7 * sgn:.2f} l "
            f"{x2 + 3.4:.2f} {y2 - 7 * sgn:.2f} l h f"
        )
    return cmds


def chip(x: float, y: float, w: float, text: str, fill: Color, stroke: Color, color: Color = INK) -> list[str]:
    cmds = rect(x, y, w, 34, fill=fill, stroke=stroke, lw=0.75)
    cmds.extend(t_center(x + w / 2, y + 12.0, text, size=11.5, bold=True, leading=11.5, color=color))
    return cmds


def tiny_tokens(x: float, y: float, n: int = 8, step: float = 19.5, final_color: Color | None = None) -> list[str]:
    cmds: list[str] = []
    for i in range(n):
        fill = WHITE
        stroke = GRID
        if final_color is not None and i == n - 1:
            fill = mix(final_color, WHITE, 0.74)
            stroke = final_color
        cmds.extend(rect(x + i * step, y, 13, 14, fill=fill, stroke=stroke, lw=0.65))
    return cmds


def bar_signal(x: float, y: float, values: list[float], color: Color, fade_after: int | None = None) -> list[str]:
    cmds: list[str] = []
    step = 15
    base = y
    cmds.extend(line(x - 5, base, x + step * (len(values) - 1) + 13, base, color=(0.58, 0.62, 0.69), lw=0.7))
    for i, val in enumerate(values):
        c = color
        if val < 0:
            c = RED
        if fade_after is not None and i >= fade_after:
            c = mix(c, WHITE, min(0.78, 0.25 + 0.14 * (i - fade_after)))
        h = abs(val)
        by = base if val >= 0 else base - h
        cmds.extend(rect(x + i * step, by, 8.5, h, fill=c, stroke=None))
    return cmds


def card(x: float, y: float, w: float, h: float, title: str, accent: Color, soft: Color, ours: bool = False) -> list[str]:
    border = accent if ours else GRID
    lw = 1.35 if ours else 0.85
    cmds = rect(x, y, w, h, fill=CARD, stroke=border, lw=lw)
    cmds.extend(rect(x, y + h - 34, w, 34, fill=soft, stroke=None))
    cmds.extend(t_left(x + 13, y + h - 23, title, size=14.0, bold=True, color=accent if ours else INK))
    if ours:
        cmds.extend(rect(x + w - 66, y + h - 28, 52, 21, fill=accent, stroke=accent, lw=0.5))
        cmds.extend(t_center(x + w - 40, y + h - 21.0, "OURS", size=11.5, bold=True, color=WHITE))
    return cmds


def build_content() -> str:
    FONT_SIZES_USED.clear()
    cmds: list[str] = [f"{rgb(PAPER, 'rg')} 0 0 {PAGE_W} {PAGE_H} re f"]

    # Common input trajectory.
    cmds.extend(rect(32, 211, 696, 27, fill=SOFT_GRAY, stroke=GRID, lw=0.8))
    cmds.extend(t_left(47, 220, "Student trajectory", size=12.5, bold=True, color=INK))
    cmds.extend(tiny_tokens(248, 217, n=9, step=25.0))
    cmds.extend(rect(623, 215, 88, 19, fill=ORANGE_SOFT, stroke=ORANGE, lw=0.8))
    cmds.extend(t_center(667, 221.0, "Final answer", size=11.5, color=ORANGE, bold=True))

    # Three estimator cards.
    y0, h = 18, 170
    w = 224
    xs = [24, 268, 512]

    # GRPO.
    cmds.extend(card(xs[0], y0, w, h, "GRPO", ORANGE, ORANGE_SOFT))
    cmds.extend(tiny_tokens(xs[0] + 28, y0 + 112, n=8, step=22.0, final_color=ORANGE))
    cmds.extend(arrow(xs[0] + 188, y0 + 111, xs[0] + GRPO_ARROW_END_X, y0 + 99, color=ORANGE, lw=1.1))
    cmds.extend(rect(xs[0] + GRPO_REWARD_X, y0 + 66, GRPO_REWARD_W, 32, fill=ORANGE, stroke=ORANGE, lw=0.8))
    cmds.extend(t_center(xs[0] + 112, y0 + 77.0, "0 / 1", size=14.0, bold=True, color=WHITE))
    cmds.extend(t_center(xs[0] + 112, y0 + 28, "Sparse outcome reward", size=12.2, bold=True, color=ORANGE))

    # OPD.
    cmds.extend(card(xs[1], y0, w, h, "OPD", BLUE, BLUE_SOFT))
    cmds.extend(rect(xs[1] + 18, y0 + 111, 82, 28, fill=WHITE, stroke=BLUE, lw=0.8))
    cmds.extend(t_center(xs[1] + 59, y0 + 121.0, "Teacher", size=12.0, bold=True, color=BLUE))
    cmds.extend(rect(xs[1] + 124, y0 + 111, 82, 28, fill=WHITE, stroke=BLUE, lw=0.8))
    cmds.extend(t_center(xs[1] + 165, y0 + 121.0, "Student", size=12.0, bold=True, color=BLUE))
    cmds.extend(arrow(xs[1] + 59, y0 + 111, xs[1] + 93, y0 + 91, color=BLUE, lw=1.0))
    cmds.extend(arrow(xs[1] + 165, y0 + 111, xs[1] + 131, y0 + 91, color=BLUE, lw=1.0))
    cmds.extend(rect(xs[1] + 67, y0 + 63, 90, 31, fill=BLUE, stroke=BLUE, lw=0.8))
    cmds.extend(t_center(xs[1] + 112, y0 + 74.0, "Token score", size=13.0, bold=True, color=WHITE))
    cmds.extend(bar_signal(xs[1] + 48, y0 + OPD_BAR_BASE, OPD_BAR_VALUES, BLUE))
    cmds.extend(t_center(xs[1] + 112, y0 + 28, "Dense probability reward", size=12.2, bold=True, color=BLUE))

    # POV-OPD.
    cmds.extend(card(xs[2], y0, w, h, "POV-OPD", GREEN, GREEN_SOFT, ours=True))
    cmds.extend(chip(xs[2] + 10, y0 + 105, 62, "Dense\nreward", BLUE_SOFT, BLUE, color=BLUE))
    cmds.extend(chip(xs[2] + 80, y0 + 105, 64, "Outcome\ngate", ORANGE_SOFT, ORANGE, color=ORANGE))
    cmds.extend(chip(xs[2] + 152, y0 + 105, 62, "Prefix\ngate", PURPLE_SOFT, PURPLE, color=PURPLE))
    cmds.extend(arrow(xs[2] + 41, y0 + 105, xs[2] + 61, y0 + 88, color=BLUE, lw=1.0))
    cmds.extend(arrow(xs[2] + 112, y0 + 105, xs[2] + 112, y0 + 88, color=ORANGE, lw=1.0))
    cmds.extend(arrow(xs[2] + 183, y0 + 105, xs[2] + 163, y0 + 88, color=PURPLE, lw=1.0))
    cmds.extend(rect(xs[2] + 25, y0 + 58, 174, 32, fill=GREEN, stroke=GREEN, lw=0.8))
    cmds.extend(t_center(xs[2] + 112, y0 + 69.0, "Validated token reward", size=12.5, bold=True, color=WHITE))
    cmds.extend(t_center(xs[2] + 112, y0 + 28, "Dense, aligned, prefix-safe", size=12.0, bold=True, color=GREEN_DARK))

    # Cross-card arrows from common trajectory.
    for cx in [xs[0] + w / 2, xs[1] + w / 2, xs[2] + w / 2]:
        cmds.extend(arrow(cx, 211, cx, 193, color=(0.62, 0.66, 0.73), lw=0.9))

    return "\n".join(cmds) + "\n"


def write_pdf(path: Path, content: str) -> None:
    objects = []
    stream = content.encode("latin-1")
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
        f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode("latin-1")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"endstream")

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{idx} 0 obj\n".encode("latin-1"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    data.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        data.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    write_pdf(OUT, build_content())
    print(OUT)


if __name__ == "__main__":
    main()
