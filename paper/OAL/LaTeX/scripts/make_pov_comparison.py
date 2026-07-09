#!/usr/bin/env python3
"""Generate a dependency-free vector PDF comparison figure."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "pov_comparison.pdf"

PAGE_W = 720
PAGE_H = 320


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def color(gray: float) -> str:
    return f"{gray:.3f} g {gray:.3f} G"


def text_center(x: float, y: float, text: str, size: float = 9, bold: bool = False, leading: float = 11) -> list[str]:
    font = "F2" if bold else "F1"
    lines = text.split("\n")
    out = []
    start_y = y + (len(lines) - 1) * leading / 2
    for i, line in enumerate(lines):
        # Approximate text width for centered placement in Helvetica.
        width = len(line) * size * 0.255
        out.append(f"BT /{font} {size:.1f} Tf 0 g {x - width:.2f} {start_y - i * leading:.2f} Td ({esc(line)}) Tj ET")
    return out


def box(x: float, y: float, w: float, h: float, label: str, fill: float = 0.96, stroke: float = 0.15) -> list[str]:
    r = 5
    cmds = [
        color(stroke),
        f"{fill:.3f} g",
        f"{x+r:.2f} {y:.2f} m",
        f"{x+w-r:.2f} {y:.2f} l",
        f"{x+w:.2f} {y:.2f} {x+w:.2f} {y+r:.2f} {x+w:.2f} {y+r:.2f} c",
        f"{x+w:.2f} {y+h-r:.2f} l",
        f"{x+w:.2f} {y+h:.2f} {x+w-r:.2f} {y+h:.2f} {x+w-r:.2f} {y+h:.2f} c",
        f"{x+r:.2f} {y+h:.2f} l",
        f"{x:.2f} {y+h:.2f} {x:.2f} {y+h-r:.2f} {x:.2f} {y+h-r:.2f} c",
        f"{x:.2f} {y+r:.2f} l",
        f"{x:.2f} {y:.2f} {x+r:.2f} {y:.2f} {x+r:.2f} {y:.2f} c",
        "B",
    ]
    cmds.extend(text_center(x + w / 2, y + h / 2 - 2, label, size=9.2, leading=11))
    return cmds


def arrow(x1: float, y1: float, x2: float, y2: float) -> list[str]:
    return [
        "0.2 G 0.2 g 1 w",
        f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S",
        f"{x2:.2f} {y2:.2f} m {x2-4:.2f} {y2+7:.2f} l {x2+4:.2f} {y2+7:.2f} l h f",
    ]


def add_column(x: float, title: str, labels: list[str], note: str, final_fill: float) -> list[str]:
    cmds: list[str] = []
    col_w = 185
    cx = x + col_w / 2
    cmds.extend(text_center(cx, 294, title, size=13.5, bold=True, leading=14))
    ys = [240, 184, 128, 72]
    for i, (y, label) in enumerate(zip(ys, labels)):
        fill = final_fill if i == len(labels) - 1 else 0.965
        cmds.extend(box(x, y, col_w, 38, label, fill=fill, stroke=0.10))
        if i > 0:
            cmds.extend(arrow(cx, ys[i - 1], cx, y + 38))
    cmds.extend(text_center(cx, 29, note, size=9.2, leading=11))
    return cmds


def build_content() -> str:
    cmds: list[str] = ["1 1 1 rg 0 0 720 320 re f"]
    cmds.extend(
        add_column(
            26,
            "GRPO",
            [
                "Student samples\nmultiple responses",
                "Verifier gives\nfinal correctness",
                "Group-relative\nscalar advantage",
                "Broadcast to\nall tokens",
            ],
            "Reliable outcome signal,\nbut coarse credit assignment",
            0.90,
        )
    )
    cmds.extend(
        add_column(
            268,
            "OPD",
            [
                "Student samples\none trajectory",
                "Teacher scores\nvisited prefixes",
                "Dense logit gap\nl_T - l_S",
                "Use as token\nadvantage",
            ],
            "Fine-grained supervision,\nbut unvalidated teacher deltas",
            0.86,
        )
    )
    cmds.extend(
        add_column(
            510,
            "POV-OPD",
            [
                "Student samples\none trajectory",
                "Teacher dense gap\n+ final outcome",
                "Outcome split\n+ prefix CUSUM",
                "Validated dense\nadvantage",
            ],
            "Dense OPD signal validated by\noutcome and prefix support",
            0.82,
        )
    )
    # Prefix decay cue.
    x0, y0 = 523, 55
    for i, gray in enumerate([0.10, 0.16, 0.25, 0.38, 0.52, 0.66, 0.78, 0.87]):
        cmds.append(f"{gray:.3f} g {x0 + i*18:.2f} {y0:.2f} 14 8 re f")
    cmds.extend(text_center(603, 45, "prefix weights decay after drift", size=7.8))
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
    data.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    data.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        data.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    data.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    write_pdf(OUT, build_content())
    print(OUT)


if __name__ == "__main__":
    main()
