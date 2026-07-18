#!/usr/bin/env python3
"""Compile the editable TikZ source for the POV-OPD method figure."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "pov_comparison_source.tex"
DEFAULT_OUTPUT = ROOT / "figures" / "pov_comparison.pdf"


def find_pdflatex() -> str:
    candidates = (
        shutil.which("pdflatex"),
        "/Library/TeX/texbin/pdflatex",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("pdflatex is required to compile pov_comparison_source.tex")


def main() -> None:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    jobname = output.stem
    command = [
        find_pdflatex(),
        "-interaction=batchmode",
        "-halt-on-error",
        f"-jobname={jobname}",
        f"-output-directory={output.parent}",
        str(SOURCE),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    compiled = output.parent / f"{jobname}.pdf"
    if compiled != output:
        compiled.replace(output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"figure was not generated: {output}")
    print(output)


if __name__ == "__main__":
    main()
