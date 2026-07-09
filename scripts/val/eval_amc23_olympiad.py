#!/usr/bin/env python3
"""Compatibility wrapper for the verl-based AMC23/Olympiad validation job."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).with_suffix(".sh")
    env = os.environ.copy()

    if len(sys.argv) > 1:
        env.setdefault("ACTOR_MODEL_PATH", sys.argv[1])

    return subprocess.call(["bash", str(script_path)], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
