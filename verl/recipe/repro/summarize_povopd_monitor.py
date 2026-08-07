#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize compact POVOPD heartbeat records from training logs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

MONITOR_PREFIX = "[pov-monitor] "
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "OutOfMemoryError",
    "CUDA out of memory",
    "CUDA error",
    "Error executing job",
    "Failed to connect",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _read_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if any(marker in stripped for marker in ERROR_MARKERS):
                errors.append(f"{line_number}: {stripped.strip()}")
            marker_position = stripped.find(MONITOR_PREFIX)
            if marker_position < 0:
                continue
            payload = stripped[marker_position + len(MONITOR_PREFIX) :]
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                errors.append(f"{line_number}: malformed POV monitor JSON")
                continue
            if isinstance(record, dict):
                records.append(record)
    return records, errors


def _metric_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_values: dict[str, list[float]] = {}
    for record in records:
        for key, raw_value in record.items():
            if key.startswith("val-core/") or key in {"training/global_step", "training/epoch"}:
                continue
            value = _finite_number(raw_value)
            if value is not None:
                metric_values.setdefault(key, []).append(value)
    return {
        key: {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
        for key, values in sorted(metric_values.items())
    }


def summarize_log(path: Path) -> dict[str, Any]:
    records, errors = _read_log(path)
    validation_records = [record for record in records if any(key.startswith("val-core/") for key in record)]
    return {
        "log": str(path),
        "heartbeat_records": len(records),
        "latest_step": records[-1].get("training/global_step") if records else None,
        "latest": records[-1] if records else {},
        "validation": validation_records,
        "metric_statistics": _metric_statistics(records),
        "errors": errors[-30:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Training log files to summarize")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    args = parser.parse_args()

    missing = [str(path) for path in args.logs if not path.is_file()]
    if missing:
        parser.error("log file not found: " + ", ".join(missing))

    summaries = [summarize_log(path) for path in args.logs]
    print(json.dumps(summaries, indent=args.indent, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
