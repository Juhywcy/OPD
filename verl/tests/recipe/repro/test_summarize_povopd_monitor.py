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

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "recipe" / "repro" / "summarize_povopd_monitor.py"
SPEC = importlib.util.spec_from_file_location("summarize_povopd_monitor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MONITOR_PREFIX = MODULE.MONITOR_PREFIX
summarize_log = MODULE.summarize_log


def test_summarize_log_collects_heartbeats_validation_and_errors(tmp_path):
    log_path = tmp_path / "run.log"
    records = [
        {
            "training/global_step": 1,
            "oal/outcome_correction_weight_mean": 0.1,
            "pt_oal/prefix_weight_mean": 0.9,
        },
        {
            "training/global_step": 20,
            "oal/outcome_correction_weight_mean": 0.2,
            "pt_oal/prefix_weight_mean": 0.8,
            "val-core/AIME24/reward/mean@16": 0.35,
            "val-core/AIME24/reward/best@16": 0.5,
        },
    ]
    log_path.write_text(
        "startup\n"
        + "\n".join(MONITOR_PREFIX + json.dumps(record) for record in records)
        + "\nRuntimeError: example failure\n",
        encoding="utf-8",
    )

    summary = summarize_log(log_path)

    assert summary["heartbeat_records"] == 2
    assert summary["latest_step"] == 20
    assert summary["latest"] == records[-1]
    assert summary["validation"] == [records[-1]]
    correction_stats = summary["metric_statistics"]["oal/outcome_correction_weight_mean"]
    assert correction_stats["mean"] == pytest.approx(0.15)
    assert correction_stats["min"] == 0.1
    assert correction_stats["max"] == 0.2
    assert correction_stats["last"] == 0.2
    assert summary["errors"] == ["4: RuntimeError: example failure"]


def test_summarize_log_reports_malformed_monitor_json(tmp_path):
    log_path = tmp_path / "bad.log"
    log_path.write_text(MONITOR_PREFIX + "{bad json}\n", encoding="utf-8")

    summary = summarize_log(log_path)

    assert summary["heartbeat_records"] == 0
    assert summary["latest_step"] is None
    assert summary["errors"] == ["1: malformed POV monitor JSON"]
