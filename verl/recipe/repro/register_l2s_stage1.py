# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Bootstrap L2S stage-1 extensions inside the Ray TaskRunner process."""

import importlib

from verl.utils.reward_score import default_compute_score

from recipe.repro.l2s_stage1 import normalize_math_data_source


importlib.import_module("recipe.repro.l2s_stage1_reward")
importlib.import_module("recipe.repro.l2s_stage1_advantage")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score local math datasets with VERL's built-in math verifiers."""
    return default_compute_score(
        data_source=normalize_math_data_source(data_source),
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
