# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

"""Standalone L2S stage-1 PPO entry point.

This module keeps the implementation isolated from the original code path. It
registers the recipe-local reward manager and advantage estimator before
delegating distributed training to the upstream `verl.trainer.main_ppo`.

Run:

    python3 -m recipe.repro.main_ppo ...
"""

import importlib

# Register recipe-local L2S reward and advantage implementations.
importlib.import_module("recipe.repro.l2s_stage1_reward")
importlib.import_module("recipe.repro.l2s_stage1_advantage")

from verl.trainer.main_ppo import main as _main


def main():
    """Run the upstream PPO CLI entry after loading the side-path reward module."""
    _main()


if __name__ == "__main__":
    main()
