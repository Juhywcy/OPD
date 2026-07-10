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

"""Recipe-local advantage estimator for L2S stage-1."""

from verl.trainer.ppo.core_algos import register_adv_est

from recipe.repro.l2s_stage1 import broadcast_sequence_rewards


@register_adv_est("l2s_stage1_reward_as_advantage")
def compute_l2s_stage1_advantage(token_level_rewards, response_mask, **kwargs):
    """Use the sequence reward directly as every valid token's advantage."""
    del kwargs
    advantages = broadcast_sequence_rewards(token_level_rewards, response_mask)
    return advantages, advantages.clone()
