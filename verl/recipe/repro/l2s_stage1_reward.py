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

"""L2S stage-1 length-decayed terminal reward manager."""

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.repro.l2s_stage1 import compute_length_decayed_reward, place_terminal_reward


def _extract_correctness(result: Any) -> bool:
    if isinstance(result, dict):
        if "is_correct" in result:
            return bool(result["is_correct"])
        if "acc" in result:
            return bool(result["acc"])
        if "score" in result:
            return float(result["score"]) > 0.0
        raise ValueError("score dictionary must contain is_correct, acc, or score")
    if isinstance(result, bool):
        return result
    return float(result) > 0.0


@register("l2s_stage1")
class L2SStage1RewardManager(AbstractRewardManager):
    """Place a correctness-gated length reward on the final response token."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        beta=2.0,
        decay_target=0.01,
        decay_horizon=8192,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.beta = float(beta)
        self.decay_target = float(decay_target)
        self.decay_horizon = int(decay_horizon)
        self.kwargs = dict(kwargs)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[Any]] = defaultdict(list)
        printed_per_source: dict[str, int] = defaultdict(int)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]

            valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # DeepScaler judges the complete, unstripped sequence: its verifier
            # needs the ``<think>`` token from the prompt and ``</think>`` from
            # the response.  This precisely matches the original L2S code.
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth")
            if ground_truth is None:
                ground_truth = data_item.non_tensor_batch.get("ground_truth")
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            result = self.compute_score(
                data_source=data_source,
                solution_str=sequence_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            is_correct = _extract_correctness(result)
            reward = compute_length_decayed_reward(
                is_correct=is_correct,
                response_length=valid_response_length,
                beta=self.beta,
                decay_target=self.decay_target,
                decay_horizon=self.decay_horizon,
            )
            place_terminal_reward(reward_tensor[i], valid_response_length, reward)

            if isinstance(result, dict):
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            reward_extra_info["is_correct"].append(is_correct)
            reward_extra_info["l2s_reward"].append(reward)
            reward_extra_info["valid_response_length"].append(valid_response_length)

            if printed_per_source[data_source] < self.num_examine:
                printed_per_source[data_source] += 1
                print("[l2s-stage1] [prompt]", prompt_str)
                print("[l2s-stage1] [response]", response_str)
                print("[l2s-stage1] [ground_truth]", ground_truth)
                print("[l2s-stage1] [is_correct]", is_correct)
                print("[l2s-stage1] [response_length]", valid_response_length)
                print("[l2s-stage1] [reward]", reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
