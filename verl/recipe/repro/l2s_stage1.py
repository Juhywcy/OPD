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

"""Framework-independent primitives for the L2S stage-1 objective."""

import math
from collections.abc import Sequence
from typing import Any


def normalize_math_data_source(data_source: str) -> str:
    """Map local math dataset labels to VERL's built-in verifier names."""
    normalized = str(data_source).lower()
    if normalized in {"deepscaler", "dapo"}:
        return "math_dapo"
    if normalized.startswith("aime"):
        return normalized
    return str(data_source)


def compute_length_decayed_reward(
    is_correct: bool,
    response_length: int,
    beta: float = 2.0,
    decay_target: float = 0.01,
    decay_horizon: int = 8192,
) -> float:
    """Return the original L2S stage-1 sequence reward."""
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")
    if decay_horizon <= 0:
        raise ValueError(f"decay_horizon must be positive, got {decay_horizon}")
    if not 0.0 < decay_target < 1.0:
        raise ValueError(f"decay_target must be in (0, 1), got {decay_target}")
    if not is_correct:
        return 0.0

    decay_rate = -math.log(decay_target) / decay_horizon
    return float(beta * math.exp(-decay_rate * response_length))


def place_terminal_reward(row: Any, valid_response_length: int, reward: float) -> Any:
    """Place a scalar reward on the last valid response token in-place."""
    if valid_response_length < 0:
        raise ValueError(f"valid_response_length must be non-negative, got {valid_response_length}")
    if valid_response_length > len(row):
        raise ValueError(
            f"valid_response_length ({valid_response_length}) exceeds reward row width ({len(row)})"
        )
    if valid_response_length > 0:
        row[valid_response_length - 1] = reward
    return row


def broadcast_sequence_rewards(token_level_rewards: Any, response_mask: Any) -> Any:
    """Broadcast each summed sequence reward over its valid response tokens.

    PyTorch tensors use the fast tensor path. Nested Python sequences are
    supported so the exact objective can be tested without a GPU environment.
    """
    if hasattr(token_level_rewards, "sum") and hasattr(response_mask, "shape"):
        sequence_rewards = token_level_rewards.sum(dim=-1)
        return sequence_rewards.unsqueeze(-1) * response_mask

    if not isinstance(token_level_rewards, Sequence) or not isinstance(response_mask, Sequence):
        raise TypeError("token_level_rewards and response_mask must be tensors or nested sequences")
    if len(token_level_rewards) != len(response_mask):
        raise ValueError("token_level_rewards and response_mask must have the same batch size")

    broadcast_rewards = []
    for reward_row, mask_row in zip(token_level_rewards, response_mask):
        if len(reward_row) != len(mask_row):
            raise ValueError("reward and mask rows must have the same width")
        sequence_reward = sum(reward_row)
        broadcast_rewards.append([sequence_reward * mask for mask in mask_row])
    return broadcast_rewards
