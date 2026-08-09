# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est("token_grpo")
def compute_grpo_process_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Process Supervision GRPO.
    
    Based on the formula:
    1. Normalize rewards: r_tilde = (r - mean(R)) / std(R)
       where R is the set of all step rewards in the group.
    2. Compute Advantage: A_t = sum(r_tilde_j) for j >= t
       (Sum of normalized future rewards).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length). Contains the immediate reward for each token.
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). 1 for valid response tokens, 0 for padding.
        index: `(np.ndarray)`
            index array for grouping samples (mapping batch items to prompts).
        epsilon: `(float)`
            small value to avoid division by zero.
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the reward by std (standard GRPO) or just center it.
        config: `(Optional[AlgoConfig])`
            algorithm configuration object.

    Returns:
        advantages: `(torch.Tensor)` shape (bs, response_length)
        returns: `(torch.Tensor)` shape (bs, response_length) (in this context, same as advantages)
    """
    
    # 1. 初始化变量
    # 确保 padding 部分的 reward 为 0，避免统计错误
    masked_rewards = token_level_rewards * response_mask
    normalized_rewards = torch.zeros_like(masked_rewards)
    
    # 用于存储每个组对应的 batch indices
    # index map: group_id -> [batch_idx_1, batch_idx_2, ...]
    id2indices = defaultdict(list)
    bsz = masked_rewards.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    # 2. 组内归一化 (Step 1 in Image)
    # 遍历每个组，计算该组所有 valid token rewards 的 mean 和 std
    with torch.no_grad():
        for group_idx, batch_indices in id2indices.items():
            # 获取该组的所有 rewards 和 masks: shape (G, L)
            group_rewards = masked_rewards[batch_indices]
            group_mask = response_mask[batch_indices]

            # 展平该组内所有有效的 token rewards 以计算统计量
            # R = {r_1...r_N} across all steps and all outputs in the group
            valid_rewards_flat = group_rewards[group_mask.bool()]

            if valid_rewards_flat.numel() == 0:
                continue
                
            # 计算 Mean 和 Std
            if valid_rewards_flat.numel() > 1:
                mean = valid_rewards_flat.mean()
                std = valid_rewards_flat.std()
            else:
                mean = torch.tensor(0.0, device=token_level_rewards.device, dtype=token_level_rewards.dtype)
                std = torch.tensor(1.0, device=token_level_rewards.device, dtype=token_level_rewards.dtype)

            # 计算归一化的 reward: r_tilde
            # 注意：这里需要保持 (G, L) 的形状以便放回
            if norm_adv_by_std_in_grpo:
                # Formula: (r - mean) / std
                group_norm_rewards = (group_rewards - mean) / (std + epsilon)
            else:
                # Formula: r - mean (Dr.GRPO style)
                group_norm_rewards = group_rewards - mean
            
            # 再次应用 mask，确保 padding 位置归一化后仍为 0 (减去均值后 padding 处可能变成负均值)
            group_norm_rewards = group_norm_rewards * group_mask
            
            # 将计算好的值填回总的 tensor 中
            normalized_rewards[batch_indices] = group_norm_rewards

    # 3. 计算 Advantage (Step 2 in Image)
    # Formula: A_{i,t} = sum_{index(j) >= t} r_tilde_j
    # 即：当前 token 的 advantage 是当前及未来所有 normalized rewards 的和
    # 这相当于对 normalized_rewards 做从右向左的累加 (Reverse Cumulative Sum)
    
    # torch.cumsum 是从左向右，所以我们先 flip dim=1，cumsum 后再 flip 回来
    advantages = torch.flip(torch.cumsum(torch.flip(normalized_rewards, dims=[1]), dim=1), dims=[1])
    
    # 再次 mask 确保 padding 处的 advantage 为 0
    advantages = advantages * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv

@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    直接使用token-level reward作为advantage的最简单estimator
    Now supports 3D rewards (batch, seq_len, k) when using top-k sampling
    
    Args:
        token_level_rewards: (bs, response_length) or (bs, response_length, k) 每个token的奖励
        response_mask: (bs, response_length) 响应掩码
    
    Returns:
        advantages: (bs, response_length) or (bs, response_length, k) 与rewards相同
        returns: (bs, response_length) or (bs, response_length, k) 与rewards相同
    """
    with torch.no_grad():
        original_response_mask = response_mask
        oal_cfg = _oal_get_config(config)
        margin = oal_cfg["margin"]

        # If rewards are 3D (batch, seq_len, k), broadcast mask to (batch, seq_len, 1)
        if token_level_rewards.dim() == 3:
            reward_mask = response_mask.unsqueeze(-1)
        else:
            reward_mask = response_mask
        advantages = token_level_rewards * reward_mask
        returns = advantages.clone()

        align_scores = kwargs.get("logit_delta_scores", token_level_rewards).to(
            device=token_level_rewards.device, dtype=token_level_rewards.dtype
        )
        outcome_scores = _odw_outcome_scores(
            kwargs.get("true_reward_score", None),
            token_level_rewards,
            original_response_mask,
        )
        correct = outcome_scores > 0.5

        if token_level_rewards.dim() == 3:
            correct_view = correct.view(-1, 1, 1)
            valid_mask = original_response_mask.unsqueeze(-1).expand_as(align_scores).to(
                dtype=token_level_rewards.dtype
            )
            pos_align_mask = (correct_view & (align_scores > margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            pos_anti_mask = (correct_view & (align_scores < -margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            neg_align_mask = ((~correct_view) & (align_scores < -margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            neg_anti_mask = ((~correct_view) & (align_scores > margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            token_keep_mask = original_response_mask.to(dtype=token_level_rewards.dtype)
        else:
            correct_view = correct.view(-1, 1)
            valid_mask = original_response_mask.to(dtype=token_level_rewards.dtype)
            pos_align_mask = (correct_view & (align_scores > margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            pos_anti_mask = (correct_view & (align_scores < -margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            neg_align_mask = ((~correct_view) & (align_scores < -margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            neg_anti_mask = ((~correct_view) & (align_scores > margin)).to(dtype=token_level_rewards.dtype) * valid_mask
            token_keep_mask = valid_mask

        extra_metrics = {
            "oal_keep_mask": valid_mask.detach(),
            "oal_token_keep_mask": token_keep_mask.detach(),
            "oal_outcome_scores": outcome_scores.detach(),
            "oal_correct_mask": correct.to(dtype=original_response_mask.dtype).detach(),
            "oal_pos_align_mask": pos_align_mask.detach(),
            "oal_pos_anti_mask": pos_anti_mask.detach(),
            "oal_neg_align_mask": neg_align_mask.detach(),
            "oal_neg_anti_mask": neg_anti_mask.detach(),
            "oal_token_weights": valid_mask.detach(),
        }

    return advantages, returns, extra_metrics


def _config_get(config, key, default):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _ahopd_get_config(config):
    raw = _config_get(config, "ah_opd", {}) or {}
    return {
        "enabled": bool(_config_get(raw, "enabled", True)),
        "min_horizon": int(_config_get(raw, "min_horizon", 1024)),
        "window_size": max(1, int(_config_get(raw, "window_size", 256))),
        "threshold": float(_config_get(raw, "threshold", 0.45)),
        "patience_windows": max(1, int(_config_get(raw, "patience_windows", 2))),
        "soft_weighting": bool(_config_get(raw, "soft_weighting", True)),
        "overlap_weight": float(_config_get(raw, "overlap_weight", 0.35)),
        "entropy_weight": float(_config_get(raw, "entropy_weight", 0.25)),
        "reward_weight": float(_config_get(raw, "reward_weight", 0.40)),
        "entropy_threshold": float(_config_get(raw, "entropy_threshold", 3.0)),
        "entropy_temperature": max(1e-6, float(_config_get(raw, "entropy_temperature", 1.0))),
        "reward_temperature": max(1e-6, float(_config_get(raw, "reward_temperature", 1.0))),
        "entropy_gap_temperature": max(1e-6, float(_config_get(raw, "entropy_gap_temperature", 0.3))),
        "prefix_window_size": max(1, int(_config_get(raw, "prefix_window_size", 1024))),
        "eps": max(1e-8, float(_config_get(raw, "eps", 1e-6))),
        "outcome_mix": bool(_config_get(raw, "outcome_mix", True)),
        "transition_power": max(
            0.0,
            float(_config_get(raw, "transition_power", _config_get(raw, "outcome_decay_power", 1.0))),
        ),
        "difficulty_low": float(_config_get(raw, "difficulty_low", 0.25)),
        "difficulty_high": float(_config_get(raw, "difficulty_high", 0.75)),
        "difficulty_power": max(0.0, float(_config_get(raw, "difficulty_power", 1.0))),
        "outcome_weight": float(_config_get(raw, "outcome_weight", 1.0)),
    }


def _ahopd_token_scores(token_level_rewards):
    if token_level_rewards.dim() == 3:
        return token_level_rewards.sum(dim=-1)
    if token_level_rewards.dim() == 2:
        return token_level_rewards
    raise ValueError(f"Expected 2D or 3D token rewards, got {tuple(token_level_rewards.shape)}")


def _ahopd_smooth(values, response_mask, window_size):
    if window_size <= 1:
        return values * response_mask
    kernel = torch.ones(1, 1, window_size, device=values.device, dtype=values.dtype)
    pad = window_size // 2
    numer = torch.nn.functional.conv1d((values * response_mask).unsqueeze(1), kernel, padding=pad).squeeze(1)
    denom = torch.nn.functional.conv1d(response_mask.unsqueeze(1), kernel, padding=pad).squeeze(1).clamp_min(1.0)
    if numer.shape[-1] != values.shape[-1]:
        numer = numer[:, : values.shape[-1]]
        denom = denom[:, : values.shape[-1]]
    return numer / denom


def _ahopd_causal_prefix_mean(values, response_mask, window_size):
    """Mean over previous tokens in [t - window_size, t), excluding current token."""
    shifted_values = torch.nn.functional.pad(values[:, :-1], (1, 0))
    shifted_mask = torch.nn.functional.pad(response_mask[:, :-1], (1, 0))
    kernel = torch.ones(1, 1, window_size, device=values.device, dtype=values.dtype)
    numer = torch.nn.functional.conv1d((shifted_values * shifted_mask).unsqueeze(1), kernel, padding=window_size - 1)
    denom = torch.nn.functional.conv1d(shifted_mask.unsqueeze(1), kernel, padding=window_size - 1)
    numer = numer.squeeze(1)[:, : values.shape[-1]]
    denom = denom.squeeze(1)[:, : values.shape[-1]]
    prefix_mean = numer / denom.clamp_min(1.0)
    return torch.where(denom > 0, prefix_mean, torch.ones_like(prefix_mean))


def _ahopd_find_horizon(reliability, response_mask, ah_cfg):
    batch_size, seq_len = response_mask.shape
    valid_lens = response_mask.sum(dim=-1).long()
    horizons = valid_lens.clone()
    min_horizon = min(ah_cfg["min_horizon"], seq_len)
    window_size = ah_cfg["window_size"]
    patience_windows = ah_cfg["patience_windows"]

    for batch_idx in range(batch_size):
        valid_len = int(valid_lens[batch_idx].item())
        if valid_len <= min_horizon:
            continue

        bad_windows = 0
        for start in range(min_horizon, valid_len, window_size):
            end = min(start + window_size, valid_len)
            window_mask = response_mask[batch_idx, start:end] > 0
            if not window_mask.any():
                continue
            window_score = reliability[batch_idx, start:end][window_mask].mean()
            if window_score < ah_cfg["threshold"]:
                bad_windows += 1
                if bad_windows >= patience_windows:
                    horizons[batch_idx] = max(min_horizon, start - (patience_windows - 1) * window_size)
                    break
            else:
                bad_windows = 0

    return horizons


def _ahopd_transition_progress(response_mask, horizons, power):
    mask = response_mask.to(dtype=torch.float32)
    positions = torch.arange(mask.shape[-1], device=mask.device, dtype=mask.dtype).unsqueeze(0)
    valid_lens = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    horizon_pos = horizons.to(device=mask.device, dtype=mask.dtype).unsqueeze(-1)
    suffix_len = valid_lens - horizon_pos
    suffix_span = (suffix_len - 1.0).clamp_min(1.0)
    progress = ((positions - horizon_pos) / suffix_span).clamp(0.0, 1.0)
    single_token_suffix = (suffix_len <= 1.0) & (positions >= horizon_pos)
    progress = torch.where(single_token_suffix, torch.ones_like(progress), progress)
    return progress.pow(power) * mask


def _ahopd_reliability_weights(
    token_level_rewards,
    response_mask,
    config,
    overlap_mask=None,
    teacher_entropy=None,
    student_entropy=None,
):
    ah_cfg = _ahopd_get_config(config)
    mask = response_mask.to(dtype=token_level_rewards.dtype)

    if overlap_mask is not None:
        overlap_score = overlap_mask.to(dtype=token_level_rewards.dtype)
        if overlap_score.dim() == 3:
            overlap_score = overlap_score.mean(dim=-1)
        overlap_score = overlap_score.clamp(0.0, 1.0)
    else:
        overlap_score = torch.ones_like(mask)

    if student_entropy is not None and teacher_entropy is not None:
        student_entropy = student_entropy.to(dtype=token_level_rewards.dtype)
        teacher_entropy = teacher_entropy.to(dtype=token_level_rewards.dtype)
        gap_score = torch.sigmoid(
            (student_entropy - teacher_entropy) / ah_cfg["entropy_gap_temperature"]
        )
    elif teacher_entropy is not None:
        teacher_entropy = teacher_entropy.to(dtype=token_level_rewards.dtype)
        gap_score = torch.sigmoid((ah_cfg["entropy_threshold"] - teacher_entropy) / ah_cfg["entropy_temperature"])
    else:
        gap_score = torch.ones_like(mask)
    gap_score = gap_score.clamp(0.0, 1.0)

    prefix_score = _ahopd_causal_prefix_mean(overlap_score, mask, ah_cfg["prefix_window_size"]).clamp(0.0, 1.0)

    eps = ah_cfg["eps"]
    weight_sum = max(ah_cfg["overlap_weight"] + ah_cfg["entropy_weight"] + ah_cfg["reward_weight"], eps)
    log_reliability = (
        ah_cfg["overlap_weight"] * torch.log(overlap_score.clamp_min(eps))
        + ah_cfg["entropy_weight"] * torch.log(gap_score.clamp_min(eps))
        + ah_cfg["reward_weight"] * torch.log(prefix_score.clamp_min(eps))
    ) / weight_sum
    reliability = torch.exp(log_reliability)
    reliability = _ahopd_smooth(reliability, mask, ah_cfg["window_size"]) * mask

    horizons = _ahopd_find_horizon(reliability, mask, ah_cfg)
    positions = torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0)
    prefix_mask = (positions < horizons.unsqueeze(-1)).to(dtype=mask.dtype) * mask
    transition_progress = _ahopd_transition_progress(
        response_mask=response_mask,
        horizons=horizons,
        power=ah_cfg["transition_power"],
    ).to(dtype=mask.dtype)
    teacher_weight = 1.0 - transition_progress

    token_weights = torch.where(
        prefix_mask > 0,
        torch.ones_like(mask),
        teacher_weight,
    )

    token_weights = token_weights * mask
    components = {
        "ahopd_overlap_score": overlap_score.detach(),
        "ahopd_gap_score": gap_score.detach(),
        "ahopd_prefix_score": prefix_score.detach(),
    }
    return token_weights, reliability, horizons, components


@register_adv_est("adaptive_horizon_token_reward")
def compute_adaptive_horizon_token_reward_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    overlap_mask: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    student_entropy: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Adaptive Horizon OPD.

    It starts from token-level OPD rewards, estimates whether dense teacher
    supervision is reliable at each response position, and down-weights or
    truncates unreliable suffix tokens. Reliability is computed from available
    OPD diagnostics: teacher/student top-k overlap, student-teacher entropy gap,
    and causal prefix overlap.
    """
    with torch.no_grad():
        ah_cfg = _ahopd_get_config(config)
        if token_level_rewards.dim() == 3:
            reward_mask = response_mask.unsqueeze(-1).to(dtype=token_level_rewards.dtype)
        else:
            reward_mask = response_mask.to(dtype=token_level_rewards.dtype)

        if not ah_cfg["enabled"]:
            advantages = token_level_rewards * reward_mask
            returns = advantages.clone()
            return advantages, returns, {}

        token_weights, reliability, horizons, reliability_components = _ahopd_reliability_weights(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            config=config,
            overlap_mask=overlap_mask,
            teacher_entropy=teacher_entropy,
            student_entropy=student_entropy,
        )

        if token_level_rewards.dim() == 3:
            loss_weights = token_weights.unsqueeze(-1)
        else:
            loss_weights = token_weights

        dense_advantages = token_level_rewards * reward_mask
        advantages = dense_advantages * loss_weights
        outcome_advantages = None
        transition_alpha = (1.0 - token_weights) * response_mask.to(dtype=token_weights.dtype)
        outcome_weights_for_log = torch.zeros_like(token_weights)
        if ah_cfg["outcome_mix"]:
            rewards_for_outcome = kwargs.get("true_reward_score", token_level_rewards)
            if rewards_for_outcome.dim() == 3:
                rewards_for_outcome = rewards_for_outcome.sum(dim=-1)
            outcome_scores = rewards_for_outcome.sum(dim=-1)
            outcome_advantages = outcome_scores.unsqueeze(-1) * response_mask.to(dtype=rewards_for_outcome.dtype)
            outcome_weights = transition_alpha
            outcome_weights_for_log = outcome_weights
            if token_level_rewards.dim() == 3:
                outcome_advantages = outcome_advantages.unsqueeze(-1)
                outcome_weights = outcome_weights.unsqueeze(-1)
            advantages = advantages + outcome_advantages * outcome_weights
        returns = advantages.clone()

    extra_metrics = {
        "ahopd_token_weights": token_weights.detach(),
        "ahopd_transition_alpha": transition_alpha.detach(),
        "ahopd_reliability": reliability.detach(),
        "ahopd_horizon": horizons.to(dtype=token_level_rewards.dtype).detach(),
        "ahopd_outcome_weights": outcome_weights_for_log.detach(),
    }
    if outcome_advantages is not None:
        extra_metrics["ahopd_outcome_advantages"] = outcome_advantages.detach()
    extra_metrics.update(reliability_components)
    return advantages, returns, extra_metrics


def _ahopd_group_accuracy(outcome_scores, index):
    if index is None:
        return outcome_scores.clamp(0.0, 1.0)

    id2scores = defaultdict(list)
    for batch_idx in range(outcome_scores.shape[0]):
        id2scores[index[batch_idx]].append(outcome_scores[batch_idx])

    group_accuracy = torch.zeros_like(outcome_scores)
    for batch_idx in range(outcome_scores.shape[0]):
        group_scores = torch.stack(id2scores[index[batch_idx]])
        group_accuracy[batch_idx] = group_scores.float().mean()
    return group_accuracy


def _ahopd_difficulty_lambda(group_accuracy, ah_cfg):
    denom = max(ah_cfg["difficulty_high"] - ah_cfg["difficulty_low"], 1e-6)
    lam = ((group_accuracy - ah_cfg["difficulty_low"]) / denom).clamp(0.0, 1.0)
    return lam.pow(ah_cfg["difficulty_power"])


def _odw_get_config(config):
    raw = _config_get(config, "odw_opd", {}) or {}
    return {
        "enabled": bool(_config_get(raw, "enabled", True)),
        "window_size": max(1, int(_config_get(raw, "window_size", 512))),
        "margin_delta": float(_config_get(raw, "margin_delta", 0.0)),
        "temperature": max(1e-6, float(_config_get(raw, "temperature", 0.1))),
        "filter_mixed": bool(_config_get(raw, "filter_mixed", True)),
        "eps": max(1e-8, float(_config_get(raw, "eps", 1e-6))),
    }


def _odw_outcome_scores(true_reward_score, token_level_rewards, response_mask):
    response_mask = response_mask.to(device=token_level_rewards.device)
    rewards_for_outcome = true_reward_score if true_reward_score is not None else token_level_rewards
    rewards_for_outcome = rewards_for_outcome.to(
        device=token_level_rewards.device, dtype=token_level_rewards.dtype
    )
    batch_size, response_length = response_mask.shape
    if rewards_for_outcome.dim() == 0 or rewards_for_outcome.shape[0] != batch_size:
        raise ValueError(
            "Outcome rewards must have one leading entry per response, got "
            f"shape {tuple(rewards_for_outcome.shape)} for batch size {batch_size}."
        )
    if rewards_for_outcome.dim() == 1:
        return rewards_for_outcome.clamp(0.0, 1.0)
    if rewards_for_outcome.dim() == 2 and rewards_for_outcome.shape[1] == 1:
        return rewards_for_outcome[:, 0].clamp(0.0, 1.0)
    if rewards_for_outcome.dim() not in (2, 3):
        raise ValueError(
            "Outcome rewards must have shape [batch], [batch, 1], [batch, response], or "
            f"[batch, response, candidates], got {tuple(rewards_for_outcome.shape)}."
        )
    if rewards_for_outcome.shape[1] != response_length:
        raise ValueError(
            "Token-level outcome rewards must match response_mask length, got "
            f"{tuple(rewards_for_outcome.shape)} and {tuple(response_mask.shape)}."
        )
    if rewards_for_outcome.dim() == 3:
        expanded_mask = (response_mask > 0).unsqueeze(-1).expand_as(rewards_for_outcome)
        if not bool(torch.isfinite(rewards_for_outcome[expanded_mask]).all().item()):
            raise ValueError("Outcome rewards contain non-finite values on valid response positions.")
        rewards_for_outcome = rewards_for_outcome.sum(dim=-1)
    valid_tokens = response_mask > 0
    if not bool(torch.isfinite(rewards_for_outcome[valid_tokens]).all().item()):
        raise ValueError("Outcome rewards contain non-finite values on valid response positions.")
    masked_rewards = torch.where(valid_tokens, rewards_for_outcome, torch.zeros_like(rewards_for_outcome))
    return masked_rewards.sum(dim=-1).clamp(0.0, 1.0)


def _oal_get_config(config):
    raw = _config_get(config, "pt_oal", {}) or {}
    outcome_validation_enabled = _config_get(raw, "outcome_validation_enabled", None)
    if outcome_validation_enabled is None:
        # Backward compatibility for the old prefix-only ablation.
        outcome_validation_enabled = str(_config_get(raw, "split_mode", "oal")) != "all"
    return {
        "enabled": bool(_config_get(raw, "enabled", True)),
        # Compatibility-only: token_reward_direct uses this for diagnostics.
        # The parameter-free POV estimator below does not read this value.
        "margin": float(_config_get(raw, "margin", 0.0)),
        "outcome_validation_enabled": bool(outcome_validation_enabled),
        "prefix_trust_enabled": bool(_config_get(raw, "prefix_trust_enabled", True)),
        "prefix_window_size": max(1, int(_config_get(raw, "prefix_window_size", 128))),
    }


def _oal_leave_one_out_outcome_advantage(outcome_scores, index):
    """Return a bounded, group-relative outcome residual for each response."""
    if index is None:
        raise ValueError("POV outcome validation requires response-group ids in `index`.")
    if len(index) != outcome_scores.shape[0]:
        raise ValueError(
            f"POV received {len(index)} response-group ids for a batch of {outcome_scores.shape[0]} responses."
        )

    id_to_positions = defaultdict(list)
    for batch_idx, group_id in enumerate(index):
        id_to_positions[group_id].append(batch_idx)

    advantages = torch.zeros_like(outcome_scores)
    for positions in id_to_positions.values():
        if len(positions) <= 1:
            continue
        position_tensor = torch.as_tensor(positions, device=outcome_scores.device, dtype=torch.long)
        group_scores = outcome_scores.index_select(0, position_tensor)
        leave_one_out_mean = (group_scores.sum() - group_scores) / float(len(positions) - 1)
        advantages[position_tensor] = group_scores - leave_one_out_mean
    return advantages


def _oal_group_centered_outcome_advantage(outcome_scores, index):
    """Return the parameter-free, group-centered outcome residual."""
    if index is None:
        raise ValueError("POV outcome validation requires response-group ids in `index`.")
    if len(index) != outcome_scores.shape[0]:
        raise ValueError(
            f"POV received {len(index)} response-group ids for a batch of {outcome_scores.shape[0]} responses."
        )

    id_to_positions = defaultdict(list)
    for batch_idx, group_id in enumerate(index):
        id_to_positions[group_id].append(batch_idx)

    advantages = torch.zeros_like(outcome_scores)
    for positions in id_to_positions.values():
        if len(positions) <= 1:
            continue
        position_tensor = torch.as_tensor(positions, device=outcome_scores.device, dtype=torch.long)
        group_scores = outcome_scores.index_select(0, position_tensor)
        advantages[position_tensor] = group_scores - group_scores.mean()
    return advantages


def _oal_group_relative_outcome_confidence(group_outcome_advantage, index):
    """Normalize outcome evidence within each prompt group without a threshold.

    Group centering says which rollout is better or worse than its siblings.
    Its absolute magnitude says how distinctive that observation is.  Dividing
    by the largest absolute residual in the same group maps this evidence to
    [0, 1] and preserves the strongest minority outcome while reducing the
    influence of less distinctive majority outcomes.  Homogeneous and
    singleton groups remain exactly zero.
    """
    if index is None:
        raise ValueError("POV outcome validation requires response-group ids in `index`.")
    if len(index) != group_outcome_advantage.shape[0]:
        raise ValueError(
            f"POV received {len(index)} response-group ids for a batch of "
            f"{group_outcome_advantage.shape[0]} responses."
        )

    id_to_positions = defaultdict(list)
    for batch_idx, group_id in enumerate(index):
        id_to_positions[group_id].append(batch_idx)

    confidence = torch.zeros_like(group_outcome_advantage)
    for positions in id_to_positions.values():
        position_tensor = torch.as_tensor(
            positions, device=group_outcome_advantage.device, dtype=torch.long
        )
        group_magnitude = group_outcome_advantage.index_select(0, position_tensor).abs()
        max_magnitude = group_magnitude.max()
        if bool((max_magnitude > 0).item()):
            confidence[position_tensor] = group_magnitude / max_magnitude
    return confidence


def _oal_response_median_abs_scale(values, valid_mask):
    """Return one robust OPD magnitude per response without introducing a scale hyperparameter."""
    scales = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    for batch_idx in range(values.shape[0]):
        valid_values = values[batch_idx][valid_mask[batch_idx].bool()]
        if valid_values.numel() > 0:
            scales[batch_idx] = valid_values.abs().median()
    return scales


def _oal_normalized_logit_delta(logit_delta, valid_mask, eps=1e-6):
    """Robustly normalize each response's OPD gap and bound it to (-1, 1)."""
    normalized = torch.zeros_like(logit_delta)
    for batch_idx in range(logit_delta.shape[0]):
        valid = valid_mask[batch_idx].bool()
        values = logit_delta[batch_idx][valid]
        if values.numel() == 0:
            continue
        scale = values.abs().median().clamp_min(eps)
        normalized[batch_idx][valid] = torch.tanh(values / scale)
    return normalized


def _pt_oal_prefix_trust_weights(
    teacher_sampled_log_probs,
    teacher_entropy,
    response_mask,
    pt_cfg,
):
    """Compute recoverable prefix trust from running-best relative teacher support.

    Window size is the only structural choice.  There is no baseline length,
    CUSUM drift, trigger threshold, or decay coefficient.  A window receives
    the ratio between its support and the best support observed so far, so a
    temporary drop does not permanently suppress the remaining response.
    """
    if teacher_sampled_log_probs.shape != response_mask.shape:
        raise ValueError(
            "POV prefix trust expects teacher_sampled_log_probs to match "
            f"response_mask, got {tuple(teacher_sampled_log_probs.shape)} and {tuple(response_mask.shape)}."
        )
    if teacher_entropy.shape != response_mask.shape:
        raise ValueError(
            "POV prefix trust expects teacher_entropy to match "
            f"response_mask, got {tuple(teacher_entropy.shape)} and {tuple(response_mask.shape)}."
        )

    teacher_sampled_log_probs = teacher_sampled_log_probs.to(device=teacher_entropy.device)
    response_mask = response_mask.to(device=teacher_entropy.device)
    mask = response_mask.to(dtype=teacher_entropy.dtype)
    valid = response_mask > 0
    if not bool(torch.isfinite(teacher_sampled_log_probs[valid]).all().item()):
        raise ValueError("POV prefix trust received non-finite teacher_sampled_log_probs on valid tokens.")
    if not bool(torch.isfinite(teacher_entropy[valid]).all().item()):
        raise ValueError("POV prefix trust received non-finite teacher_entropy on valid tokens.")
    raw_excess_surprisal = -teacher_sampled_log_probs - teacher_entropy
    excess_surprisal = torch.where(valid, raw_excess_surprisal, torch.zeros_like(raw_excess_surprisal))
    raw_token_support = torch.exp(-torch.relu(excess_surprisal))
    token_support = torch.where(valid, raw_token_support, torch.zeros_like(raw_token_support))
    prefix_weights = mask.clone()
    window_support = torch.zeros_like(mask)
    window_log_support_drop = torch.zeros_like(mask)
    relative_drop_values = torch.zeros_like(mask)
    reference_support_values = torch.zeros_like(mask)
    valid_lens = response_mask.sum(dim=-1).long()
    worst_positions = valid_lens.clone()
    half_trust_reached = torch.zeros_like(valid_lens, dtype=mask.dtype)

    window_size = pt_cfg["prefix_window_size"]
    eps = 1e-6

    for batch_idx in range(mask.shape[0]):
        valid_positions = torch.nonzero(response_mask[batch_idx] > 0, as_tuple=False).flatten()
        valid_len = int(valid_positions.numel())
        if valid_len <= 0:
            continue

        # Chunk the actual valid positions instead of assuming that response
        # masks are always a contiguous right-padded prefix.
        position_blocks = [
            valid_positions[start : min(start + window_size, valid_len)]
            for start in range(0, valid_len, window_size)
        ]
        block_support = torch.stack(
            [token_support[batch_idx, positions].mean() for positions in position_blocks]
        )
        running_best = block_support[0]
        minimum_weight = torch.ones((), device=mask.device, dtype=mask.dtype)
        worst_positions[batch_idx] = valid_positions[-1] + 1

        for block_idx, positions in enumerate(position_blocks):
            support = block_support[block_idx]
            running_best = torch.maximum(running_best, support)
            relative_weight = ((support + eps) / (running_best + eps)).clamp(0.0, 1.0)
            log_support_drop = torch.relu(torch.log((running_best + eps) / (support + eps)))
            window_support[batch_idx, positions] = support
            window_log_support_drop[batch_idx, positions] = log_support_drop
            relative_drop_values[batch_idx, positions] = log_support_drop
            reference_support_values[batch_idx, positions] = running_best
            prefix_weights[batch_idx, positions] = relative_weight

            # b* is the most weakly supported window.  Reaching half trust is
            # retained only as a diagnostic for existing audit tooling.
            if bool((relative_weight < minimum_weight).item()):
                minimum_weight = relative_weight
                worst_positions[batch_idx] = positions[0]

        if bool((minimum_weight <= 0.5 + eps).item()):
            half_trust_reached[batch_idx] = 1.0

    prefix_weights = prefix_weights * mask
    return (
        prefix_weights,
        excess_surprisal,
        token_support,
        window_support,
        window_log_support_drop,
        relative_drop_values,
        reference_support_values,
        worst_positions,
        half_trust_reached,
    )


@register_adv_est("prefix_trust_oal_opd")
def compute_outcome_aligned_logit_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    index: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Prefix-Outcome Validity interpolation for dense OPD.

    Informative outcome groups correct dense OPD only when two independent
    signals agree: the prefix is unreliable and the OPD direction conflicts
    with the trajectory outcome.  Prefix invalidity and directional conflict
    are combined with their harmonic conjunction, which stays on the scale of
    its inputs instead of vanishing as their product.  A parameter-free,
    within-group outcome confidence then prevents weak majority evidence from
    receiving the same correction strength as a distinctive minority outcome.
    The resulting weight continuously interpolates raw OPD toward a
    group-centered outcome target, robustly scaled by each response's median
    absolute OPD magnitude.

    The construction adds no tunable mixing coefficient.  A fully reliable
    prefix, zero directional conflict, or a homogeneous outcome group recovers
    raw OPD exactly.
    """
    with torch.no_grad():
        oal_cfg = _oal_get_config(config)
        if response_mask.dim() != 2:
            raise ValueError(f"POV expects a 2D response_mask, got shape {tuple(response_mask.shape)}.")
        if token_level_rewards.dim() not in (2, 3):
            raise ValueError(
                "POV expects token_level_rewards with shape [batch, response] or "
                f"[batch, response, candidates], got {tuple(token_level_rewards.shape)}."
            )
        if token_level_rewards.shape[:2] != response_mask.shape:
            raise ValueError(
                "POV expects the first two reward dimensions to match response_mask, got "
                f"{tuple(token_level_rewards.shape)} and {tuple(response_mask.shape)}."
            )

        response_mask = response_mask.to(device=token_level_rewards.device)
        mask = response_mask.to(dtype=token_level_rewards.dtype)
        if token_level_rewards.dim() == 3:
            response_candidate_mask = mask.unsqueeze(-1).expand_as(token_level_rewards)
            if oal_cfg["enabled"]:
                candidate_valid_mask = kwargs.get("candidate_valid_mask", None)
                if candidate_valid_mask is None:
                    raise ValueError(
                        "POV top-k mode requires candidate_valid_mask so interpolation cannot activate "
                        "masked intersection/union candidates."
                    )
                if candidate_valid_mask.shape != token_level_rewards.shape:
                    raise ValueError(
                        "POV expects candidate_valid_mask to match token_level_rewards, got "
                        f"{tuple(candidate_valid_mask.shape)} and {tuple(token_level_rewards.shape)}."
                    )
                valid_mask = (
                    candidate_valid_mask.to(device=token_level_rewards.device, dtype=token_level_rewards.dtype)
                    * response_candidate_mask
                )
            else:
                valid_mask = response_candidate_mask
            outcome_view_shape = (-1, 1, 1)
        else:
            valid_mask = mask
            outcome_view_shape = (-1, 1)
        reward_mask = valid_mask

        dense_advantages = torch.where(
            valid_mask.bool(), token_level_rewards, torch.zeros_like(token_level_rewards)
        )
        if not oal_cfg["enabled"]:
            returns = dense_advantages.clone()
            return dense_advantages, returns, {}

        true_reward_score = kwargs.get("true_reward_score", None)
        if oal_cfg["outcome_validation_enabled"] and true_reward_score is None:
            raise ValueError(
                "POV outcome validation requires true_reward_score; dense OPD rewards cannot be used "
                "as a correctness label."
            )
        if true_reward_score is None:
            outcome_scores = torch.zeros(
                token_level_rewards.shape[0], device=token_level_rewards.device, dtype=token_level_rewards.dtype
            )
        else:
            outcome_scores = _odw_outcome_scores(true_reward_score, token_level_rewards, response_mask)
        if outcome_scores.shape != (token_level_rewards.shape[0],):
            raise ValueError(
                "POV expects one scalar correctness outcome per response after masking, got "
                f"shape {tuple(outcome_scores.shape)} for batch size {token_level_rewards.shape[0]}."
            )
        if not bool(torch.isfinite(outcome_scores).all().item()):
            raise ValueError("POV received non-finite true_reward_score values.")
        correct = outcome_scores > 0.5
        align_scores = kwargs.get("logit_delta_scores", token_level_rewards).to(
            device=token_level_rewards.device, dtype=token_level_rewards.dtype
        )
        if align_scores.shape != token_level_rewards.shape:
            raise ValueError(
                "POV expects logit_delta_scores to match token_level_rewards, got "
                f"{tuple(align_scores.shape)} and {tuple(token_level_rewards.shape)}."
            )
        if not bool(torch.isfinite(dense_advantages[valid_mask.bool()]).all().item()):
            raise ValueError("POV received non-finite dense OPD rewards on valid positions.")
        if not bool(torch.isfinite(align_scores[valid_mask.bool()]).all().item()):
            raise ValueError("POV received non-finite logit_delta_scores on valid positions.")

        normalized_delta = _oal_normalized_logit_delta(align_scores, valid_mask)
        if oal_cfg["outcome_validation_enabled"]:
            group_outcome_advantage = _oal_group_centered_outcome_advantage(outcome_scores, index)
            outcome_confidence = _oal_group_relative_outcome_confidence(
                group_outcome_advantage, index
            )
            # The centered outcome magnitude belongs in the outcome target,
            # not in conflict detection.  Multiplying it into the conflict
            # score as well makes the correction depend quadratically on
            # group accuracy and nearly disables it for moderately mixed
            # groups.  Directional alignment is sufficient to decide whether
            # OPD points against the independently observed outcome.
            outcome_direction = torch.sign(group_outcome_advantage).view(*outcome_view_shape)
            outcome_alignment = outcome_direction * normalized_delta
            conflict_score = torch.relu(-outcome_alignment) * valid_mask
            outcome_weights = valid_mask / (1.0 + conflict_score)
        else:
            group_outcome_advantage = torch.zeros_like(outcome_scores)
            outcome_confidence = torch.zeros_like(outcome_scores)
            outcome_alignment = torch.zeros_like(align_scores)
            conflict_score = torch.zeros_like(align_scores)
            outcome_weights = valid_mask

        outcome_target_scale = _oal_response_median_abs_scale(dense_advantages, valid_mask)
        informative_outcome = group_outcome_advantage != 0
        outcome_target = (
            group_outcome_advantage * outcome_target_scale
        ).view(*outcome_view_shape) * valid_mask

        positive_outcome = group_outcome_advantage.view(*outcome_view_shape) > 0
        negative_outcome = group_outcome_advantage.view(*outcome_view_shape) < 0
        aligned = outcome_alignment > 0
        anti_aligned = outcome_alignment < 0
        pos_align_mask = (positive_outcome & aligned).to(dtype=valid_mask.dtype) * valid_mask
        pos_anti_mask = (positive_outcome & anti_aligned).to(dtype=valid_mask.dtype) * valid_mask
        neg_align_mask = (negative_outcome & aligned).to(dtype=valid_mask.dtype) * valid_mask
        neg_anti_mask = (negative_outcome & anti_aligned).to(dtype=valid_mask.dtype) * valid_mask

        teacher_sampled_log_probs = kwargs.get("teacher_sampled_log_probs", None)
        teacher_entropy = kwargs.get("teacher_entropy", None)
        if oal_cfg["prefix_trust_enabled"]:
            if teacher_sampled_log_probs is None or teacher_entropy is None:
                raise ValueError(
                    "PT-OAL requires teacher_sampled_log_probs and teacher_entropy when prefix trust is enabled."
                )
            (
                prefix_weights,
                excess_surprisal,
                candidate_support,
                window_support,
                window_log_support_drop,
                prefix_relative_drop,
                prefix_reference_support,
                prefix_horizons,
                prefix_triggered,
            ) = _pt_oal_prefix_trust_weights(
                teacher_sampled_log_probs.to(device=mask.device, dtype=mask.dtype),
                teacher_entropy.to(device=mask.device, dtype=mask.dtype),
                response_mask,
                oal_cfg,
            )
        else:
            prefix_weights = mask
            excess_surprisal = torch.zeros_like(mask)
            if teacher_sampled_log_probs is not None and teacher_entropy is not None:
                teacher_logp_for_support = teacher_sampled_log_probs.to(device=mask.device, dtype=mask.dtype)
                teacher_entropy_for_support = teacher_entropy.to(device=mask.device, dtype=mask.dtype)
                valid_tokens = response_mask > 0
                if not bool(torch.isfinite(teacher_logp_for_support[valid_tokens]).all().item()):
                    raise ValueError("POV received non-finite teacher_sampled_log_probs on valid tokens.")
                if not bool(torch.isfinite(teacher_entropy_for_support[valid_tokens]).all().item()):
                    raise ValueError("POV received non-finite teacher_entropy on valid tokens.")
                raw_candidate_support = torch.exp(
                    -torch.relu(-teacher_logp_for_support - teacher_entropy_for_support)
                )
                candidate_support = torch.where(
                    valid_tokens, raw_candidate_support, torch.zeros_like(raw_candidate_support)
                )
            else:
                candidate_support = mask
            window_support = candidate_support
            window_log_support_drop = torch.zeros_like(mask)
            prefix_relative_drop = torch.zeros_like(mask)
            prefix_reference_support = candidate_support
            prefix_horizons = response_mask.sum(dim=-1).long()
            prefix_triggered = torch.zeros_like(prefix_horizons, dtype=mask.dtype)

        if outcome_weights.dim() == 3:
            prefix_weight_view = prefix_weights.unsqueeze(-1)
        else:
            prefix_weight_view = prefix_weights

        if oal_cfg["outcome_validation_enabled"]:
            # A response-level outcome is too coarse to correct token-level
            # OPD by itself.  Require prefix invalidity at the same position.
            # The harmonic conjunction is a parameter-free fuzzy AND:
            #
            #              2 (1 - p) c
            #     alpha = -----------------,
            #              (1 - p) + c
            #
            # It is zero if either signal is absent, bounded by one, and does
            # not suffer the scale collapse of the raw product (1-p)c.  The
            # group-relative outcome confidence q then gives
            #
            #                  2 (1 - p) c
            #     alpha = q * -----------------,
            #                  (1 - p) + c
            #
            # so weak outcome residuals cannot overrule dense OPD as strongly
            # as the most distinctive outcome in the same rollout group.
            prefix_invalidity = (1.0 - prefix_weight_view).clamp(0.0, 1.0)
            conjunction_denominator = prefix_invalidity + conflict_score
            joint_invalidity = torch.where(
                conjunction_denominator > 0,
                2.0 * prefix_invalidity * conflict_score / conjunction_denominator,
                torch.zeros_like(conjunction_denominator),
            ) * valid_mask
            outcome_confidence_view = outcome_confidence.view(*outcome_view_shape)
            outcome_correction_weight = joint_invalidity * outcome_confidence_view * valid_mask
            # Preserve the dense OPD gradient exactly and use the independently
            # validated outcome only as a gated residual.  Replacing an alpha
            # fraction of raw OPD weakens the teacher signal precisely at the
            # detected conflict positions; that is especially harmful for a
            # strong student, even when alpha is small.  Residual correction
            # keeps the OPD baseline as an exact backbone and introduces no
            # additional mixing hyperparameter.
            opd_interpolation_weight = valid_mask
        else:
            # Prefix-only ablation intentionally applies prefix validity to
            # every valid OPD position because no outcome gate is available.
            opd_interpolation_weight = outcome_weights * prefix_weight_view

        # Full POV adds a scale-matched outcome residual only at jointly
        # invalid-and-conflicting positions.  The raw OPD term is never
        # attenuated.  The prefix-only ablation retains its historical
        # positional reweighting for a clean component comparison.
        if oal_cfg["outcome_validation_enabled"]:
            preliminary_advantages = (
                dense_advantages * opd_interpolation_weight
                + outcome_target * outcome_correction_weight
            ) * valid_mask
        else:
            outcome_correction_weight = torch.zeros_like(valid_mask)
            preliminary_advantages = dense_advantages * opd_interpolation_weight

        if oal_cfg["outcome_validation_enabled"]:
            # Do not apply a batch-wide mass normalization: it would couple
            # unrelated prompts and turn the conflict ratio into a stochastic
            # learning-rate multiplier.  The convex correction is already
            # locally bounded by its raw OPD and scale-matched outcome target.
            mass_renorm_scale = torch.ones((), device=mask.device, dtype=token_level_rewards.dtype)
            advantages = preliminary_advantages
        else:
            # The prefix-only ablation has no outcome-conflict gate, so retain
            # its historical absolute-mass normalization to isolate positional
            # redistribution from a global learning-rate change.  Accumulate
            # scalars in fp64 because long fp16/bf16 responses can otherwise
            # overflow even when every individual reward is finite.
            raw_abs_mass = dense_advantages.abs().sum(dtype=torch.float64)
            preliminary_abs_mass = preliminary_advantages.abs().sum(dtype=torch.float64)
            mass_eps = torch.finfo(torch.float64).tiny
            if bool((raw_abs_mass <= mass_eps).item()):
                mass_renorm_scale = torch.ones((), device=mask.device, dtype=token_level_rewards.dtype)
                advantages = dense_advantages.clone()
            elif bool((preliminary_abs_mass <= mass_eps).item()):
                # Exact cancellation should not erase a non-zero OPD batch.
                mass_renorm_scale = torch.ones((), device=mask.device, dtype=token_level_rewards.dtype)
                advantages = dense_advantages.clone()
            else:
                scale_fp64 = raw_abs_mass / preliminary_abs_mass
                mass_renorm_scale = scale_fp64.to(dtype=token_level_rewards.dtype)
                advantages = preliminary_advantages * mass_renorm_scale
                if not bool(torch.isfinite(advantages[valid_mask.bool()]).all().item()):
                    # An extreme near-cancellation can overflow when the scalar
                    # is cast back to a low-precision reward dtype.  Preserve
                    # raw OPD instead of injecting Inf/NaN into the policy loss.
                    mass_renorm_scale = torch.ones((), device=mask.device, dtype=token_level_rewards.dtype)
                    advantages = dense_advantages.clone()

        raw_response_mass = dense_advantages.abs().flatten(start_dim=1).sum(dim=-1, dtype=torch.float64)
        preliminary_response_mass = preliminary_advantages.abs().flatten(start_dim=1).sum(
            dim=-1, dtype=torch.float64
        )
        pre_renorm_mass_ratio = torch.where(
            raw_response_mass > 0,
            preliminary_response_mass / raw_response_mass.clamp_min(torch.finfo(raw_response_mass.dtype).tiny),
            torch.ones_like(raw_response_mass),
        )
        mass_renorm_scales = torch.ones_like(outcome_scores) * mass_renorm_scale
        keep_mask = opd_interpolation_weight
        if keep_mask.dim() == 3:
            token_keep_mask = (keep_mask.sum(dim=-1) > 0).to(dtype=mask.dtype) * mask
        else:
            token_keep_mask = (keep_mask > 0).to(dtype=mask.dtype) * mask

        returns = advantages.clone()

    extra_metrics = {
        "oal_keep_mask": keep_mask.detach(),
        "oal_token_keep_mask": token_keep_mask.detach(),
        "oal_outcome_scores": outcome_scores.detach(),
        "oal_correct_mask": correct.to(dtype=mask.dtype).detach(),
        "oal_group_outcome_advantage": group_outcome_advantage.detach(),
        "oal_outcome_confidence": outcome_confidence.detach(),
        "oal_normalized_logit_delta": normalized_delta.detach(),
        "oal_outcome_alignment": outcome_alignment.detach(),
        "oal_conflict_score": conflict_score.detach(),
        "oal_outcome_correction_weight": outcome_correction_weight.detach(),
        "oal_outcome_weights": outcome_weights.detach(),
        "oal_outcome_target": outcome_target.detach(),
        "oal_outcome_target_scale": outcome_target_scale.detach(),
        "oal_informative_outcome_mask": informative_outcome.to(dtype=mask.dtype).detach(),
        "oal_pre_renorm_mass_ratio": pre_renorm_mass_ratio.detach(),
        "oal_mass_renorm_scale": mass_renorm_scales.detach(),
        "oal_pos_align_mask": pos_align_mask.detach(),
        "oal_pos_anti_mask": pos_anti_mask.detach(),
        "oal_neg_align_mask": neg_align_mask.detach(),
        "oal_neg_anti_mask": neg_anti_mask.detach(),
        "oal_token_weights": keep_mask.detach(),
        "pt_oal_prefix_weights": prefix_weights.detach(),
        "pt_oal_excess_surprisal": excess_surprisal.detach(),
        "pt_oal_candidate_support": candidate_support.detach(),
        "pt_oal_window_support": window_support.detach(),
        "pt_oal_window_log_support_drop": window_log_support_drop.detach(),
        # Legacy alias retained so older dashboards do not fail.
        "pt_oal_window_excess": window_log_support_drop.detach(),
        "pt_oal_relative_log_support_drop": prefix_relative_drop.detach(),
        # Legacy aliases retained so older dashboards and audit scripts load.
        "pt_oal_cusum": prefix_relative_drop.detach(),
        "pt_oal_reference_support": prefix_reference_support.detach(),
        "pt_oal_baseline_support": prefix_reference_support.detach(),
        "pt_oal_horizon": prefix_horizons.to(dtype=mask.dtype).detach(),
        "pt_oal_triggered": prefix_triggered.detach(),
    }
    return advantages, returns, extra_metrics


@register_adv_est("outcome_discriminative_window_opd")
def compute_outcome_discriminative_window_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    index: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Outcome-Discriminative Window OPD.

    For each prompt group, keep only mixed-outcome groups. In every local
    response window, compare the teacher dense reward assigned to correct
    trajectories against wrong trajectories. A window receives a large weight
    only when teacher rewards rank correct trajectories above wrong ones.
    """
    with torch.no_grad():
        odw_cfg = _odw_get_config(config)
        if token_level_rewards.dim() == 3:
            reward_mask = response_mask.unsqueeze(-1).to(dtype=token_level_rewards.dtype)
        else:
            reward_mask = response_mask.to(dtype=token_level_rewards.dtype)

        dense_advantages = token_level_rewards * reward_mask
        if not odw_cfg["enabled"]:
            returns = dense_advantages.clone()
            return dense_advantages, returns, {}

        mask = response_mask.to(dtype=token_level_rewards.dtype)
        token_scores = _ahopd_token_scores(token_level_rewards).to(dtype=token_level_rewards.dtype)
        outcome_scores = _odw_outcome_scores(kwargs.get("true_reward_score", None), token_level_rewards, response_mask)
        correct = outcome_scores > 0.5

        batch_size, seq_len = response_mask.shape
        window_size = odw_cfg["window_size"]
        window_weights = torch.zeros_like(mask)
        window_deltas = torch.zeros_like(mask)
        group_keep = torch.zeros(batch_size, device=mask.device, dtype=mask.dtype)
        mixed_group = torch.zeros_like(group_keep)
        all_correct_group = torch.zeros_like(group_keep)
        all_wrong_group = torch.zeros_like(group_keep)

        if index is None:
            index = np.zeros(batch_size, dtype=np.int64)

        id2indices = defaultdict(list)
        for batch_idx in range(batch_size):
            id2indices[index[batch_idx]].append(batch_idx)

        for group_indices in id2indices.values():
            group_correct = correct[group_indices]
            has_correct = bool(group_correct.any().item())
            has_wrong = bool((~group_correct).any().item())

            if has_correct and has_wrong:
                for batch_idx in group_indices:
                    group_keep[batch_idx] = 1.0
                    mixed_group[batch_idx] = 1.0
            elif has_correct:
                for batch_idx in group_indices:
                    all_correct_group[batch_idx] = 1.0
                if odw_cfg["filter_mixed"]:
                    continue
            else:
                for batch_idx in group_indices:
                    all_wrong_group[batch_idx] = 1.0
                if odw_cfg["filter_mixed"]:
                    continue

            correct_indices = [i for i in group_indices if bool(correct[i].item())]
            wrong_indices = [i for i in group_indices if not bool(correct[i].item())]
            if len(correct_indices) == 0 or len(wrong_indices) == 0:
                for start in range(0, seq_len, window_size):
                    end = min(start + window_size, seq_len)
                    for batch_idx in group_indices:
                        window_weights[batch_idx, start:end] = 1.0
                continue

            for start in range(0, seq_len, window_size):
                end = min(start + window_size, seq_len)

                pos_values = []
                neg_values = []
                for batch_idx in correct_indices:
                    valid = mask[batch_idx, start:end] > 0
                    if valid.any():
                        pos_values.append(token_scores[batch_idx, start:end][valid].mean())
                for batch_idx in wrong_indices:
                    valid = mask[batch_idx, start:end] > 0
                    if valid.any():
                        neg_values.append(token_scores[batch_idx, start:end][valid].mean())

                if len(pos_values) == 0 or len(neg_values) == 0:
                    continue

                pos_mean = torch.stack(pos_values).mean()
                neg_mean = torch.stack(neg_values).mean()
                delta = pos_mean - neg_mean
                weight = torch.sigmoid((delta - odw_cfg["margin_delta"]) / odw_cfg["temperature"])

                for batch_idx in group_indices:
                    window_weights[batch_idx, start:end] = weight
                    window_deltas[batch_idx, start:end] = delta

        window_weights = window_weights * mask
        if token_level_rewards.dim() == 3:
            advantages = dense_advantages * window_weights.unsqueeze(-1)
        else:
            advantages = dense_advantages * window_weights
        returns = advantages.clone()

    extra_metrics = {
        "odw_window_weights": window_weights.detach(),
        "odw_window_deltas": window_deltas.detach(),
        "odw_group_keep": group_keep.detach(),
        "odw_mixed_group": mixed_group.detach(),
        "odw_all_correct_group": all_correct_group.detach(),
        "odw_all_wrong_group": all_wrong_group.detach(),
        "odw_outcome_scores": outcome_scores.detach(),
        "odw_positive_mask": correct.to(dtype=mask.dtype).detach(),
        "odw_negative_mask": (~correct).to(dtype=mask.dtype).detach(),
        "odw_loss_mask": (group_keep.unsqueeze(-1) * mask).detach(),
    }
    return advantages, returns, extra_metrics


@register_adv_est("difficulty_aware_outcome_augmented_opd")
def compute_difficulty_aware_outcome_augmented_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    index: Optional[np.ndarray] = None,
    overlap_mask: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    student_entropy: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Difficulty-Aware Outcome-Augmented OPD.

    Keep teacher dense OPD supervision everywhere and add outcome supervision
    only where it is likely to be useful:

        A_t = A_teacher_t + lambda(q) * alpha_t * A_outcome

    lambda(q) is estimated from the on-policy group accuracy of the prompt.
    alpha_t is zero before the adaptive horizon and increases toward the end
    of the response.
    """
    with torch.no_grad():
        ah_cfg = _ahopd_get_config(config)
        if token_level_rewards.dim() == 3:
            reward_mask = response_mask.unsqueeze(-1).to(dtype=token_level_rewards.dtype)
        else:
            reward_mask = response_mask.to(dtype=token_level_rewards.dtype)

        dense_advantages = token_level_rewards * reward_mask
        if not ah_cfg["enabled"]:
            returns = dense_advantages.clone()
            return dense_advantages, returns, {}

        _, reliability, horizons, reliability_components = _ahopd_reliability_weights(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            config=config,
            overlap_mask=overlap_mask,
            teacher_entropy=teacher_entropy,
            student_entropy=student_entropy,
        )

        transition_alpha = _ahopd_transition_progress(
            response_mask=response_mask,
            horizons=horizons,
            power=ah_cfg["transition_power"],
        ).to(dtype=token_level_rewards.dtype)

        rewards_for_outcome = kwargs.get("true_reward_score", token_level_rewards)
        if rewards_for_outcome.dim() == 3:
            rewards_for_outcome = rewards_for_outcome.sum(dim=-1)
        rewards_for_outcome = rewards_for_outcome.to(dtype=token_level_rewards.dtype)
        outcome_scores = rewards_for_outcome.sum(dim=-1).clamp(0.0, 1.0)
        group_accuracy = _ahopd_group_accuracy(outcome_scores, index)
        difficulty_lambda = _ahopd_difficulty_lambda(group_accuracy, ah_cfg)

        outcome_weights = (
            ah_cfg["outcome_weight"]
            * difficulty_lambda.unsqueeze(-1)
            * transition_alpha
            * response_mask.to(dtype=token_level_rewards.dtype)
        )
        outcome_advantages = outcome_scores.unsqueeze(-1) * response_mask.to(dtype=token_level_rewards.dtype)

        if token_level_rewards.dim() == 3:
            advantages = dense_advantages + outcome_advantages.unsqueeze(-1) * outcome_weights.unsqueeze(-1)
        else:
            advantages = dense_advantages + outcome_advantages * outcome_weights
        returns = advantages.clone()

    extra_metrics = {
        "daoa_horizon": horizons.to(dtype=token_level_rewards.dtype).detach(),
        "daoa_reliability": reliability.detach(),
        "daoa_transition_alpha": transition_alpha.detach(),
        "daoa_group_accuracy": group_accuracy.detach(),
        "daoa_difficulty_lambda": difficulty_lambda.detach(),
        "daoa_outcome_scores": outcome_scores.detach(),
        "daoa_outcome_weights": outcome_weights.detach(),
        "daoa_outcome_advantages": outcome_advantages.detach(),
        "daoa_teacher_weights": response_mask.to(dtype=token_level_rewards.dtype).detach(),
        # Reuse the existing AH-OPD visualization code path for position-wise
        # reliability, teacher weight, outcome weight, and horizon plots.
        "ahopd_horizon": horizons.to(dtype=token_level_rewards.dtype).detach(),
        "ahopd_reliability": reliability.detach(),
        "ahopd_transition_alpha": transition_alpha.detach(),
        "ahopd_token_weights": response_mask.to(dtype=token_level_rewards.dtype).detach(),
        "ahopd_outcome_weights": outcome_weights.detach(),
        "ahopd_outcome_advantages": outcome_advantages.detach(),
    }
    extra_metrics.update(reliability_components)
    return advantages, returns, extra_metrics


@register_adv_est("token_reward_direct_plus_grpo")
def compute_token_reward_direct_plus_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine token_reward_direct and GRPO outcome advantage.
    adv = direct_adv + weight * grpo_adv
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group index
        config: AlgoConfig
    """
    # 1. Compute direct advantage
    direct_adv, _ = compute_token_reward_direct_advantage(
        token_level_rewards, response_mask, config, **kwargs
    )
    
    # 2. Compute GRPO advantage
    # Use true_reward_score if available (raw reward without KL penalty), otherwise use token_level_rewards
    rewards_for_grpo = kwargs.get("true_reward_score", token_level_rewards)
    
    norm_adv_by_std_in_grpo = config.norm_adv_by_std_in_grpo if config else True
    grpo_adv, _ = compute_grpo_outcome_advantage(
        rewards_for_grpo, response_mask, index, 
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config
    )
    
    # 3. Combine
    weight = config.grpo_outcome_weight if config else 1.0
    
    combined_adv = direct_adv + weight * grpo_adv
    # Since token_reward_direct sets returns=adv, we follow suit
    combined_returns = combined_adv.clone()
    
    return combined_adv, combined_returns, {"token_level_advantage_direct": direct_adv}

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        loss = verl_F.masked_mean(seq_losses, seq_mask)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        loss = verl_F.masked_mean(seq_losses, seq_mask)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    format_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    # Handle 3D tensors from top-k sampling (e.g., top_k_strategy="only_stu")
    # When log_prob and old_log_prob are 3D, we compute per-token-in-top-k losses
    # 
    # For ppo_epochs=1 (on-policy), ratio(x) ≈ 1, so we use a MEMORY-EFFICIENT formulation:
    # Standard PPO: L = -ratio × A, ∇L = -A × ratio × ∇log π
    # With ratio=1: ∇L ≈ -A × ∇log π = ∇(-A × log π)
    # So we compute: L = -Σ_x [A(x) × log π_θ(x)]  (treating A as constant)
    # This avoids storing the 3D ratio tensor, saving significant memory.
    #
    if log_prob.dim() == 3 and old_log_prob.dim() == 3:
        print(f"Advantages shape: {advantages.shape}")
        
        # Standard PPO case adapted for 3D tensors (Top-K)
        # 1. Compute KL and Ratio (3D)
        negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)

        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        pg_losses = torch.sum(pg_losses, dim=-1) # [MOD] for wings, sum across wings-tokens
        
        # 4. Metrics
        with torch.no_grad():
            # KL: Sum over K, then masked mean
            ppo_kl = verl_F.masked_mean(-negative_approx_kl.sum(dim=-1), response_mask)
            
            # Clipfrac: Need to broadcast mask to 3D for correct calculation
            # mask: (B, T) -> (B, T, 1) broadcastable to (B, T, K)
            mask_3d = response_mask.unsqueeze(-1)
            pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), mask_3d)
            pg_clipfrac_lower = torch.tensor(0.0, device=log_prob.device) # Placeholder
    else:
        # Standard 2D case
        negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
        )

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    if format_mask is not None:
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask * format_mask.unsqueeze(-1), loss_agg_mode=loss_agg_mode)
    else:
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_with_rollout_correction(
    rollout_log_prob,
    log_prob,
    advantages,
    eos_mask,
    loss_agg_mode="seq-mean-token-sum",
    loss_scale_factor=1.0,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: float = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
):
    """Compute policy loss with pure rollout correction (no PPO clipping).

    This function implements policy gradient with importance sampling correction
    for rollout-training policy mismatch, without PPO's clipping mechanism.

    Mathematical formulation:
        Without IS (rollout_is=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is enabled):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length)
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        eos_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        loss_scale_factor: Multiplicative scaling factor applied to final loss.
        rollout_is: IS aggregation level ("token", "sequence", or None).
        rollout_is_threshold: Upper threshold for truncating IS weights.
        rollout_rs: Rejection sampling aggregation level (or None to disable).
        rollout_rs_threshold: Upper threshold for rejection sampling.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling.
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers.

    Returns:
        Tuple of (loss, clip_fraction, kl_divergence, clip_fraction_lower):
            - loss: Policy gradient loss with IS correction
            - clip_fraction: Always 0.0 (no clipping in this mode)
            - kl_divergence: KL between current and rollout policy
            - clip_fraction_lower: Always 0.0 (no clipping in this mode)
        Note: Rollout correction metrics are computed internally but not returned.
              Caller should compute them separately if needed.

    Note:
        Unlike compute_policy_loss (PPO), this function:
        - Does NOT use PPO clipping (no old_log_prob needed)
        - Directly applies IS correction computed from current vs rollout
        - Computes IS/RS on-the-fly during training

    Usage:
        This function is called by the actor when:
        - bypass_old_logprob_for_rollout=True (trainer uses rollout_log_prob as old_log_prob)
        - use_pure_rollout_correction=True (actor uses this function instead of compute_policy_loss)

    Example config:
        algorithm:
          rollout_correction:
            bypass_old_logprob_for_rollout: true
            use_pure_rollout_correction: true
            rollout_is: "token"
            rollout_is_threshold: 2.0
            rollout_rs: "token"
            rollout_rs_threshold: 2.0
            rollout_rs_threshold_lower: 0.5

    Performance:
        - Memory: Saves ~1MB per batch (no old_log_prob storage)
        - Speed: ~15-20% faster (skips actor.compute_log_prob())
        - Variance: Higher than PPO (no clipping safety net)
    """
    # Import rollout correction helper
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    # Compute IS weights and rejection mask on-the-fly
    rollout_is_weights_proto, modified_response_mask, rollout_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob,  # Current policy
        rollout_log_prob=rollout_log_prob,  # Rollout policy
        response_mask=eos_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
    )

    # Extract weights tensor from DataProto (or None if disabled)
    rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (if RS is enabled)
    effective_mask = modified_response_mask if rollout_rs is not None else eos_mask

    # Compute pure policy gradient loss with IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    #
    # Note: rollout_is_weights already contains w = π_current / π_rollout
    # So we apply it to the standard log-prob trick formula

    if rollout_is_weights is not None:
        # With IS correction: weight the log-prob trick by IS weight
        # w = exp(log_prob - rollout_log_prob).clamp(max=threshold)
        # L = -E[w * log π * A]
        # Gradient: ∇L = -E[w * ∇log π * A] = -E[w * A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # No IS correction: standard REINFORCE with log-prob trick
        # L = -E[log π(a|s) * A]
        # Gradient: ∇L = -E[∇log π * A] = -E[A]
        pg_losses = -advantages * log_prob

    # Aggregate loss (apply scale factor manually)
    pg_loss = (
        agg_loss(
            loss_mat=pg_losses,
            loss_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
        )
        * loss_scale_factor
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, effective_mask)

    pg_metrics = rollout_metrics
    pg_metrics.update(
        {
            "actor/ppo_kl": kl_divergence.detach().item(),
        }
    )

    return pg_loss, pg_metrics


@register_policy_loss("rollout_correction")
def compute_policy_loss_rollout_correction_wrapper(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Wrapper for compute_policy_loss_with_rollout_correction to match PolicyLossFn interface.

    This function is used when algorithm.rollout_correction.use_pure_rollout_correction=True.
    In this mode, the trainer has already set old_log_prob=rollout_log_prob (bypass mode).

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob
        log_prob: Current policy log probabilities
        advantages: Advantage estimates
        response_mask: Valid token mask
        loss_agg_mode: Loss aggregation mode
        config: Actor config containing rollout_correction settings
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally)

    Returns:
        Tuple of (loss, clip_fraction, kl, clip_fraction_lower)
    """
    assert config is not None, "config is required for rollout_correction loss mode"

    # Extract rollout_correction config
    # In ray_trainer, when use_pure_rollout_correction=True, the rollout_correction config
    # is embedded in actor config's policy_loss field
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='rollout_correction', ensure rollout_correction config is passed."
        )

    # Extract parameters
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)

    # Call the actual implementation
    # In bypass mode, old_log_prob IS rollout_log_prob
    return compute_policy_loss_with_rollout_correction(
        rollout_log_prob=old_log_prob,  # This is rollout_log_prob in bypass mode
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        loss_scale_factor=1.0,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
    )
