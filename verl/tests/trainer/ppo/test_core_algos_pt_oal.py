import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos_pt_oal import (
    _oal_leave_one_out_outcome_advantage,
    _oal_normalized_logit_delta,
    _pt_oal_prefix_trust_weights,
    compute_outcome_aligned_logit_opd_advantage,
)


def test_leave_one_out_outcome_advantage_is_group_relative():
    scores = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    group_ids = np.array(["mixed", "mixed", "same", "same", "singleton"], dtype=object)

    result = _oal_leave_one_out_outcome_advantage(scores, group_ids)

    torch.testing.assert_close(result, torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0]))


def test_leave_one_out_outcome_advantage_requires_group_ids():
    with pytest.raises(ValueError, match="response-group ids"):
        _oal_leave_one_out_outcome_advantage(torch.tensor([1.0, 0.0]), None)


def test_normalized_logit_delta_ignores_padding_when_computing_scale():
    delta = torch.tensor([[1.0, 2.0, 1000.0]])
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    result = _oal_normalized_logit_delta(delta, valid_mask)

    # The valid absolute median is 1.0; the padded outlier must not alter it.
    torch.testing.assert_close(result[0, :2], torch.tanh(torch.tensor([1.0, 2.0])))
    assert result[0, 2].item() == 0.0


def test_prefix_trust_is_one_for_constant_support():
    mask = torch.ones(1, 6)
    teacher_log_probs = torch.zeros_like(mask)
    teacher_entropy = torch.zeros_like(mask)

    result = _pt_oal_prefix_trust_weights(
        teacher_log_probs,
        teacher_entropy,
        mask,
        {"prefix_window_size": 2, "prefix_baseline_blocks": 1},
    )

    prefix_weights, _, _, _, _, cusum, _, horizons, half_trust_reached = result
    torch.testing.assert_close(prefix_weights, mask)
    torch.testing.assert_close(cusum, torch.zeros_like(mask))
    assert horizons.item() == 6
    assert half_trust_reached.item() == 0.0


def test_sustained_support_drop_produces_monotone_threshold_free_decay():
    mask = torch.ones(1, 6)
    # W=2 and L=1. Baseline support is 1; both later windows have support 0.5.
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056, -0.69314718056, -0.69314718056]]
    )
    teacher_entropy = torch.zeros_like(mask)

    result = _pt_oal_prefix_trust_weights(
        teacher_log_probs,
        teacher_entropy,
        mask,
        {"prefix_window_size": 2, "prefix_baseline_blocks": 1},
    )

    prefix_weights, _, _, window_support, log_drop, cusum, _, horizons, half_trust_reached = result
    torch.testing.assert_close(prefix_weights[0, :2], torch.ones(2))
    torch.testing.assert_close(prefix_weights[0, 2:4], torch.full((2,), 0.5), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prefix_weights[0, 4:6], torch.full((2,), 0.25), atol=1e-5, rtol=1e-5)
    assert torch.all(prefix_weights[:, 1:] <= prefix_weights[:, :-1])
    torch.testing.assert_close(window_support[0, 2:], torch.full((4,), 0.5), atol=1e-5, rtol=1e-5)
    assert torch.all(log_drop[0, 2:] > 0)
    assert torch.all(cusum[0, 4:] > cusum[0, 2:4])
    assert horizons.item() == 2
    assert half_trust_reached.item() == 1.0


def test_full_pov_softly_suppresses_only_outcome_conflicts():
    raw_opd = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5],
            [1.0, -1.0, 0.5, -0.5],
        ]
    )
    mask = torch.ones_like(raw_opd)
    zero_teacher_stat = torch.zeros_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config={
            "pt_oal": {
                "enabled": True,
                "outcome_validation_enabled": True,
                "prefix_trust_enabled": True,
                "prefix_window_size": 2,
                "prefix_baseline_blocks": 1,
            }
        },
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=zero_teacher_stat,
        teacher_entropy=zero_teacher_stat,
        logit_delta_scores=raw_opd,
    )

    torch.testing.assert_close(extras["oal_group_outcome_advantage"], torch.tensor([1.0, -1.0]))
    # Correct rollout: positive OPD is aligned. Wrong rollout: negative OPD is aligned.
    assert extras["oal_outcome_weights"][0, 0].item() == 1.0
    assert extras["oal_outcome_weights"][0, 1].item() < 1.0
    assert extras["oal_outcome_weights"][1, 0].item() < 1.0
    assert extras["oal_outcome_weights"][1, 1].item() == 1.0
    assert torch.all(extras["oal_outcome_weights"] >= 0)
    assert torch.all(extras["oal_outcome_weights"] <= 1)
    assert torch.all(extras["pt_oal_prefix_weights"] == 1)
    torch.testing.assert_close(advantages, raw_opd * extras["oal_outcome_weights"])
