import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos_pt_oal import (
    _oal_group_centered_outcome_advantage,
    _oal_normalized_logit_delta,
    _pt_oal_prefix_trust_weights,
    compute_outcome_aligned_logit_opd_advantage,
)


def _pov_config(*, outcome=True, prefix=True, window_size=2):
    return {
        "pt_oal": {
            "enabled": True,
            "outcome_validation_enabled": outcome,
            "prefix_trust_enabled": prefix,
            "prefix_window_size": window_size,
        }
    }


def test_group_centered_outcome_advantage_is_parameter_free():
    scores = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    group_ids = np.array(["mixed", "mixed", "same", "same", "singleton"], dtype=object)

    result = _oal_group_centered_outcome_advantage(scores, group_ids)

    torch.testing.assert_close(result, torch.tensor([0.5, -0.5, 0.0, 0.0, 0.0]))


def test_group_centered_outcome_advantage_requires_group_ids():
    with pytest.raises(ValueError, match="response-group ids"):
        _oal_group_centered_outcome_advantage(torch.tensor([1.0, 0.0]), None)


def test_normalized_logit_delta_ignores_padding_when_computing_scale():
    delta = torch.tensor([[1.0, 2.0, 1000.0]])
    valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    result = _oal_normalized_logit_delta(delta, valid_mask)

    # torch.median uses the lower middle value, so the valid scale is 1.0.
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
        {"prefix_window_size": 2},
    )

    prefix_weights, _, _, _, _, relative_drop, reference, horizons, half_trust_reached = result
    torch.testing.assert_close(prefix_weights, mask)
    torch.testing.assert_close(relative_drop, torch.zeros_like(mask))
    torch.testing.assert_close(reference, mask)
    assert horizons.item() == 6
    assert half_trust_reached.item() == 0.0


def test_prefix_trust_uses_running_best_and_recovers_after_support_recovers():
    mask = torch.ones(1, 6)
    # With W=2, the three support values are 1.0, 0.5, and 1.0.
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056, 0.0, 0.0]]
    )
    teacher_entropy = torch.zeros_like(mask)

    result = _pt_oal_prefix_trust_weights(
        teacher_log_probs,
        teacher_entropy,
        mask,
        {"prefix_window_size": 2},
    )

    prefix_weights, _, _, window_support, log_drop, relative_drop, reference, horizon, half_trust = result
    torch.testing.assert_close(
        prefix_weights,
        torch.tensor([[1.0, 1.0, 0.5, 0.5, 1.0, 1.0]]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        window_support,
        torch.tensor([[1.0, 1.0, 0.5, 0.5, 1.0, 1.0]]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(reference, torch.ones_like(reference), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(log_drop, relative_drop)
    assert torch.all(relative_drop[0, 2:4] > 0)
    assert torch.all(relative_drop[0, :2] == 0)
    assert torch.all(relative_drop[0, 4:] == 0)
    assert horizon.item() == 2
    assert half_trust.item() == 1.0


def test_prefix_trust_chunks_actual_valid_positions_when_mask_has_holes():
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    teacher_log_probs = torch.tensor([[0.0, float("nan"), -0.69314718056, float("inf")]])
    teacher_entropy = torch.zeros_like(mask)

    result = _pt_oal_prefix_trust_weights(
        teacher_log_probs,
        teacher_entropy,
        mask,
        {"prefix_window_size": 1},
    )

    prefix_weights, _, _, _, _, _, _, horizon, _ = result
    torch.testing.assert_close(
        prefix_weights,
        torch.tensor([[1.0, 0.0, 0.5, 0.0]]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert horizon.item() == 2
    assert torch.isfinite(prefix_weights).all()


def test_full_pov_interpolates_conflicts_and_preserves_batch_absolute_mass():
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
        config=_pov_config(),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=zero_teacher_stat,
        teacher_entropy=zero_teacher_stat,
        logit_delta_scores=raw_opd,
    )

    torch.testing.assert_close(extras["oal_group_outcome_advantage"], torch.tensor([0.5, -0.5]))
    # Correct rollout: positive OPD is aligned. Wrong rollout: negative OPD is aligned.
    assert extras["oal_outcome_weights"][0, 0].item() == 1.0
    assert extras["oal_outcome_weights"][0, 1].item() < 1.0
    assert extras["oal_outcome_weights"][1, 0].item() < 1.0
    assert extras["oal_outcome_weights"][1, 1].item() == 1.0
    assert torch.all(extras["pt_oal_prefix_weights"] == 1)

    target_scale = extras["oal_outcome_target_scale"].view(-1, 1)
    target = extras["oal_group_outcome_advantage"].view(-1, 1) * target_scale
    q = extras["oal_keep_mask"]
    preliminary = q * raw_opd + (1.0 - q) * target
    expected = preliminary * (raw_opd.abs().sum() / preliminary.abs().sum())
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(advantages.abs().sum(), raw_opd.abs().sum())
    assert not torch.allclose(advantages, raw_opd * q)


def test_full_pov_applies_prefix_only_to_outcome_conflicts():
    raw_opd = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    mask = torch.ones_like(raw_opd)
    # The second window has half the teacher support of the first window.
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056]]
    ).expand_as(raw_opd)
    teacher_entropy = torch.zeros_like(raw_opd)

    _, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )

    keep = extras["oal_keep_mask"]
    prefix = extras["pt_oal_prefix_weights"]
    outcome = extras["oal_outcome_weights"]
    conflict = extras["oal_conflict_score"] > 0

    torch.testing.assert_close(prefix[:, 2:], torch.full((2, 2), 0.5), atol=1e-5, rtol=1e-5)
    # Prefix validity may strengthen an existing outcome correction, but it
    # must not alter raw OPD where outcome and OPD already agree.
    torch.testing.assert_close(keep[~conflict], torch.ones_like(keep[~conflict]))
    torch.testing.assert_close(keep[conflict], (outcome * prefix)[conflict])
    assert keep[0, 2].item() == 1.0  # correct rollout, positive OPD: aligned
    assert keep[1, 3].item() == 1.0  # wrong rollout, negative OPD: aligned
    assert keep[0, 3].item() < 0.5   # correct rollout, negative OPD: conflict
    assert keep[1, 2].item() < 0.5   # wrong rollout, positive OPD: conflict


def test_homogeneous_outcomes_fall_back_to_raw_opd_even_with_weak_prefix():
    raw_opd = torch.tensor([[1.0, -0.5, 0.7, -0.2], [-0.3, 0.8, -1.0, 0.4]])
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056]]
    ).expand_as(raw_opd)
    teacher_entropy = torch.zeros_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.ones(2),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )

    torch.testing.assert_close(extras["oal_group_outcome_advantage"], torch.zeros(2))
    assert extras["pt_oal_prefix_weights"][0, -1].item() < 1.0
    torch.testing.assert_close(advantages, raw_opd)


def test_prefix_only_ablation_reweights_positions_but_preserves_absolute_mass():
    raw_opd = torch.ones(1, 6)
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056, 0.0, 0.0]]
    )
    teacher_entropy = torch.zeros_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(outcome=False),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )

    expected_preliminary = raw_opd * extras["pt_oal_prefix_weights"]
    expected = expected_preliminary * (raw_opd.abs().sum() / expected_preliminary.abs().sum())
    torch.testing.assert_close(advantages, expected)
    assert advantages[0, 2].item() < advantages[0, 0].item()
    assert advantages[0, 4].item() == pytest.approx(advantages[0, 0].item())
    torch.testing.assert_close(advantages.abs().sum(), raw_opd.abs().sum())


def test_low_precision_mass_normalization_stays_finite():
    raw_opd = torch.tensor([[0.0, 0.0, 40000.0, 40000.0]], dtype=torch.float16)
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor(
        [[0.0, 0.0, -0.69314718056, -0.69314718056]], dtype=torch.float16
    )
    teacher_entropy = torch.zeros_like(raw_opd)

    advantages, _, _ = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(outcome=False),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )

    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(
        advantages.float().abs().sum(), raw_opd.float().abs().sum(), rtol=1e-3, atol=1.0
    )


def test_zero_raw_opd_stays_zero_and_finite():
    raw_opd = torch.zeros(2, 4)
    mask = torch.ones_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=raw_opd,
    )

    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages, raw_opd)
    torch.testing.assert_close(extras["oal_outcome_target_scale"], torch.zeros(2))


def test_column_vector_outcome_is_treated_as_one_scalar_per_response():
    raw_opd = torch.ones(2, 4)
    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=torch.ones_like(raw_opd),
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([[0.25], [0.75]]),
        logit_delta_scores=raw_opd,
    )

    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(extras["oal_outcome_scores"], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(extras["oal_group_outcome_advantage"], torch.tensor([-0.25, 0.25]))


def test_token_outcome_length_must_match_response_mask():
    with pytest.raises(ValueError, match="match response_mask length"):
        compute_outcome_aligned_logit_opd_advantage(
            token_level_rewards=torch.ones(2, 4),
            response_mask=torch.ones(2, 4),
            config=_pov_config(prefix=False),
            index=np.array(["prompt", "prompt"], dtype=object),
            true_reward_score=torch.ones(2, 3),
        )


def test_topk_pov_keeps_invalid_candidates_zero():
    raw_opd = torch.tensor(
        [
            [[1.0, 100000.0, -0.5], [100000.0, -4.0, 100000.0]],
            [[-1.0, 2.5, 100000.0], [100000.0, 100000.0, -0.7]],
        ]
    )
    response_mask = torch.ones(2, 2)
    candidate_mask = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 1]],
        ],
        dtype=torch.bool,
    )

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=response_mask,
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=raw_opd,
        candidate_valid_mask=candidate_mask,
    )

    assert torch.all(advantages[~candidate_mask] == 0)
    dense_valid = raw_opd * candidate_mask
    torch.testing.assert_close(advantages.abs().sum(), dense_valid.abs().sum())

    perturbed = raw_opd.clone()
    perturbed[~candidate_mask] = -900000.0
    perturbed_advantages, _, perturbed_extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=perturbed,
        response_mask=response_mask,
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=perturbed,
        candidate_valid_mask=candidate_mask,
    )
    torch.testing.assert_close(
        extras["oal_outcome_target_scale"], perturbed_extras["oal_outcome_target_scale"]
    )
    torch.testing.assert_close(advantages[candidate_mask], perturbed_advantages[candidate_mask])


def test_topk_pov_requires_candidate_valid_mask():
    with pytest.raises(ValueError, match="candidate_valid_mask"):
        compute_outcome_aligned_logit_opd_advantage(
            token_level_rewards=torch.ones(2, 2, 3),
            response_mask=torch.ones(2, 2),
            config=_pov_config(prefix=False),
            index=np.array(["prompt", "prompt"], dtype=object),
            true_reward_score=torch.tensor([1.0, 0.0]),
        )


def test_topk_pov_clears_nonfinite_values_on_invalid_candidates():
    candidate_mask = torch.tensor(
        [
            [[1, 0], [1, 0]],
            [[1, 0], [1, 0]],
        ],
        dtype=torch.bool,
    )
    raw_opd = torch.tensor(
        [
            [[1.0, float("nan")], [-0.5, float("inf")]],
            [[-1.0, float("-inf")], [0.5, float("nan")]],
        ]
    )

    advantages, _, _ = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=torch.ones(2, 2),
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=raw_opd,
        candidate_valid_mask=candidate_mask,
    )

    assert torch.isfinite(advantages).all()
    assert torch.all(advantages[~candidate_mask] == 0)


def test_outcome_validation_requires_true_correctness_reward():
    with pytest.raises(ValueError, match="true_reward_score"):
        compute_outcome_aligned_logit_opd_advantage(
            token_level_rewards=torch.ones(2, 2),
            response_mask=torch.ones(2, 2),
            config=_pov_config(prefix=False),
            index=np.array(["prompt", "prompt"], dtype=object),
        )
