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


def _harmonic_joint(prefix, conflict):
    invalidity = 1.0 - prefix
    denominator = invalidity + conflict
    return torch.where(
        denominator > 0,
        2.0 * invalidity * conflict / denominator,
        torch.zeros_like(denominator),
    )


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


def test_full_pov_requires_joint_prefix_and_outcome_conflict():
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

    q = extras["oal_keep_mask"]
    preliminary = q * raw_opd
    torch.testing.assert_close(advantages, preliminary)
    # Outcome conflict alone must not change OPD when prefix validity is one.
    torch.testing.assert_close(q, torch.ones_like(q))
    torch.testing.assert_close(advantages, raw_opd)
    torch.testing.assert_close(extras["oal_mass_renorm_scale"], torch.ones(2))
    assert advantages.abs().sum().item() == raw_opd.abs().sum().item()


def test_full_pov_is_continuous_as_conflict_approaches_zero():
    raw_opd = torch.tensor([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]])
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor([[0.0, -0.69314718056, -0.69314718056]]).expand_as(raw_opd)
    teacher_entropy = torch.zeros_like(raw_opd)

    def run(second_delta):
        logit_delta = torch.tensor([[1.0, second_delta, 1.0], [-1.0, -1.0, -1.0]])
        return compute_outcome_aligned_logit_opd_advantage(
            token_level_rewards=raw_opd,
            response_mask=mask,
            config=_pov_config(window_size=1),
            index=np.array(["prompt", "prompt"], dtype=object),
            true_reward_score=torch.tensor([1.0, 0.0]),
            teacher_sampled_log_probs=teacher_log_probs,
            teacher_entropy=teacher_entropy,
            logit_delta_scores=logit_delta,
        )

    zero_conflict_advantages, _, zero_extras = run(0.0)
    tiny_conflict_advantages, _, tiny_extras = run(-1e-6)

    assert zero_extras["oal_conflict_score"][0, 1].item() == 0.0
    assert tiny_extras["oal_conflict_score"][0, 1].item() > 0.0
    assert tiny_extras["pt_oal_prefix_weights"][0, 1].item() == pytest.approx(0.5, abs=1e-5)
    # The previous hard gate jumped from q=1 to approximately p=0.5 here.
    # Harmonic fusion must instead converge continuously to the c=0 result.
    assert tiny_extras["oal_keep_mask"][0, 1].item() > 0.99999
    assert tiny_conflict_advantages[0, 1].item() == pytest.approx(
        zero_conflict_advantages[0, 1].item(), abs=5e-6
    )


def test_full_pov_with_unit_prefix_falls_back_to_raw_opd():
    raw_opd = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=torch.ones_like(raw_opd),
        config=_pov_config(prefix=False),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=raw_opd,
    )

    torch.testing.assert_close(extras["oal_keep_mask"], torch.ones_like(raw_opd))
    torch.testing.assert_close(advantages, raw_opd)


def test_full_pov_outcome_correction_is_monotone_in_conflict_evidence():
    raw_opd = -torch.ones(2, 3)
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor([[0.0, -0.69314718056, -0.69314718056]]).expand_as(raw_opd)
    teacher_entropy = torch.zeros_like(raw_opd)
    logit_delta = torch.tensor([[1.0, -0.1, -1.0], [-1.0, -0.1, -1.0]])

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(window_size=1),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=logit_delta,
    )

    conflict = extras["oal_conflict_score"][0]
    keep = extras["oal_keep_mask"][0]
    prefix = extras["pt_oal_prefix_weights"][0]
    torch.testing.assert_close(prefix[1:], torch.tensor([0.5, 0.5]), atol=1e-5, rtol=1e-5)
    assert conflict[2].item() > conflict[1].item() > 0.0
    assert keep[2].item() < keep[1].item() < 1.0
    # Both raw OPD values are identical and negative, so stronger conflict
    # evidence must move them monotonically toward the positive outcome target.
    assert advantages[0, 2].item() > advantages[0, 1].item()
    assert advantages[0, 2].item() < 0.0


def test_directional_conflict_is_not_shrunk_by_group_outcome_magnitude():
    raw_opd = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    )
    mask = torch.ones_like(raw_opd)
    teacher_log_probs = torch.tensor([[0.0, -0.69314718056]]).expand_as(raw_opd)
    teacher_entropy = torch.zeros_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(window_size=1),
        index=np.array(["prompt"] * 4, dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )

    # The single correct rollout has centered outcome +0.75 while each wrong
    # rollout has -0.25.  Conflict detection and prefix gating use direction,
    # so equal-magnitude directional conflicts receive equal correction.
    torch.testing.assert_close(
        extras["oal_group_outcome_advantage"],
        torch.tensor([0.75, -0.25, -0.25, -0.25]),
    )
    torch.testing.assert_close(
        extras["oal_conflict_score"][:, 1],
        torch.tanh(torch.ones(4)),
    )
    torch.testing.assert_close(
        extras["oal_outcome_correction_weight"][:, 1],
        extras["oal_outcome_correction_weight"][0, 1].expand(4),
    )
    assert advantages[0, 1].item() > raw_opd[0, 1].item()
    assert torch.all(advantages[1:, 1] < raw_opd[1:, 1])


def test_full_pov_zero_conflict_is_exact_and_padding_is_finite():
    raw_opd = torch.tensor(
        [[1.0, 2.0, float("nan"), float("inf")], [-1.0, float("nan"), -2.0, float("-inf")]]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    logit_delta = torch.tensor(
        [[1.0, 2.0, float("nan"), float("inf")], [-1.0, float("nan"), -2.0, float("-inf")]]
    )
    teacher_log_probs = torch.tensor(
        [[0.0, -0.69314718056, float("nan"), float("inf")],
         [0.0, float("nan"), -0.69314718056, float("-inf")]]
    )
    teacher_entropy = torch.zeros_like(raw_opd)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(window_size=1),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=logit_delta,
    )

    valid = mask.bool()
    assert torch.all(extras["oal_conflict_score"][valid] == 0)
    assert torch.equal(advantages[valid], raw_opd[valid])
    assert torch.all(advantages[~valid] == 0)
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(extras["oal_mass_renorm_scale"], torch.ones(2))


def test_full_pov_fp16_joint_gate_remains_finite():
    raw_opd = torch.tensor([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]], dtype=torch.float16)
    mask = torch.ones_like(raw_opd)
    # Very weak support and tiny conflict evidence must remain finite.
    teacher_log_probs = torch.tensor([[0.0, -20.0, -20.0]], dtype=torch.float16).expand_as(raw_opd)
    logit_delta = torch.tensor(
        [[1.0, -1e-6, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float16
    )

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=mask,
        config=_pov_config(window_size=1),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0], dtype=torch.float16),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=torch.zeros_like(raw_opd),
        logit_delta_scores=logit_delta,
    )

    prefix = extras["pt_oal_prefix_weights"][0, 1]
    conflict = extras["oal_conflict_score"][0, 1]
    expected_keep = 1.0 - _harmonic_joint(prefix, conflict)
    torch.testing.assert_close(extras["oal_keep_mask"][0, 1], expected_keep)
    assert torch.isfinite(advantages).all()


def test_full_pov_does_not_couple_unrelated_prompt_groups():
    raw_prompt = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5],
            [1.0, -1.0, 0.5, -0.5],
        ]
    )
    mask_prompt = torch.ones_like(raw_prompt)

    prompt_advantages, _, _ = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_prompt,
        response_mask=mask_prompt,
        config=_pov_config(prefix=False),
        index=np.array(["prompt-a", "prompt-a"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        logit_delta_scores=raw_prompt,
    )

    unrelated_prompt = torch.tensor(
        [
            [8.0, -0.1, 4.0, -0.1],
            [8.0, -0.1, 4.0, -0.1],
        ]
    )
    combined_raw = torch.cat((raw_prompt, unrelated_prompt), dim=0)
    combined_advantages, _, _ = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=combined_raw,
        response_mask=torch.ones_like(combined_raw),
        config=_pov_config(prefix=False),
        index=np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        logit_delta_scores=combined_raw,
    )

    torch.testing.assert_close(combined_advantages[:2], prompt_advantages)


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

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
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
    conflict = extras["oal_conflict_score"] > 0

    torch.testing.assert_close(prefix[:, 2:], torch.full((2, 2), 0.5), atol=1e-5, rtol=1e-5)
    # Prefix validity may strengthen an existing outcome correction, but it
    # must not alter raw OPD where outcome and OPD already agree.
    torch.testing.assert_close(keep[~conflict], torch.ones_like(keep[~conflict]))
    expected_alpha = _harmonic_joint(prefix, extras["oal_conflict_score"])
    expected_keep = (1.0 - expected_alpha) * mask
    torch.testing.assert_close(keep[conflict], expected_keep[conflict])
    torch.testing.assert_close(advantages[~conflict], raw_opd[~conflict])
    assert keep[0, 2].item() == 1.0  # correct rollout, positive OPD: aligned
    assert keep[1, 3].item() == 1.0  # wrong rollout, negative OPD: aligned
    assert keep[0, 3].item() < 1.0   # correct rollout, negative OPD: conflict
    assert keep[1, 2].item() < 1.0   # wrong rollout, positive OPD: conflict
    expected_target = (
        extras["oal_group_outcome_advantage"]
        * extras["oal_outcome_target_scale"]
    ).unsqueeze(-1).expand_as(raw_opd)
    expected_advantages = raw_opd * expected_keep + expected_target * expected_alpha
    torch.testing.assert_close(advantages, expected_advantages)
    # At equal conflict evidence, weaker prefix support increases the joint
    # correction and therefore decreases the OPD coefficient continuously.
    assert keep[0, 3].item() < keep[0, 1].item()
    assert keep[1, 2].item() < keep[1, 0].item()


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
    conflict = extras["oal_conflict_score"] > 0
    preserved = candidate_mask & ~conflict
    torch.testing.assert_close(advantages[preserved], dense_valid[preserved])

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


def test_topk_full_pov_broadcasts_prefix_and_masks_invalid_candidates():
    candidate_mask = torch.tensor(
        [
            [[1, 0], [1, 0]],
            [[1, 0], [1, 0]],
        ],
        dtype=torch.bool,
    )
    raw_opd = torch.tensor(
        [
            [[1.0, float("nan")], [-1.0, float("inf")]],
            [[-1.0, float("-inf")], [1.0, float("nan")]],
        ]
    )
    teacher_log_probs = torch.tensor([[0.0, -0.69314718056]]).expand(2, -1)
    teacher_entropy = torch.zeros_like(teacher_log_probs)

    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=torch.ones(2, 2),
        config=_pov_config(window_size=1),
        index=np.array(["prompt", "prompt"], dtype=object),
        true_reward_score=torch.tensor([1.0, 0.0]),
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
        candidate_valid_mask=candidate_mask,
    )

    prefix = extras["pt_oal_prefix_weights"].unsqueeze(-1)
    conflict = extras["oal_conflict_score"]
    expected_alpha = _harmonic_joint(prefix, conflict)
    expected_keep = (1.0 - expected_alpha) * candidate_mask

    torch.testing.assert_close(extras["oal_keep_mask"], expected_keep)
    assert torch.isfinite(advantages).all()
    assert torch.all(advantages[~candidate_mask] == 0)


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
