from recipe.repro.pov_audit_utils import cluster_bootstrap_mean_ci, match_prefix_controls


def _trajectory(identifier, *, prompt="p", outcome=1, length=100, triggered=False, horizon=40):
    windows = []
    for index, start in enumerate(range(0, length, 20)):
        windows.append(
            {
                "window_index": index,
                "start": start,
                "end": min(start + 20, length),
                "cosine": 0.5 - 0.2 * index,
                "conflict": index >= 3,
            }
        )
    return {
        "trajectory_id": identifier,
        "prompt_id": prompt,
        "outcome_class": outcome,
        "response_length": length,
        "triggered": triggered,
        "horizon_token": horizon,
        "windows": windows,
    }


def test_match_prefix_controls_enforces_prompt_outcome_and_no_replacement():
    trajectories = [
        _trajectory("t1", triggered=True, length=100),
        _trajectory("t2", triggered=True, length=104),
        _trajectory("c1", length=102),
        _trajectory("wrong-prompt", prompt="q", length=100),
        _trajectory("wrong-outcome", outcome=0, length=100),
    ]
    pairs = match_prefix_controls(trajectories, length_caliper=0.10)
    assert len(pairs) == 1
    assert pairs[0]["control_trajectory_id"] == "c1"
    assert pairs[0]["prompt_id"] == "p"
    assert pairs[0]["outcome_class"] == 1


def test_trigger_window_is_in_post_period():
    pairs = match_prefix_controls(
        [_trajectory("t", triggered=True, horizon=40), _trajectory("c")],
        length_caliper=0.0,
    )
    assert len(pairs) == 1
    assert pairs[0]["triggered_pre"]["window_count"] == 2
    assert pairs[0]["triggered_post"]["window_count"] == 3


def test_cluster_bootstrap_reports_cluster_count():
    result = cluster_bootstrap_mean_ci({"p1": [1.0, 2.0], "p2": [3.0]}, samples=100, seed=7)
    assert result["mean"] == 2.0
    assert result["clusters"] == 2
    assert result["ci_low"] is not None
    assert result["ci_high"] is not None

