from __future__ import annotations

from rank_gpu_checkpoints import rank_rows


def test_rank_rows_prefers_safer_checkpoint_by_default() -> None:
    rows = [
        {
            "checkpoint": "fast_but_unsafe",
            "primary_objective_metric": "mean_speed_last100",
            "string_stability_metric": "max_downstream_speed_error_amplification",
            "rl_primary_objective_avg": 5.5,
            "adaptive_primary_objective_avg": 4.5,
            "primary_objective_delta_vs_adaptive": 1.0,
            "rl_speed_avg": 5.5,
            "adaptive_speed_avg": 4.5,
            "speed_delta_vs_adaptive": 1.0,
            "rl_reward_avg": -600.0,
            "rl_speed_var_avg": 0.8,
            "rl_min_gap_min": 0.9,
            "rl_safety_pass_rate_min": 0.05,
            "rl_string_stability_avg": 0.95,
            "rl_string_stability_valid_rate_min": 0.5,
            "rl_string_stability_stable_rate_min": 0.0,
        },
        {
            "checkpoint": "slower_but_safer",
            "primary_objective_metric": "mean_speed_last100",
            "string_stability_metric": "max_downstream_speed_error_amplification",
            "rl_primary_objective_avg": 4.8,
            "adaptive_primary_objective_avg": 4.5,
            "primary_objective_delta_vs_adaptive": 0.3,
            "rl_speed_avg": 4.8,
            "adaptive_speed_avg": 4.5,
            "speed_delta_vs_adaptive": 0.3,
            "rl_reward_avg": -590.0,
            "rl_speed_var_avg": 0.5,
            "rl_min_gap_min": 1.6,
            "rl_safety_pass_rate_min": 0.5,
            "rl_string_stability_avg": 0.72,
            "rl_string_stability_valid_rate_min": 0.8,
            "rl_string_stability_stable_rate_min": 0.0,
        },
    ]

    ranked = rank_rows(rows, policy="safety_first")

    assert ranked[0]["checkpoint"] == "slower_but_safer"


def test_rank_rows_can_still_reproduce_objective_first_ordering() -> None:
    rows = [
        {
            "checkpoint": "fast_but_unsafe",
            "primary_objective_metric": "mean_speed_last100",
            "string_stability_metric": "max_downstream_speed_error_amplification",
            "rl_primary_objective_avg": 5.5,
            "adaptive_primary_objective_avg": 4.5,
            "primary_objective_delta_vs_adaptive": 1.0,
            "rl_speed_avg": 5.5,
            "adaptive_speed_avg": 4.5,
            "speed_delta_vs_adaptive": 1.0,
            "rl_reward_avg": -600.0,
            "rl_speed_var_avg": 0.8,
            "rl_min_gap_min": 0.9,
            "rl_safety_pass_rate_min": 0.05,
            "rl_string_stability_avg": 0.95,
            "rl_string_stability_valid_rate_min": 0.5,
            "rl_string_stability_stable_rate_min": 0.0,
        },
        {
            "checkpoint": "slower_but_safer",
            "primary_objective_metric": "mean_speed_last100",
            "string_stability_metric": "max_downstream_speed_error_amplification",
            "rl_primary_objective_avg": 4.8,
            "adaptive_primary_objective_avg": 4.5,
            "primary_objective_delta_vs_adaptive": 0.3,
            "rl_speed_avg": 4.8,
            "adaptive_speed_avg": 4.5,
            "speed_delta_vs_adaptive": 0.3,
            "rl_reward_avg": -590.0,
            "rl_speed_var_avg": 0.5,
            "rl_min_gap_min": 1.6,
            "rl_safety_pass_rate_min": 0.5,
            "rl_string_stability_avg": 0.72,
            "rl_string_stability_valid_rate_min": 0.8,
            "rl_string_stability_stable_rate_min": 0.0,
        },
    ]

    ranked = rank_rows(rows, policy="objective_first")

    assert ranked[0]["checkpoint"] == "fast_but_unsafe"
