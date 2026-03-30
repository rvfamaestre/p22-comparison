from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.analysis_plots import (
    generate_all_from_directory,
    generate_comparison_report,
    generate_generalization_report,
)


def test_generate_comparison_report_writes_expected_files(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "mode": mode,
                "cav_share": cav_share,
                "seed": seed,
                "mean_speed": 8.0 + cav_share + seed * 0.05 + mode_offset,
                "mean_speed_last100": 8.4 + cav_share + seed * 0.05 + mode_offset,
                "min_gap": 4.0 + cav_share * 0.8 + mode_offset * 0.2,
                "rms_jerk": 1.5 + seed * 0.03 + mode_offset * 0.1,
                "speed_var_global": 0.8 + (1.0 - cav_share) * 0.2 + mode_offset * 0.05,
                "speed_std_time_mean": 0.6 + mode_offset * 0.04,
                "oscillation_amplitude": 1.0 + cav_share * 0.3,
                "primary_objective_metric": "mean_speed_last100",
                "primary_objective_value": 8.4 + cav_share + seed * 0.05 + mode_offset,
                "string_stability_value": 0.9 + 0.1 * mode_offset + 0.05 * (1.0 - cav_share),
                "safety_constraint_satisfied": True,
            }
            for mode, mode_offset in (("baseline", 0.0), ("adaptive", 0.2), ("sac", 0.35))
            for cav_share in (0.25, 0.75)
            for seed in (0, 1, 2)
        ]
    )
    summary_csv = tmp_path / "summary_runs.csv"
    summary.to_csv(summary_csv, index=False)

    output_dir = tmp_path / "summary_plots"
    saved = generate_comparison_report(summary_csv, output_dir)

    assert saved
    assert (output_dir / "overview_metrics_grid.png").exists()
    assert (output_dir / "overview_metrics_grid.svg").exists()
    assert (output_dir / "tradeoff_speed_vs_variance.png").exists()
    assert (output_dir / "mean_speed_boxplot.svg").exists()
    assert (output_dir / "tradeoff_primary_objective_vs_variance.png").exists()
    assert (output_dir / "primary_objective_boxplot.svg").exists()
    assert (output_dir / "primary_objective_heatmap.png").exists()
    assert (output_dir / "tradeoff_primary_objective_vs_string_stability.png").exists()
    assert (output_dir / "string_stability_boxplot.svg").exists()
    assert (output_dir / "string_stability_heatmap.png").exists()


def test_generate_generalization_report_writes_expected_files(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "mode": mode,
                "human_rate": human_rate,
                "cav_share": 1.0 - human_rate,
                "shuffle_cav_positions": shuffle,
                "perturbation_enabled": perturbation,
                "primary_objective_metric": "mean_speed_last100",
                "primary_objective_mean": 8.0 + human_rate + mode_offset,
                "primary_objective_std": 0.2,
                "min_gap_episode_mean": 2.5 + 0.1 * mode_offset,
                "min_gap_episode_std": 0.05,
                "safety_constraint_satisfied_rate": 0.9 - 0.1 * float(perturbation),
                "string_stability_value_mean": (
                    0.85 + 0.15 * mode_offset + 0.05 * human_rate
                    if perturbation
                    else float("nan")
                ),
                "string_stability_value_std": 0.08 if perturbation else float("nan"),
                "string_stability_metric_valid_rate": 0.95 if perturbation else float("nan"),
            }
            for mode, mode_offset in (("baseline", 0.0), ("adaptive", 0.2), ("rl", 0.35))
            for human_rate in (0.25, 0.75)
            for shuffle in (False, True)
            for perturbation in (False, True)
        ]
    )
    summary_csv = tmp_path / "gpu_eval_summary.csv"
    summary.to_csv(summary_csv, index=False)

    output_dir = tmp_path / "summary_plots"
    saved = generate_generalization_report(summary_csv, output_dir)

    assert saved
    assert (output_dir / "generalization_primary_objective_grid.png").exists()
    assert (output_dir / "generalization_min_gap_grid.svg").exists()
    assert (output_dir / "generalization_safety_rate_grid.png").exists()
    assert (output_dir / "generalization_rl_primary_objective_heatmap.svg").exists()
    assert (output_dir / "generalization_rl_safety_heatmap.png").exists()
    assert (output_dir / "generalization_string_stability_grid.png").exists()
    assert (output_dir / "generalization_string_stability_validity_grid.svg").exists()
    assert (output_dir / "generalization_rl_string_stability_heatmap.png").exists()


def test_generate_comparison_report_accepts_gpu_eval_raw_schema(tmp_path: Path) -> None:
    raw_like = pd.DataFrame(
        [
            {
                "mode": mode,
                "cav_share": cav_share,
                "seed": seed,
                "mean_speed_all": 8.1 + cav_share + seed * 0.05 + mode_offset,
                "mean_speed_last100": 8.4 + cav_share + seed * 0.05 + mode_offset,
                "min_gap_episode": 3.8 + cav_share * 0.7 + mode_offset * 0.15,
                "speed_var_last100": 0.7 + (1.0 - cav_share) * 0.15 + mode_offset * 0.04,
                "primary_objective_metric": "mean_speed_last100",
                "primary_objective_value": 8.4 + cav_share + seed * 0.05 + mode_offset,
                "string_stability_value": 0.9 + 0.08 * mode_offset + 0.04 * (1.0 - cav_share),
                "safety_constraint_satisfied": True,
            }
            for mode, mode_offset in (("baseline", 0.0), ("adaptive", 0.2), ("rl", 0.35))
            for cav_share in (0.25, 0.75)
            for seed in (0, 1, 2)
        ]
    )
    raw_csv = tmp_path / "gpu_eval_raw.csv"
    raw_like.to_csv(raw_csv, index=False)

    output_dir = tmp_path / "summary_plots_overall"
    saved = generate_comparison_report(raw_csv, output_dir)

    assert saved
    assert (output_dir / "overview_metrics_grid.png").exists()
    assert (output_dir / "tradeoff_speed_vs_variance.png").exists()
    assert (output_dir / "mean_speed_boxplot.svg").exists()
    assert (output_dir / "mean_speed_heatmap.png").exists()
    assert (output_dir / "tradeoff_primary_objective_vs_string_stability.png").exists()


def test_generate_all_from_directory_with_training_and_eval_logs(tmp_path: Path) -> None:
    training_log = pd.DataFrame(
        [
            {
                "algorithm": "ppo",
                "step": 1024,
                "update": 1,
                "elapsed_s": 4.0,
                "steps_per_second": 256.0,
                "reward": -2.3,
                "pg_loss": 0.01,
                "v_loss": 1.25,
                "entropy": 1.4,
                "lr": 3.0e-4,
                "w_alpha": 0.3,
            },
            {
                "algorithm": "ppo",
                "step": 2048,
                "update": 2,
                "elapsed_s": 8.5,
                "steps_per_second": 241.0,
                "reward": -1.7,
                "pg_loss": 0.008,
                "v_loss": 0.94,
                "entropy": 1.3,
                "lr": 2.7e-4,
                "w_alpha": 0.28,
            },
        ]
    )
    eval_log = pd.DataFrame(
        [
            {
                "algorithm": "ppo",
                "step": step,
                "elapsed_s": float(step) / 256.0,
                "human_rate": human_rate,
                "primary_objective_metric": "mean_speed_last100",
                "eval_mean_speed": speed,
                "eval_mean_speed_over_human_rates": mean_speed,
            }
            for step, mean_speed, speeds in (
                (1024, 8.2, {0.0: 8.5, 0.5: 7.9}),
                (2048, 8.6, {0.0: 8.9, 0.5: 8.3}),
            )
            for human_rate, speed in speeds.items()
        ]
    )

    training_log.to_csv(tmp_path / "training_log.csv", index=False)
    eval_log.to_csv(tmp_path / "eval_log.csv", index=False)

    saved = generate_all_from_directory(tmp_path)
    plots_dir = tmp_path / "plots"

    assert saved
    assert (plots_dir / "training_reward_curve.png").exists()
    assert (plots_dir / "training_diagnostics_grid.svg").exists()
    assert (plots_dir / "learning_rate_schedule.png").exists()
    assert (plots_dir / "policy_entropy.svg").exists()
    assert (plots_dir / "eval_convergence.png").exists()
