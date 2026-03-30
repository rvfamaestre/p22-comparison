from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from run_experiments import compute_metrics, write_summary_by_human_rate
from src.utils.primary_objective import PRIMARY_OBJECTIVE_METRIC, SAFE_HEADWAY_M
from src.utils.string_stability_metrics import STRING_STABILITY_METRIC


def build_traces(
    *,
    num_steps: int = 120,
    early_speed: float = 1.0,
    late_speed: float = 3.0,
    min_gap: float = 2.5,
) -> pd.DataFrame:
    """Build a minimal per-vehicle trace DataFrame for testing.

    Default min_gap=2.5 m is intentionally above SAFE_HEADWAY_M (2.0 m) so that
    the happy-path test asserts safety_min_gap_ok is True.
    """
    rows = []
    for step in range(num_steps):
        speed = early_speed if step < num_steps - 100 else late_speed
        for vehicle_id in (0, 1):
            rows.append(
                {
                    "time": float(step),
                    "vehicle_id": vehicle_id,
                    "vehicle_type": "HDV",
                    "v": speed,
                    "a": 0.0,
                    "gap_s": min_gap,
                }
            )
    return pd.DataFrame(rows)


def build_perturbation_traces(
    *,
    num_steps: int = 120,
    perturbation_step: int = 20,
    baseline_speed: float = 3.0,
    source_drop: float = 1.0,
    follower_drop: float = 0.8,
    min_gap: float = 2.5,
) -> pd.DataFrame:
    rows = []
    for step in range(num_steps):
        if step < perturbation_step:
            speeds = {0: baseline_speed, 1: baseline_speed}
        else:
            speeds = {
                0: baseline_speed - source_drop,
                1: baseline_speed - follower_drop,
            }
        for vehicle_id in (0, 1):
            rows.append(
                {
                    "time": float(step),
                    "vehicle_id": vehicle_id,
                    "vehicle_type": "HDV",
                    "x": float(vehicle_id * 10.0),
                    "v": float(speeds[vehicle_id]),
                    "a": 0.0,
                    "gap_s": min_gap,
                }
            )
    return pd.DataFrame(rows)


def test_compute_metrics_exposes_primary_objective_and_safety() -> None:
    traces = build_perturbation_traces()
    metrics = compute_metrics(
        traces,
        dt=0.1,
        mode="baseline",
        metadata={
            "collision_count": 0,
            "collision_clamp_count": 0,
            "string_stability_valid": True,
            "perturbation_enabled": True,
            "perturbation_vehicle": 0,
            "perturbation_time": 20.0,
            "steps_completed": 120,
        },
    )

    assert metrics["mean_speed"] == pytest.approx((20 * 3.0 + 100 * 2.1) / 120.0)
    assert metrics["mean_speed_last100"] == pytest.approx(2.1)
    assert metrics["primary_objective_metric"] == PRIMARY_OBJECTIVE_METRIC
    assert metrics["primary_objective_value"] == pytest.approx(2.1)
    assert metrics["safety_min_gap_threshold"] == pytest.approx(SAFE_HEADWAY_M)
    assert metrics["safety_min_gap_ok"] is True
    assert metrics["safety_collision_free"] is True
    assert metrics["safety_no_collision_clamps"] is True
    assert metrics["safety_constraint_satisfied"] is True
    assert metrics["string_stability_metric"] == STRING_STABILITY_METRIC
    assert metrics["string_stability_valid"] is True
    assert metrics["string_stability_metric_valid"] is True
    assert metrics["string_stability_is_stable"] is True
    assert metrics["string_stability_value"] == pytest.approx(0.8)


def test_compute_metrics_marks_constraint_failures() -> None:
    traces = build_perturbation_traces(min_gap=0.2)
    metrics = compute_metrics(
        traces,
        dt=0.1,
        mode="baseline",
        metadata={
            "collision_count": 1,
            "collision_clamp_count": 2,
            "string_stability_valid": False,
            "perturbation_enabled": True,
            "perturbation_vehicle": 0,
            "perturbation_time": 20.0,
            "steps_completed": 120,
        },
    )

    assert metrics["primary_objective_value"] == pytest.approx(2.1)
    assert metrics["safety_min_gap_ok"] is False
    assert metrics["safety_collision_free"] is False
    assert metrics["safety_no_collision_clamps"] is False
    assert metrics["safety_constraint_satisfied"] is False
    assert metrics["string_stability_valid"] is False
    assert metrics["string_stability_metric_valid"] is False
    assert pd.isna(metrics["string_stability_value"])


def test_compute_metrics_reports_instability_without_redefining_safety() -> None:
    traces = build_perturbation_traces(follower_drop=1.3)
    metrics = compute_metrics(
        traces,
        dt=0.1,
        mode="baseline",
        metadata={
            "collision_count": 0,
            "collision_clamp_count": 0,
            "string_stability_valid": True,
            "perturbation_enabled": True,
            "perturbation_vehicle": 0,
            "perturbation_time": 20.0,
            "steps_completed": 120,
        },
    )

    assert metrics["string_stability_metric_valid"] is True
    assert metrics["string_stability_is_stable"] is False
    assert metrics["string_stability_value"] == pytest.approx(1.3)
    assert metrics["safety_constraint_satisfied"] is True


def test_compute_metrics_headway_boundary() -> None:
    """Verify the exact threshold: gap==SAFE_HEADWAY_M passes, gap just below fails."""
    traces_pass = build_traces(min_gap=SAFE_HEADWAY_M)
    metrics_pass = compute_metrics(traces_pass, dt=0.1, mode="baseline")
    assert metrics_pass["safety_min_gap_ok"] is True

    traces_fail = build_traces(min_gap=SAFE_HEADWAY_M - 0.01)
    metrics_fail = compute_metrics(traces_fail, dt=0.1, mode="baseline")
    assert metrics_fail["safety_min_gap_ok"] is False


def test_summary_by_human_rate_surfaces_primary_objective_columns(
    tmp_path: Path,
) -> None:
    # Row 0: min_gap=2.5 > SAFE_HEADWAY_M  → safety_min_gap_ok=True
    # Row 1: min_gap=0.2 < SAFE_HEADWAY_M  → safety_min_gap_ok=False
    summary_runs = pd.DataFrame(
        [
            {
                "mode": "sac",
                "human_rate": 0.75,
                "cav_share": 0.25,
                "speed_var_global": 0.8,
                "speed_std_time_mean": 0.6,
                "oscillation_amplitude": 1.2,
                "min_gap": 2.5,
                "rms_acc": 0.2,
                "rms_jerk": 0.1,
                "mean_speed": 2.6,
                "mean_speed_last100": 3.0,
                "primary_objective_value": 3.0,
                "string_stability_metric": STRING_STABILITY_METRIC,
                "string_stability_threshold": 1.0,
                "string_stability_value": 0.8,
                "string_stability_valid": True,
                "string_stability_metric_valid": True,
                "string_stability_is_stable": True,
                "safety_min_gap_threshold": SAFE_HEADWAY_M,
                "safety_min_gap_ok": True,
                "safety_constraint_satisfied": True,
            },
            {
                "mode": "sac",
                "human_rate": 0.75,
                "cav_share": 0.25,
                "speed_var_global": 1.0,
                "speed_std_time_mean": 0.8,
                "oscillation_amplitude": 1.4,
                "min_gap": 0.2,
                "rms_acc": 0.3,
                "rms_jerk": 0.2,
                "mean_speed": 2.8,
                "mean_speed_last100": 3.4,
                "primary_objective_value": 3.4,
                "string_stability_metric": STRING_STABILITY_METRIC,
                "string_stability_threshold": 1.0,
                "string_stability_value": 1.2,
                "string_stability_valid": True,
                "string_stability_metric_valid": True,
                "string_stability_is_stable": False,
                "safety_min_gap_threshold": SAFE_HEADWAY_M,
                "safety_min_gap_ok": False,
                "safety_constraint_satisfied": False,
            },
        ]
    )

    path = write_summary_by_human_rate(summary_runs, tmp_path)
    grouped = pd.read_csv(path)

    assert list(grouped["primary_objective_metric"]) == [PRIMARY_OBJECTIVE_METRIC]
    assert grouped.loc[0, "primary_objective_mean"] == pytest.approx(3.2)
    assert grouped.loc[0, "primary_objective_std"] == pytest.approx(
        pd.Series([3.0, 3.4]).std()
    )
    assert grouped.loc[0, "string_stability_metric"] == STRING_STABILITY_METRIC
    assert grouped.loc[0, "string_stability_value_mean"] == pytest.approx(1.0)
    assert grouped.loc[0, "string_stability_metric_valid_rate"] == pytest.approx(1.0)
    assert grouped.loc[0, "string_stability_is_stable_rate"] == pytest.approx(0.5)
    assert grouped.loc[0, "safety_constraint_satisfied_rate"] == pytest.approx(0.5)
    assert grouped.loc[0, "safety_min_gap_ok_rate"] == pytest.approx(0.5)
    assert grouped.loc[0, "safety_min_gap_threshold"] == pytest.approx(SAFE_HEADWAY_M)
