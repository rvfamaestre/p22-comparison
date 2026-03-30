from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


STRING_STABILITY_METRIC = "max_downstream_speed_error_amplification"
STRING_STABILITY_LABEL = "Maximum downstream speed-error amplification"
STRING_STABILITY_THRESHOLD = 1.0
STRING_STABILITY_BASELINE_WINDOW_STEPS = 20
STRING_STABILITY_MIN_SOURCE_PEAK = 1e-3


def string_stability_report_metadata() -> dict[str, Any]:
    """Describe the canonical string-stability experiment and metric."""
    return {
        "metric": STRING_STABILITY_METRIC,
        "label": STRING_STABILITY_LABEL,
        "threshold": float(STRING_STABILITY_THRESHOLD),
        "baseline_window_steps": int(STRING_STABILITY_BASELINE_WINDOW_STEPS),
        "min_source_peak_speed_error": float(STRING_STABILITY_MIN_SOURCE_PEAK),
        "definition": (
            "Apply the configured one-shot speed perturbation to the designated "
            "vehicle, measure each vehicle's peak absolute speed error after the "
            "perturbation relative to its mean pre-perturbation speed, and report "
            "the maximum downstream follower peak divided by the perturbed "
            "vehicle's peak."
        ),
        "interpretation": (
            "Values below or equal to 1.0 indicate that the disturbance does not "
            "grow downstream. Values above 1.0 indicate amplification and thus "
            "string instability under this controlled perturbation protocol."
        ),
        "validity_conditions": [
            "The perturbation scenario must be enabled and occur within the episode horizon.",
            "A non-empty pre-perturbation baseline window must exist.",
            "The perturbed vehicle must exhibit a measurable peak speed error.",
            "No collision clamps may occur in the analyzed rollout.",
        ],
    }


def downstream_vehicle_order(
    sorted_vehicle_ids: Sequence[int],
    perturb_vehicle_id: int,
) -> list[int]:
    """Return the perturbed vehicle followed by its downstream followers."""
    ordered = [int(vehicle_id) for vehicle_id in sorted_vehicle_ids]
    if not ordered:
        raise ValueError("sorted_vehicle_ids must contain at least one vehicle.")
    if int(perturb_vehicle_id) not in ordered:
        raise ValueError(
            f"Perturbed vehicle {perturb_vehicle_id} is not present in the snapshot."
        )

    start = ordered.index(int(perturb_vehicle_id))
    n = len(ordered)
    return [ordered[(start - offset) % n] for offset in range(n)]


def downstream_vehicle_order_from_follower_map(
    follower_map: Sequence[int],
    perturb_vehicle_id: int,
) -> list[int]:
    """Build the downstream order from a vehicle->follower lookup."""
    followers = [int(item) for item in follower_map]
    if not followers:
        raise ValueError("follower_map must contain at least one vehicle.")

    order = [int(perturb_vehicle_id)]
    current = int(perturb_vehicle_id)
    seen = {current}
    for _ in range(len(followers) - 1):
        current = followers[current]
        if current in seen:
            break
        order.append(current)
        seen.add(current)
    return order


def _default_result(
    *,
    applicable: bool,
    threshold: float,
    baseline_window_steps: int,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "string_stability_metric": STRING_STABILITY_METRIC,
        "string_stability_threshold": float(threshold),
        "string_stability_baseline_window_steps": int(baseline_window_steps),
        "string_stability_applicable": bool(applicable),
        "string_stability_metric_valid": None if not applicable else False,
        "string_stability_is_stable": None if not applicable else False,
        "string_stability_value": float("nan"),
        "string_stability_max_pairwise_speed_gain": float("nan"),
        "string_stability_max_downstream_spacing_error_amplification": float("nan"),
        "string_stability_source_peak_speed_error": float("nan"),
        "string_stability_source_peak_spacing_error": float("nan"),
        "string_stability_failure_reason": failure_reason,
    }


def compute_string_stability_from_ordered_series(
    speed_series: np.ndarray,
    *,
    perturbation_step: int,
    valid_base: bool,
    applicable: bool = True,
    gap_series: np.ndarray | None = None,
    baseline_window_steps: int = STRING_STABILITY_BASELINE_WINDOW_STEPS,
    threshold: float = STRING_STABILITY_THRESHOLD,
    min_source_peak: float = STRING_STABILITY_MIN_SOURCE_PEAK,
) -> dict[str, Any]:
    """Compute amplification metrics from ordered per-vehicle traces.

    The input must already be ordered so column 0 is the perturbed vehicle and
    columns 1..N-1 are its downstream followers in order.
    """
    if not applicable:
        return _default_result(
            applicable=False,
            threshold=threshold,
            baseline_window_steps=baseline_window_steps,
            failure_reason="perturbation_disabled",
        )

    speed = np.asarray(speed_series, dtype=float)
    if speed.ndim != 2 or speed.shape[1] == 0:
        raise ValueError("speed_series must have shape (num_steps, num_vehicles).")
    if perturbation_step < 0:
        raise ValueError("perturbation_step must be non-negative.")

    result = _default_result(
        applicable=True,
        threshold=threshold,
        baseline_window_steps=baseline_window_steps,
        failure_reason="",
    )
    if not valid_base:
        result["string_stability_failure_reason"] = "collision_clamp_invalidates_metric"
        return result

    if perturbation_step >= speed.shape[0]:
        result["string_stability_failure_reason"] = "perturbation_after_horizon"
        return result

    pre_start = max(0, perturbation_step - baseline_window_steps)
    pre_slice = speed[pre_start:perturbation_step]
    post_slice = speed[perturbation_step:]
    if pre_slice.shape[0] == 0:
        result["string_stability_failure_reason"] = "missing_pre_perturbation_window"
        return result
    if post_slice.shape[0] == 0:
        result["string_stability_failure_reason"] = "missing_post_perturbation_window"
        return result

    baseline_speed = pre_slice.mean(axis=0)
    speed_peak_errors = np.max(np.abs(post_slice - baseline_speed), axis=0)
    source_peak_speed = float(speed_peak_errors[0])
    result["string_stability_source_peak_speed_error"] = source_peak_speed
    if source_peak_speed < float(min_source_peak):
        result["string_stability_failure_reason"] = "perturbed_vehicle_peak_too_small"
        return result

    downstream_speed_gains = (
        speed_peak_errors[1:] / source_peak_speed if speed_peak_errors.size > 1 else np.array([])
    )
    pairwise_speed_gains = (
        speed_peak_errors[1:] / np.maximum(speed_peak_errors[:-1], float(min_source_peak))
        if speed_peak_errors.size > 1
        else np.array([])
    )
    max_downstream_speed_gain = (
        float(np.max(downstream_speed_gains)) if downstream_speed_gains.size else 0.0
    )
    max_pairwise_speed_gain = (
        float(np.max(pairwise_speed_gains)) if pairwise_speed_gains.size else 0.0
    )

    spacing_gain = float("nan")
    if gap_series is not None:
        gaps = np.asarray(gap_series, dtype=float)
        if gaps.shape != speed.shape:
            raise ValueError("gap_series must match speed_series shape.")
        baseline_gap = gaps[pre_start:perturbation_step].mean(axis=0)
        gap_peak_errors = np.max(np.abs(gaps[perturbation_step:] - baseline_gap), axis=0)
        source_peak_gap = float(gap_peak_errors[0])
        result["string_stability_source_peak_spacing_error"] = source_peak_gap
        if source_peak_gap >= float(min_source_peak):
            spacing_gains = (
                gap_peak_errors[1:] / source_peak_gap
                if gap_peak_errors.size > 1
                else np.array([])
            )
            spacing_gain = (
                float(np.max(spacing_gains)) if spacing_gains.size else 0.0
            )

    result.update(
        {
            "string_stability_metric_valid": True,
            "string_stability_is_stable": bool(
                max_downstream_speed_gain <= float(threshold)
            ),
            "string_stability_value": max_downstream_speed_gain,
            "string_stability_max_pairwise_speed_gain": max_pairwise_speed_gain,
            "string_stability_max_downstream_spacing_error_amplification": spacing_gain,
        }
    )
    return result


def compute_string_stability_from_traces(
    traces: pd.DataFrame,
    *,
    perturb_vehicle_id: int,
    perturbation_time: float,
    valid_base: bool,
    applicable: bool = True,
    baseline_window_steps: int = STRING_STABILITY_BASELINE_WINDOW_STEPS,
    threshold: float = STRING_STABILITY_THRESHOLD,
    min_source_peak: float = STRING_STABILITY_MIN_SOURCE_PEAK,
) -> dict[str, Any]:
    """Compute the canonical string-stability metric from trace logs."""
    if not applicable:
        return _default_result(
            applicable=False,
            threshold=threshold,
            baseline_window_steps=baseline_window_steps,
            failure_reason="perturbation_disabled",
        )

    required = {"time", "vehicle_id", "v"}
    missing = required - set(traces.columns)
    if missing:
        raise KeyError(f"Missing required trace columns for string stability: {missing}")

    times = np.sort(traces["time"].astype(float).unique())
    perturbation_step = int(np.searchsorted(times, float(perturbation_time), side="left"))
    if perturbation_step >= len(times):
        return _default_result(
            applicable=True,
            threshold=threshold,
            baseline_window_steps=baseline_window_steps,
            failure_reason="perturbation_after_horizon",
        )

    perturbation_time_actual = float(times[perturbation_step])
    snapshot = traces[traces["time"].astype(float) == perturbation_time_actual].copy()
    if snapshot.empty:
        return _default_result(
            applicable=True,
            threshold=threshold,
            baseline_window_steps=baseline_window_steps,
            failure_reason="missing_perturbation_snapshot",
        )

    if "x" in snapshot.columns:
        sorted_vehicle_ids = (
            snapshot.sort_values("x")["vehicle_id"].astype(int).tolist()
        )
    else:
        sorted_vehicle_ids = sorted(snapshot["vehicle_id"].astype(int).tolist())
    ordered_vehicle_ids = downstream_vehicle_order(
        sorted_vehicle_ids,
        int(perturb_vehicle_id),
    )

    speed_frame = (
        traces.pivot_table(index="time", columns="vehicle_id", values="v", aggfunc="mean")
        .sort_index()
        .reindex(columns=ordered_vehicle_ids)
    )
    gap_frame = None
    if "gap_s" in traces.columns:
        gap_frame = (
            traces.pivot_table(
                index="time",
                columns="vehicle_id",
                values="gap_s",
                aggfunc="mean",
            )
            .sort_index()
            .reindex(columns=ordered_vehicle_ids)
        )

    return compute_string_stability_from_ordered_series(
        speed_frame.to_numpy(dtype=float),
        perturbation_step=perturbation_step,
        valid_base=valid_base,
        applicable=applicable,
        gap_series=gap_frame.to_numpy(dtype=float) if gap_frame is not None else None,
        baseline_window_steps=baseline_window_steps,
        threshold=threshold,
        min_source_peak=min_source_peak,
    )
