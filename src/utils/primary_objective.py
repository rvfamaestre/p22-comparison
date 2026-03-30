from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence


PRIMARY_OBJECTIVE_METRIC = "string_stability_value"
PRIMARY_OBJECTIVE_ROW_KEY = "primary_objective_value"
PRIMARY_OBJECTIVE_LABEL = "String-stability amplification ratio under the configured perturbation"

# Physical collision-prevention clamp used by the simulator (VecRingRoadEnv.min_gap).
# A run that violates this has had near-contact between bumpers and is not admissible.
COLLISION_CLEARANCE_M: float = 0.3

# Minimum safe following distance from the reward function (RLConfig.s_min).
# This remains a useful safety-quality target, but it is not the hard validity gate.
SAFE_HEADWAY_M: float = 2.0

TRAINING_OBJECTIVE_METRIC = "training_objective"
TRAINING_FALLBACK_METRIC = "speed_std_time_mean"


def mean_last_window(values: Sequence[float], window: int = 100) -> float:
    """Return the mean of the last `window` values, or all values if shorter."""
    if window <= 0:
        raise ValueError("window must be positive.")
    if not values:
        return 0.0
    tail = values[-min(window, len(values)) :]
    return float(fmean(float(value) for value in tail))


def coerce_bool(value: Any) -> bool:
    """Convert CSV-/JSON-style booleans to Python bools."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def compute_training_objective(
    metrics: Mapping[str, Any],
    *,
    min_gap_key: str,
    collision_count_key: str | None = None,
    collision_clamp_count_key: str | None = None,
    string_stability_value_key: str = PRIMARY_OBJECTIVE_METRIC,
    string_stability_valid_key: str = "string_stability_metric_valid",
    fallback_metric_key: str = TRAINING_FALLBACK_METRIC,
    min_gap_threshold: float = SAFE_HEADWAY_M,
) -> float:
    """Collapse the safety-first hierarchy into one scalar for early stopping.

    Order of preference:
    1. avoid collisions / collision clamps;
    2. keep a reasonable headway margin;
    3. minimize formal string-stability amplification when valid;
    4. fall back to a wave proxy while safety is still unresolved.
    """
    row = dict(metrics)

    min_gap = _coerce_float(row.get(min_gap_key), COLLISION_CLEARANCE_M)
    clearance_deficit = max(0.0, COLLISION_CLEARANCE_M - min_gap)
    headway_deficit = max(0.0, float(min_gap_threshold) - min_gap)

    hard_violations = 0
    if collision_count_key and collision_count_key in row:
        hard_violations += _coerce_nonnegative_int(row[collision_count_key])
    if collision_clamp_count_key and collision_clamp_count_key in row:
        hard_violations += _coerce_nonnegative_int(row[collision_clamp_count_key])
    if hard_violations == 0 and min_gap < COLLISION_CLEARANCE_M:
        hard_violations = 1

    fallback_value = _coerce_float(row.get(fallback_metric_key), 0.0)
    if not math.isfinite(fallback_value):
        fallback_value = 0.0

    primary_value = _coerce_float(row.get(string_stability_value_key))
    primary_valid = coerce_bool(row.get(string_stability_valid_key, False))
    primary_valid = primary_valid and math.isfinite(primary_value)

    if hard_violations > 0:
        return (
            1_000_000.0
            + 1_000.0 * float(hard_violations)
            + 100.0 * clearance_deficit
            + 10.0 * headway_deficit
            + fallback_value
        )

    if primary_valid:
        return 10.0 * headway_deficit + primary_value

    return 10_000.0 + 10.0 * headway_deficit + fallback_value


def annotate_with_primary_objective(
    metrics: Mapping[str, Any],
    *,
    min_gap_key: str,
    min_gap_threshold: float = SAFE_HEADWAY_M,
    collision_count_key: str | None = None,
    collision_clamp_count_key: str | None = None,
    string_stability_key: str | None = None,
) -> dict[str, Any]:
    """Add canonical objective and safety fields to a metrics row."""
    if PRIMARY_OBJECTIVE_METRIC not in metrics:
        raise KeyError(
            f"Missing canonical objective metric '{PRIMARY_OBJECTIVE_METRIC}'."
        )

    row = dict(metrics)
    primary_value = _coerce_float(row[PRIMARY_OBJECTIVE_METRIC])
    row["primary_objective_metric"] = PRIMARY_OBJECTIVE_METRIC
    row[PRIMARY_OBJECTIVE_ROW_KEY] = primary_value
    row["primary_objective_label"] = PRIMARY_OBJECTIVE_LABEL
    row["primary_objective_available"] = math.isfinite(primary_value)

    min_gap = _coerce_float(row[min_gap_key], COLLISION_CLEARANCE_M)
    row["safety_collision_clearance_threshold"] = float(COLLISION_CLEARANCE_M)
    row["safety_collision_clearance_ok"] = min_gap >= float(COLLISION_CLEARANCE_M)
    row["safety_min_gap_threshold"] = float(min_gap_threshold)
    row["safety_min_gap_ok"] = min_gap >= float(min_gap_threshold)

    hard_checks = [bool(row["safety_collision_clearance_ok"])]

    if collision_count_key and collision_count_key in row:
        row["safety_collision_free"] = _coerce_nonnegative_int(row[collision_count_key]) == 0
        hard_checks.append(bool(row["safety_collision_free"]))

    if collision_clamp_count_key and collision_clamp_count_key in row:
        row["safety_no_collision_clamps"] = (
            _coerce_nonnegative_int(row[collision_clamp_count_key]) == 0
        )
        hard_checks.append(bool(row["safety_no_collision_clamps"]))

    row["safety_hard_ok"] = all(hard_checks)
    row["safety_constraint_satisfied"] = bool(row["safety_hard_ok"])

    if string_stability_key and string_stability_key in row:
        row["objective_string_stability_ok"] = coerce_bool(row[string_stability_key])

    row["primary_objective_eligible"] = (
        bool(row["safety_constraint_satisfied"])
        and bool(row["primary_objective_available"])
    )
    return row
