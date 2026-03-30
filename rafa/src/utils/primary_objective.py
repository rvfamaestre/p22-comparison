from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence


PRIMARY_OBJECTIVE_METRIC = "mean_speed_last100"
PRIMARY_OBJECTIVE_ROW_KEY = "primary_objective_value"
PRIMARY_OBJECTIVE_LABEL = "Mean fleet speed over the last 100 steps"

# Physical collision-prevention clamp used by the simulator (VecRingRoadEnv.min_gap).
# A run that violates this has had near-contact between bumpers.
COLLISION_CLEARANCE_M: float = 0.3

# Minimum safe following distance from the reward function (RLConfig.s_min).
# This is the threshold below which the gap-safety penalty activates.
# Use this as the default for safety reporting — it is the operationally
# meaningful check: did the controller maintain the intended safe headway?
SAFE_HEADWAY_M: float = 2.0


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


def annotate_with_primary_objective(
    metrics: Mapping[str, Any],
    *,
    min_gap_key: str,
    min_gap_threshold: float = SAFE_HEADWAY_M,
    collision_count_key: str | None = None,
    collision_clamp_count_key: str | None = None,
    string_stability_key: str | None = None,
) -> dict[str, Any]:
    """Add canonical objective and safety-constraint fields to a metrics row."""
    if PRIMARY_OBJECTIVE_METRIC not in metrics:
        raise KeyError(
            f"Missing canonical objective metric '{PRIMARY_OBJECTIVE_METRIC}'."
        )

    row = dict(metrics)
    row["primary_objective_metric"] = PRIMARY_OBJECTIVE_METRIC
    row[PRIMARY_OBJECTIVE_ROW_KEY] = float(row[PRIMARY_OBJECTIVE_METRIC])

    checks: list[bool] = []

    min_gap = float(row[min_gap_key])
    row["safety_min_gap_threshold"] = float(min_gap_threshold)
    row["safety_min_gap_ok"] = min_gap >= float(min_gap_threshold)
    checks.append(bool(row["safety_min_gap_ok"]))

    if collision_count_key and collision_count_key in row:
        row["safety_collision_free"] = int(row[collision_count_key]) == 0
        checks.append(bool(row["safety_collision_free"]))

    if collision_clamp_count_key and collision_clamp_count_key in row:
        row["safety_no_collision_clamps"] = (
            int(row[collision_clamp_count_key]) == 0
        )
        checks.append(bool(row["safety_no_collision_clamps"]))

    if string_stability_key and string_stability_key in row:
        row["safety_string_stability_ok"] = coerce_bool(row[string_stability_key])
        checks.append(bool(row["safety_string_stability_ok"]))

    row["safety_constraint_satisfied"] = all(checks) if checks else True
    return row
