from collections import deque

import pytest

from src.visualization.live_hud import compute_steps_per_tick, trim_history_window


def test_compute_steps_per_tick_matches_x1_budget_over_five_ticks():
    accumulator = 0.0
    total_steps = 0

    for _ in range(5):
        steps, accumulator = compute_steps_per_tick(
            dt_seconds=0.1,
            redraw_interval_ms=20,
            speed_multiplier=1.0,
            accumulator=accumulator,
        )
        total_steps += steps

    assert total_steps == 1
    assert accumulator == pytest.approx(0.0)


def test_compute_steps_per_tick_scales_with_multiplier():
    accumulator = 0.0
    total_steps = 0

    for _ in range(5):
        steps, accumulator = compute_steps_per_tick(
            dt_seconds=0.1,
            redraw_interval_ms=20,
            speed_multiplier=4.0,
            accumulator=accumulator,
        )
        total_steps += steps

    assert total_steps == 4
    assert accumulator == pytest.approx(0.0)


def test_compute_steps_per_tick_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_steps_per_tick(0.0, 20, 1.0, 0.0)

    with pytest.raises(ValueError):
        compute_steps_per_tick(0.1, 0, 1.0, 0.0)

    with pytest.raises(ValueError):
        compute_steps_per_tick(0.1, 20, 0.0, 0.0)


def test_trim_history_window_keeps_only_recent_window():
    history = {
        "t": deque([0.0, 30.0, 70.0, 120.0]),
        "mean_speed": deque([1.0, 2.0, 3.0, 4.0]),
        "min_gap": deque([5.0, 6.0, 7.0, 8.0]),
    }

    trim_history_window(history, window_seconds=100.0)

    assert list(history["t"]) == [30.0, 70.0, 120.0]
    assert list(history["mean_speed"]) == [2.0, 3.0, 4.0]
    assert list(history["min_gap"]) == [6.0, 7.0, 8.0]


def test_trim_history_window_rejects_invalid_window():
    with pytest.raises(ValueError):
        trim_history_window({"t": deque(), "mean_speed": deque(), "min_gap": deque()}, 0.0)
