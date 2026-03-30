from __future__ import annotations

from src.utils.early_stopping import EarlyStopMonitor, EarlyStoppingConfig


def test_tracking_starts_only_after_start_step() -> None:
    monitor = EarlyStopMonitor(
        EarlyStoppingConfig(
            enabled=True,
            metric="eval_mean_speed",
            mode="max",
            patience=2,
            min_delta=0.1,
            start_step=100,
            min_checks=2,
        )
    )

    status_prestart = monitor.update(1.0, 50)
    assert status_prestart["tracking_started"] is False
    assert status_prestart["best_metric"] is None
    assert status_prestart["eligible_checks"] == 0
    assert status_prestart["improved"] is False

    status_start = monitor.update(1.2, 100)
    assert status_start["tracking_started"] is True
    assert status_start["best_metric"] == 1.2
    assert status_start["best_step"] == 100
    assert status_start["eligible_checks"] == 1
    assert status_start["improved"] is True


def test_min_checks_count_from_tracking_start() -> None:
    monitor = EarlyStopMonitor(
        EarlyStoppingConfig(
            enabled=True,
            metric="train_reward",
            mode="max",
            patience=2,
            min_delta=0.05,
            start_step=100,
            min_checks=3,
        )
    )

    monitor.update(0.5, 32)
    monitor.update(0.4, 64)

    status_1 = monitor.update(1.0, 100)
    status_2 = monitor.update(0.98, 132)
    status_3 = monitor.update(0.97, 164)

    assert status_1["ready"] is False
    assert status_2["ready"] is False
    assert status_2["checks_since_improvement"] == 0
    assert status_3["ready"] is True
    assert status_3["checks_since_improvement"] == 1


def test_plateau_stopping_uses_post_start_best_only() -> None:
    monitor = EarlyStopMonitor(
        EarlyStoppingConfig(
            enabled=True,
            metric="train_reward",
            mode="max",
            patience=2,
            min_delta=0.01,
            start_step=100,
            min_checks=2,
        )
    )

    monitor.update(10.0, 32)
    monitor.update(9.0, 64)

    first = monitor.update(1.0, 100)
    second = monitor.update(0.995, 132)
    third = monitor.update(0.994, 164)

    assert first["best_metric"] == 1.0
    assert second["should_stop"] is False
    assert second["checks_since_improvement"] == 1
    assert third["should_stop"] is True
    assert third["best_step"] == 100
