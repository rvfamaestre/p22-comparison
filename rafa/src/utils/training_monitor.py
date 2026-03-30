from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Mapping


TRAIN_LOG_FIELDS = [
    "algorithm",
    "step",
    "update",
    "elapsed_s",
    "steps_per_second",
    "reward",
    "pg_loss",
    "v_loss",
    "entropy",
    "q1_loss",
    "actor_loss",
    "alpha",
    "replay_size",
    "lr",
    "actor_lr",
    "critic_lr",
    "alpha_lr",
    "w_alpha",
]

EVAL_LOG_FIELDS = [
    "algorithm",
    "step",
    "elapsed_s",
    "human_rate",
    "primary_objective_metric",
    "eval_mean_speed",
    "eval_mean_speed_over_human_rates",
]


def _append_csv_row(path: Path, fieldnames: list[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(fieldnames=fieldnames, f=handle, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


class TrainingMonitor:
    """Append scalar logs and periodically refresh static progress plots."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        algorithm: str,
        enable_plots: bool = True,
        plot_refresh_seconds: float = 30.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.algorithm = str(algorithm)
        self.enable_plots = bool(enable_plots)
        self.plot_refresh_seconds = max(0.0, float(plot_refresh_seconds))
        self.training_log_path = self.output_dir / "training_log.csv"
        self.eval_log_path = self.output_dir / "eval_log.csv"
        self._last_plot_refresh_time = 0.0

    def log_training(self, row: Mapping[str, object]) -> None:
        payload = {"algorithm": self.algorithm, **row}
        _append_csv_row(self.training_log_path, TRAIN_LOG_FIELDS, payload)

    def log_evaluation(
        self,
        *,
        step: int,
        elapsed_s: float,
        speeds: Mapping[float, float],
        primary_objective_metric: str,
    ) -> None:
        if not speeds:
            return
        speed_values = [float(speed) for speed in speeds.values()]
        mean_speed = sum(speed_values) / float(len(speed_values))
        for human_rate, speed in sorted(speeds.items()):
            payload = {
                "algorithm": self.algorithm,
                "step": int(step),
                "elapsed_s": float(elapsed_s),
                "human_rate": float(human_rate),
                "primary_objective_metric": str(primary_objective_metric),
                "eval_mean_speed": float(speed),
                "eval_mean_speed_over_human_rates": float(mean_speed),
            }
            _append_csv_row(self.eval_log_path, EVAL_LOG_FIELDS, payload)

    def maybe_refresh_plots(self, *, force: bool = False) -> None:
        if not self.enable_plots:
            return
        now = time.time()
        if not force and (now - self._last_plot_refresh_time) < self.plot_refresh_seconds:
            return
        try:
            from src.visualization.analysis_plots import generate_all_from_directory

            generate_all_from_directory(self.output_dir)
            self._last_plot_refresh_time = now
        except Exception as exc:
            print(f"[training-monitor] WARNING: plot refresh failed: {exc}")
