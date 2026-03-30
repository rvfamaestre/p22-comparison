from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class EarlyStoppingConfig:
    """Configuration for plateau-based early stopping."""

    enabled: bool = False
    metric: str = "eval_mean_speed"
    mode: str = "max"
    patience: int = 6
    min_delta: float = 0.02
    start_step: int = 0
    min_checks: int = 4
    ema_alpha: float = 0.0
    restore_best: bool = True

    def validate(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError(f"Unsupported early-stopping mode '{self.mode}'.")
        if self.patience <= 0:
            raise ValueError("Early-stopping patience must be positive.")
        if self.min_delta < 0:
            raise ValueError("Early-stopping min_delta must be non-negative.")
        if self.start_step < 0:
            raise ValueError("Early-stopping start_step must be non-negative.")
        if self.min_checks <= 0:
            raise ValueError("Early-stopping min_checks must be positive.")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("Early-stopping ema_alpha must be in [0, 1].")


class EarlyStopMonitor:
    """Track a scalar metric and stop when it plateaus."""

    def __init__(self, cfg: EarlyStoppingConfig):
        cfg.validate()
        self.cfg = cfg
        self.num_checks = 0
        self.eligible_checks = 0
        self.checks_since_improvement = 0
        self.best_metric: Optional[float] = None
        self.best_raw_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.last_metric: Optional[float] = None
        self.last_raw_metric: Optional[float] = None
        self.stop_step: Optional[int] = None

    def _is_better(self, candidate: float, reference: float) -> bool:
        if self.cfg.mode == "max":
            return candidate > reference + self.cfg.min_delta
        return candidate < reference - self.cfg.min_delta

    def _smooth(self, raw_metric: float) -> float:
        if self.last_metric is None or self.cfg.ema_alpha <= 0.0:
            return raw_metric
        alpha = self.cfg.ema_alpha
        return alpha * raw_metric + (1.0 - alpha) * self.last_metric

    def update(self, raw_metric: float, step: int) -> Dict[str, object]:
        if not math.isfinite(raw_metric):
            raise ValueError(
                f"Early-stopping metric must be finite, got {raw_metric}."
            )

        self.num_checks += 1
        smoothed_metric = self._smooth(float(raw_metric))
        self.last_raw_metric = float(raw_metric)
        self.last_metric = float(smoothed_metric)

        improved = False
        tracking_started = step >= self.cfg.start_step
        ready = False
        if tracking_started:
            self.eligible_checks += 1
            if self.best_metric is None or self._is_better(
                smoothed_metric, self.best_metric
            ):
                self.best_metric = float(smoothed_metric)
                self.best_raw_metric = float(raw_metric)
                self.best_step = int(step)
                self.checks_since_improvement = 0
                improved = True

            ready = self.eligible_checks >= self.cfg.min_checks
            if ready and not improved:
                self.checks_since_improvement += 1

        should_stop = ready and self.checks_since_improvement >= self.cfg.patience
        if should_stop and self.stop_step is None:
            self.stop_step = int(step)

        return {
            "raw_metric": float(raw_metric),
            "smoothed_metric": float(smoothed_metric),
            "best_metric": self.best_metric,
            "best_raw_metric": self.best_raw_metric,
            "best_step": self.best_step,
            "checks": self.num_checks,
            "eligible_checks": self.eligible_checks,
            "checks_since_improvement": self.checks_since_improvement,
            "tracking_started": tracking_started,
            "ready": ready,
            "improved": improved,
            "should_stop": should_stop,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            **asdict(self.cfg),
            "num_checks": self.num_checks,
            "eligible_checks": self.eligible_checks,
            "checks_since_improvement": self.checks_since_improvement,
            "best_metric": self.best_metric,
            "best_raw_metric": self.best_raw_metric,
            "best_step": self.best_step,
            "last_metric": self.last_metric,
            "last_raw_metric": self.last_raw_metric,
            "stop_step": self.stop_step,
        }
