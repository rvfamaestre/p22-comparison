from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from src.gpu.vec_env import VecEnvConfig, VecRingRoadEnv
from src.utils.primary_objective import (
    PRIMARY_OBJECTIVE_METRIC,
    SAFE_HEADWAY_M,
    annotate_with_primary_objective,
    mean_last_window,
)
from src.utils.string_stability_metrics import (
    STRING_STABILITY_METRIC,
    compute_string_stability_from_ordered_series,
    downstream_vehicle_order_from_follower_map,
    string_stability_report_metadata,
)


DEFAULT_EVAL_MODES = ("baseline", "adaptive", "rl")
DEFAULT_SHUFFLE_LAYOUTS = ("ordered", "shuffled")
DEFAULT_PERTURBATION_SETTINGS = ("off", "on")

_SHUFFLE_OPTION_MAP = {
    "0": False,
    "false": False,
    "non_shuffled": False,
    "nonshuffled": False,
    "off": False,
    "ordered": False,
    "unshuffled": False,
    "1": True,
    "on": True,
    "shuffle": True,
    "shuffled": True,
    "true": True,
}
_PERTURBATION_OPTION_MAP = {
    "0": False,
    "disabled": False,
    "false": False,
    "off": False,
    "1": True,
    "enabled": True,
    "on": True,
    "true": True,
}

ActionSelector = Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
InitialObsBuilder = Callable[[VecRingRoadEnv], tuple[torch.Tensor, torch.Tensor]]


def _finite_values(
    rows: Iterable[Mapping[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _optional_bool_values(
    rows: Iterable[Mapping[str, object]],
    key: str,
) -> list[bool]:
    values: list[bool] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        values.append(bool(value))
    return values


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else float("nan")


def _std_or_zero(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(stdev(values)) if len(values) > 1 else 0.0


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def _aggregate_string_stability_batch(
    results: Sequence[Mapping[str, object]],
    *,
    num_envs: int,
) -> Dict[str, object]:
    applicable_count = sum(
        1 for item in results if bool(item.get("string_stability_applicable", False))
    )
    valid_count = sum(
        1 for item in results if item.get("string_stability_metric_valid") is True
    )
    stable_count = sum(
        1 for item in results if item.get("string_stability_is_stable") is True
    )

    payload: Dict[str, object] = {
        "string_stability_metric": STRING_STABILITY_METRIC,
        "string_stability_threshold": float(results[0]["string_stability_threshold"])
        if results
        else float("nan"),
        "string_stability_baseline_window_steps": int(
            results[0]["string_stability_baseline_window_steps"]
        )
        if results
        else 0,
        "string_stability_applicable": bool(applicable_count > 0),
        "string_stability_metric_valid": None
        if applicable_count == 0
        else applicable_count == valid_count,
        "string_stability_is_stable": None
        if applicable_count == 0
        else applicable_count == stable_count,
        "string_stability_applicable_fraction": (
            float(applicable_count) / float(max(num_envs, 1))
        ),
        "string_stability_metric_valid_fraction": (
            float(valid_count) / float(applicable_count)
            if applicable_count > 0
            else float("nan")
        ),
        "string_stability_stable_fraction": (
            float(stable_count) / float(applicable_count)
            if applicable_count > 0
            else float("nan")
        ),
        "string_stability_num_applicable_parallel_envs": int(applicable_count),
        "string_stability_num_valid_parallel_envs": int(valid_count),
        "string_stability_value": _mean_or_nan(
            _finite_values(results, "string_stability_value")
        ),
        "string_stability_max_pairwise_speed_gain": _mean_or_nan(
            _finite_values(results, "string_stability_max_pairwise_speed_gain")
        ),
        "string_stability_max_downstream_spacing_error_amplification": _mean_or_nan(
            _finite_values(
                results,
                "string_stability_max_downstream_spacing_error_amplification",
            )
        ),
        "string_stability_source_peak_speed_error": _mean_or_nan(
            _finite_values(results, "string_stability_source_peak_speed_error")
        ),
    }
    return payload


def normalize_human_rates(values: Sequence[float]) -> List[float]:
    """Validate and de-duplicate human-rate values while preserving order."""
    normalized: List[float] = []
    for value in values:
        hr = float(value)
        if not 0.0 <= hr <= 1.0:
            raise ValueError(f"human_rate must be in [0, 1], got {value}.")
        if hr not in normalized:
            normalized.append(hr)
    if not normalized:
        raise ValueError("At least one human_rate must be provided.")
    return normalized


def normalize_seeds(values: Sequence[int]) -> List[int]:
    """Validate and de-duplicate evaluation seeds while preserving order."""
    normalized: List[int] = []
    for value in values:
        seed = int(value)
        if seed not in normalized:
            normalized.append(seed)
    if not normalized:
        raise ValueError("At least one seed must be provided.")
    return normalized


def resolve_eval_seeds(
    seeds: Sequence[int] | None,
    num_seeds: int,
) -> List[int]:
    """Resolve the explicit seed list, preserving the old --num_seeds fallback."""
    if seeds is not None:
        return normalize_seeds(seeds)
    if num_seeds <= 0:
        raise ValueError("--num_seeds must be positive when --seeds is omitted.")
    return list(range(num_seeds))


def normalize_eval_modes(values: Sequence[str]) -> List[str]:
    """Validate evaluation controller modes."""
    normalized: List[str] = []
    for value in values:
        mode = str(value).strip().lower()
        if mode not in DEFAULT_EVAL_MODES:
            known = ", ".join(DEFAULT_EVAL_MODES)
            raise ValueError(f"Unknown evaluation mode '{value}'. Known modes: {known}.")
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError("At least one evaluation mode must be provided.")
    return normalized


def _normalize_boolean_options(
    values: Sequence[str],
    *,
    mapping: Mapping[str, bool],
    label: str,
) -> List[bool]:
    normalized: List[bool] = []
    for value in values:
        key = str(value).strip().lower()
        if key not in mapping:
            choices = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown {label} option '{value}'. Known values: {choices}.")
        bool_value = mapping[key]
        if bool_value not in normalized:
            normalized.append(bool_value)
    if not normalized:
        raise ValueError(f"At least one {label} option must be provided.")
    return normalized


def normalize_shuffle_layouts(values: Sequence[str]) -> List[bool]:
    """Parse shuffle-layout labels such as ordered / shuffled."""
    return _normalize_boolean_options(
        values,
        mapping=_SHUFFLE_OPTION_MAP,
        label="shuffle_layout",
    )


def normalize_perturbation_settings(values: Sequence[str]) -> List[bool]:
    """Parse perturbation labels such as on / off."""
    return _normalize_boolean_options(
        values,
        mapping=_PERTURBATION_OPTION_MAP,
        label="perturbation",
    )


def shuffle_layout_label(enabled: bool) -> str:
    return "shuffled" if bool(enabled) else "ordered"


def perturbation_setting_label(enabled: bool) -> str:
    return "on" if bool(enabled) else "off"


def _iter_generalization_cases(
    *,
    modes: Sequence[str],
    human_rates: Sequence[float],
    seeds: Sequence[int],
    shuffle_layouts: Sequence[bool],
    perturbation_settings: Sequence[bool],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for human_rate in human_rates:
        for shuffle_cav_positions in shuffle_layouts:
            for perturbation_enabled in perturbation_settings:
                for mode in modes:
                    for seed in seeds:
                        cases.append(
                            {
                                "human_rate": float(human_rate),
                                "shuffle_cav_positions": bool(shuffle_cav_positions),
                                "perturbation_enabled": bool(perturbation_enabled),
                                "mode": str(mode),
                                "seed": int(seed),
                            }
                        )
    return cases


def _case_brief(case: Mapping[str, object]) -> str:
    return (
        f"HR={float(case['human_rate']):.2f} "
        f"layout={shuffle_layout_label(bool(case['shuffle_cav_positions']))} "
        f"perturb={perturbation_setting_label(bool(case['perturbation_enabled']))} "
        f"mode={case['mode']} seed={int(case['seed'])}"
    )


def seed_evaluation(seed: int, device: torch.device) -> None:
    """Set RNG state so each evaluated condition is reproducible."""
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_eval_cfg(
    env_cfg: VecEnvConfig,
    hr: float,
    *,
    meso_enabled: bool = True,
    shuffle_cav_positions: bool,
    perturbation_enabled: bool,
) -> VecEnvConfig:
    """Freeze one evaluation scenario while keeping training-time defaults elsewhere."""
    overrides = dict(env_cfg.__dict__)
    overrides["hr_options"] = (float(hr),)
    overrides["hr_weights"] = None
    overrides["meso_enabled"] = meso_enabled
    overrides["shuffle_cav_positions"] = bool(shuffle_cav_positions)
    overrides["perturbation_enabled"] = bool(perturbation_enabled)
    # Evaluation keeps perturbation timing/magnitude deterministic whenever enabled.
    overrides["perturb_curriculum"] = False
    return VecEnvConfig(**overrides)


@torch.no_grad()
def run_generalization_episode(
    trainer: Any,
    env_cfg: VecEnvConfig,
    device: torch.device,
    *,
    build_initial_obs: InitialObsBuilder,
    action_selector: ActionSelector,
    mode: str,
    human_rate: float,
    num_envs: int,
    seed: int,
    shuffle_cav_positions: bool,
    perturbation_enabled: bool,
) -> Dict[str, object]:
    """Run one deterministic evaluation episode for a fixed factor tuple."""
    seed_evaluation(seed, device)

    eval_cfg = make_eval_cfg(
        env_cfg,
        human_rate,
        meso_enabled=(mode != "baseline"),
        shuffle_cav_positions=shuffle_cav_positions,
        perturbation_enabled=perturbation_enabled,
    )
    env = VecRingRoadEnv(num_envs=num_envs, cfg=eval_cfg, device=device)
    env.reset()
    obs, mask = build_initial_obs(env)
    perturb_targets = env.perturb_target.detach().cpu().numpy().astype(int).copy()
    perturb_times = env.perturb_time.detach().cpu().numpy().astype(float).copy()
    perturb_deltas = env.perturb_dv.detach().cpu().numpy().astype(float).copy()
    trace_times = np.arange(eval_cfg.episode_steps, dtype=float) * float(eval_cfg.dt)
    perturbation_steps = np.searchsorted(trace_times, perturb_times, side="left")
    collect_string_stability = bool(perturbation_enabled)

    total_reward = torch.zeros(num_envs, device=device)
    mean_speed_trace = torch.empty(eval_cfg.episode_steps, device=device)
    speed_var_trace = torch.empty(eval_cfg.episode_steps, device=device)
    min_gap_episode = torch.full((), float("inf"), device=device)
    speed_history = (
        torch.empty(
            (eval_cfg.episode_steps, num_envs, eval_cfg.N),
            device=device,
            dtype=env.v.dtype,
        )
        if collect_string_stability
        else None
    )
    gap_history = (
        torch.empty(
            (eval_cfg.episode_steps, num_envs, eval_cfg.N),
            device=device,
            dtype=obs.dtype,
        )
        if collect_string_stability
        else None
    )
    follower_orders = np.full((num_envs, eval_cfg.N), -1, dtype=int)
    follower_order_captured = np.zeros(num_envs, dtype=bool)
    zero_delta_alphas = (
        torch.zeros(num_envs, eval_cfg.N, device=device)
        if mode in {"baseline", "adaptive"}
        else None
    )

    for step_idx in range(eval_cfg.episode_steps):
        if mode == "baseline":
            delta_alphas = zero_delta_alphas
        elif mode == "adaptive":
            delta_alphas = zero_delta_alphas
        elif mode == "rl":
            delta_alphas = action_selector(trainer, obs, mask)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        obs, reward, _, mask, _ = env.step(delta_alphas)
        total_reward += reward
        mean_speed_trace[step_idx] = env.v.mean()
        speed_var_trace[step_idx] = env.v.var(unbiased=False)
        min_gap_episode = torch.minimum(min_gap_episode, obs[:, :, 4].min())
        if collect_string_stability and speed_history is not None and gap_history is not None:
            speed_history[step_idx].copy_(env.v)
            gap_history[step_idx].copy_(obs[:, :, 4])
            capture_mask = (~follower_order_captured) & (step_idx >= perturbation_steps)
            if capture_mask.any():
                follower_orders[capture_mask] = (
                    env._follower_idx.detach().cpu().numpy()[capture_mask].astype(int)
                )
                follower_order_captured[capture_mask] = True

    clamp_counts = env.collision_clamp_count.detach().cpu().numpy().astype(int)
    if collect_string_stability and speed_history is not None and gap_history is not None:
        speed_tensor = speed_history.detach().cpu().numpy()
        gap_tensor = gap_history.detach().cpu().numpy()
        string_stability_results: List[Dict[str, object]] = []
        for env_idx in range(num_envs):
            if follower_order_captured[env_idx]:
                downstream_order = downstream_vehicle_order_from_follower_map(
                    follower_orders[env_idx].tolist(),
                    int(perturb_targets[env_idx]),
                )
            else:
                downstream_order = list(range(eval_cfg.N))
            string_stability_results.append(
                compute_string_stability_from_ordered_series(
                    speed_tensor[:, env_idx, :][:, downstream_order],
                    perturbation_step=int(perturbation_steps[env_idx]),
                    valid_base=bool(clamp_counts[env_idx] == 0),
                    applicable=True,
                    gap_series=gap_tensor[:, env_idx, :][:, downstream_order],
                )
            )
    else:
        not_applicable = compute_string_stability_from_ordered_series(
            np.zeros((1, 1), dtype=float),
            perturbation_step=0,
            valid_base=True,
            applicable=False,
        )
        string_stability_results = [dict(not_applicable) for _ in range(num_envs)]
    string_stability = _aggregate_string_stability_batch(
        string_stability_results,
        num_envs=num_envs,
    )

    mean_speed_trace_cpu = mean_speed_trace.detach().cpu().tolist()
    speed_var_trace_cpu = speed_var_trace.detach().cpu().tolist()

    return annotate_with_primary_objective(
        {
            "human_rate": float(human_rate),
            "cav_share": float(1.0 - human_rate),
            "mode": mode,
            "seed": int(seed),
            "shuffle_cav_positions": bool(shuffle_cav_positions),
            "shuffle_layout": shuffle_layout_label(shuffle_cav_positions),
            "perturbation_enabled": bool(perturbation_enabled),
            "perturbation_setting": perturbation_setting_label(perturbation_enabled),
            "mean_reward": float(total_reward.mean().item()),
            "mean_speed_all": float(fmean(mean_speed_trace_cpu)),
            "mean_speed_last100": mean_last_window(mean_speed_trace_cpu, 100),
            "speed_var_last100": mean_last_window(speed_var_trace_cpu, 100),
            "min_gap_episode": float(min_gap_episode.item()),
            "collision_clamp_count": int(clamp_counts.sum()),
            "string_stability_valid": bool(int(clamp_counts.sum()) == 0),
            "perturbation_target_vehicle": int(perturb_targets[0]) if num_envs > 0 else 0,
            "perturbation_time_s": float(perturb_times[0]) if num_envs > 0 else float("nan"),
            "perturbation_delta_v": float(perturb_deltas[0]) if num_envs > 0 else 0.0,
            **string_stability,
        },
        min_gap_key="min_gap_episode",
        min_gap_threshold=SAFE_HEADWAY_M,
        collision_clamp_count_key="collision_clamp_count",
    )


def summarise_rows(
    rows: Iterable[Dict[str, object]],
    *,
    group_keys: Sequence[str],
) -> List[Dict[str, object]]:
    """Aggregate evaluation rows into mean/std summaries."""
    row_list = list(rows)
    groups: Dict[tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in row_list:
        key = tuple(row[group_key] for group_key in group_keys)
        groups[key].append(row)

    metric_keys = [
        "mean_reward",
        "mean_speed_all",
        "mean_speed_last100",
        "speed_var_last100",
        "min_gap_episode",
        "collision_clamp_count",
        "string_stability_value",
        "string_stability_applicable_fraction",
        "string_stability_metric_valid_fraction",
        "string_stability_stable_fraction",
        "string_stability_max_pairwise_speed_gain",
        "string_stability_max_downstream_spacing_error_amplification",
        "string_stability_source_peak_speed_error",
    ]
    rate_keys = [
        "safety_min_gap_ok",
        "safety_constraint_satisfied",
        "safety_no_collision_clamps",
        "string_stability_valid",
        "string_stability_applicable",
        "string_stability_metric_valid",
        "string_stability_is_stable",
    ]
    summary: List[Dict[str, object]] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: tuple(_sort_key_value(value) for value in item[0]),
    ):
        record: Dict[str, object] = {
            group_key: key[idx] for idx, group_key in enumerate(group_keys)
        }
        record["num_trials"] = len(group)
        record["num_seeds"] = len({int(item["seed"]) for item in group if "seed" in item})
        record["primary_objective_metric"] = PRIMARY_OBJECTIVE_METRIC
        record["safety_min_gap_threshold"] = float(group[0]["safety_min_gap_threshold"])
        if "string_stability_metric" in group[0]:
            record["string_stability_metric"] = str(group[0]["string_stability_metric"])
        if "string_stability_threshold" in group[0]:
            record["string_stability_threshold"] = float(group[0]["string_stability_threshold"])
        if "shuffle_cav_positions" in record:
            record["shuffle_layout"] = shuffle_layout_label(
                bool(record["shuffle_cav_positions"])
            )
        if "perturbation_enabled" in record:
            record["perturbation_setting"] = perturbation_setting_label(
                bool(record["perturbation_enabled"])
            )

        objective_values = [float(item["primary_objective_value"]) for item in group]
        record["primary_objective_mean"] = float(fmean(objective_values))
        record["primary_objective_std"] = (
            float(stdev(objective_values)) if len(objective_values) > 1 else 0.0
        )
        for metric_key in metric_keys:
            values = _finite_values(group, metric_key)
            record[f"{metric_key}_mean"] = _mean_or_nan(values)
            record[f"{metric_key}_std"] = _std_or_zero(values)
        for rate_key in rate_keys:
            values = _optional_bool_values(group, rate_key)
            record[f"{rate_key}_rate"] = _mean_or_nan(
                [1.0 if value else 0.0 for value in values]
            )
        summary.append(record)
    return summary


def _sort_key_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: str | Path, rows: List[Dict[str, object]]) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _variation_item(
    *,
    name: str,
    present: bool,
    details: str,
    values: Sequence[object] | None = None,
) -> Dict[str, object]:
    item: Dict[str, object] = {
        "name": name,
        "present": bool(present),
        "details": details,
    }
    if values is not None:
        item["values"] = list(values)
    return item


def build_variation_manifest(
    *,
    algorithm: str,
    checkpoint: str,
    config_path: str,
    cfg: Mapping[str, object],
    env_cfg: VecEnvConfig,
    device: torch.device,
    num_envs: int,
    modes: Sequence[str],
    human_rates: Sequence[float],
    seeds: Sequence[int],
    shuffle_layouts: Sequence[bool],
    perturbation_settings: Sequence[bool],
    raw_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Document what varied during training versus what only varies in evaluation."""
    hr_weights = list(env_cfg.hr_weights) if env_cfg.hr_weights is not None else None
    human_ratio_sampling = len(env_cfg.hr_options) > 1 or hr_weights is not None
    layout_shuffle_training = bool(env_cfg.shuffle_cav_positions)
    perturbation_curriculum_training = bool(
        env_cfg.perturbation_enabled and env_cfg.perturb_curriculum
    )
    fixed_perturbation_training = bool(
        env_cfg.perturbation_enabled and not env_cfg.perturb_curriculum
    )
    stochastic_noise_training = float(env_cfg.noise_Q) > 0.0

    training_variation = [
        _variation_item(
            name="human_ratio_sampling",
            present=human_ratio_sampling,
            details=(
                "Training samples one human ratio per reset from rl.hr_options."
                if human_ratio_sampling
                else "Training uses one fixed human ratio for every reset."
            ),
            values=env_cfg.hr_options,
        ),
        _variation_item(
            name="cav_layout_shuffling",
            present=layout_shuffle_training,
            details=(
                "Training randomizes which vehicle slots are CAVs on each reset."
                if layout_shuffle_training
                else "Training keeps the CAV layout ordering fixed on each reset."
            ),
            values=[bool(env_cfg.shuffle_cav_positions)],
        ),
        _variation_item(
            name="perturbation_curriculum",
            present=perturbation_curriculum_training,
            details=(
                "Training randomizes perturbation timing, magnitude, and target vehicle."
                if perturbation_curriculum_training
                else (
                    "Training uses a fixed perturbation protocol."
                    if fixed_perturbation_training
                    else "Training disables perturbations entirely."
                )
            ),
            values=(
                [
                    env_cfg.perturb_time_min,
                    env_cfg.perturb_time_max,
                    env_cfg.perturb_dv_min,
                    env_cfg.perturb_dv_max,
                ]
                if perturbation_curriculum_training
                else [env_cfg.perturbation_time, env_cfg.perturbation_delta_v]
            ),
        ),
        _variation_item(
            name="stochastic_hdv_noise",
            present=stochastic_noise_training,
            details=(
                "Training includes Euler-Maruyama HDV noise, so each rollout realization differs."
                if stochastic_noise_training
                else "Training does not add HDV process noise."
            ),
            values=[float(env_cfg.noise_Q)],
        ),
    ]

    evaluation_variation = [
        _variation_item(
            name="explicit_human_rate_grid",
            present=True,
            details=(
                "Evaluation enumerates the requested human ratios instead of sampling them."
            ),
            values=human_rates,
        ),
        _variation_item(
            name="matched_seed_sweep",
            present=True,
            details=(
                "Evaluation re-runs each condition for every listed seed and reuses the same seed across modes for matched comparisons."
            ),
            values=seeds,
        ),
        _variation_item(
            name="layout_toggle",
            present=len(shuffle_layouts) > 1 or list(shuffle_layouts) != [layout_shuffle_training],
            details=(
                "Evaluation explicitly toggles between ordered and shuffled CAV layouts."
            ),
            values=[shuffle_layout_label(value) for value in shuffle_layouts],
        ),
        _variation_item(
            name="perturbation_toggle",
            present=len(perturbation_settings) > 1 or list(perturbation_settings) != [bool(env_cfg.perturbation_enabled)],
            details=(
                "Evaluation explicitly toggles perturbation on and off. When perturbation is on, curriculum randomization is disabled so all modes see the same fixed perturbation protocol for a given seed."
            ),
            values=[
                perturbation_setting_label(value) for value in perturbation_settings
            ],
        ),
    ]

    return {
        "algorithm": algorithm,
        "checkpoint": checkpoint,
        "config_path": config_path,
        "device": device.type,
        "num_envs": int(num_envs),
        "num_raw_rows": len(raw_rows),
        "modes": list(modes),
        "human_rates": list(human_rates),
        "cav_shares": [float(1.0 - hr) for hr in human_rates],
        "seeds": list(seeds),
        "shuffle_layouts": [shuffle_layout_label(value) for value in shuffle_layouts],
        "perturbation_settings": [
            perturbation_setting_label(value) for value in perturbation_settings
        ],
        "training_seed": int(cfg.get("seed", 42)),
        "parallel_envs_per_episode": int(num_envs),
        "training_variation": training_variation,
        "evaluation_variation": evaluation_variation,
        "string_stability": string_stability_report_metadata(),
    }


def run_generalization_sweep(
    trainer: Any,
    env_cfg: VecEnvConfig,
    device: torch.device,
    *,
    build_initial_obs: InitialObsBuilder,
    action_selector: ActionSelector,
    modes: Sequence[str],
    human_rates: Sequence[float],
    seeds: Sequence[int],
    shuffle_layouts: Sequence[bool],
    perturbation_settings: Sequence[bool],
    num_envs: int,
    progress_every: int = 1,
) -> Dict[str, List[Dict[str, object]]]:
    """Evaluate one checkpoint across the requested generalization grid."""
    progress_every = max(1, int(progress_every))
    cases = _iter_generalization_cases(
        modes=modes,
        human_rates=human_rates,
        seeds=seeds,
        shuffle_layouts=shuffle_layouts,
        perturbation_settings=perturbation_settings,
    )
    total_cases = len(cases)
    total_env_steps = total_cases * int(env_cfg.episode_steps) * int(num_envs)
    perturbation_case_count = sum(
        1 for case in cases if bool(case["perturbation_enabled"])
    )
    print(
        f"[gpu-eval] Starting sweep: cases={total_cases} "
        f"| episode_steps={int(env_cfg.episode_steps)} "
        f"| parallel_envs={int(num_envs)} "
        f"| total_env_steps={total_env_steps} "
        f"| perturbation_cases={perturbation_case_count}",
        flush=True,
    )
    rows: List[Dict[str, object]] = []
    sweep_start = time.time()
    for case_idx, case in enumerate(cases, start=1):
        case_start = time.time()
        row = run_generalization_episode(
            trainer,
            env_cfg,
            device,
            build_initial_obs=build_initial_obs,
            action_selector=action_selector,
            mode=str(case["mode"]),
            human_rate=float(case["human_rate"]),
            num_envs=num_envs,
            seed=int(case["seed"]),
            shuffle_cav_positions=bool(case["shuffle_cav_positions"]),
            perturbation_enabled=bool(case["perturbation_enabled"]),
        )
        rows.append(row)
        elapsed = time.time() - sweep_start
        avg_case_time = elapsed / float(case_idx)
        remaining_cases = total_cases - case_idx
        eta = avg_case_time * float(remaining_cases)
        if (
            case_idx == 1
            or case_idx == total_cases
            or case_idx % progress_every == 0
        ):
            primary_value = float(row["primary_objective_value"])
            string_value = row.get("string_stability_value")
            string_fragment = ""
            if string_value is not None:
                try:
                    string_numeric = float(string_value)
                except (TypeError, ValueError):
                    string_numeric = float("nan")
                if math.isfinite(string_numeric):
                    string_fragment = f" string_stability={string_numeric:.3f}"
            print(
                f"[gpu-eval {case_idx:03d}/{total_cases:03d}] {_case_brief(case)} "
                f"objective={primary_value:.3f} "
                f"safety={int(bool(row['safety_constraint_satisfied']))}"
                f"{string_fragment} "
                f"case={_format_duration(time.time() - case_start)} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}",
                flush=True,
            )
    factor_summary = summarise_rows(
        rows,
        group_keys=(
            "mode",
            "human_rate",
            "cav_share",
            "shuffle_cav_positions",
            "perturbation_enabled",
        ),
    )
    overall_summary = summarise_rows(
        rows,
        group_keys=("mode", "human_rate", "cav_share"),
    )
    return {
        "raw_rows": rows,
        "factor_summary": factor_summary,
        "overall_summary": overall_summary,
    }


def write_generalization_outputs(
    output_dir: str | Path,
    *,
    raw_rows: List[Dict[str, object]],
    factor_summary: List[Dict[str, object]],
    overall_summary: List[Dict[str, object]],
    manifest: Dict[str, object],
    generate_plots: bool = True,
) -> Dict[str, str]:
    """Persist evaluation artifacts and generate plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_csv = output_path / "gpu_eval_raw.csv"
    raw_json = output_path / "gpu_eval_raw.json"
    summary_csv = output_path / "gpu_eval_summary.csv"
    summary_json = output_path / "gpu_eval_summary.json"
    overall_csv = output_path / "gpu_eval_overall_summary.csv"
    overall_json = output_path / "gpu_eval_overall_summary.json"
    manifest_json = output_path / "gpu_eval_manifest.json"
    report_json = output_path / "gpu_eval_report.json"

    write_csv(raw_csv, raw_rows)
    write_json(raw_json, raw_rows)
    write_csv(summary_csv, factor_summary)
    write_json(summary_json, factor_summary)
    write_csv(overall_csv, overall_summary)
    write_json(overall_json, overall_summary)
    write_json(manifest_json, manifest)
    write_json(
        report_json,
        {
            "manifest": manifest,
            "raw_rows": raw_rows,
            "factor_summary": factor_summary,
            "overall_summary": overall_summary,
        },
    )

    if generate_plots:
        from src.visualization.analysis_plots import (
            generate_comparison_report,
            generate_generalization_report,
        )

        generate_generalization_report(summary_csv, output_path / "summary_plots")
        generate_comparison_report(raw_csv, output_path / "summary_plots_overall")

    return {
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "overall_csv": str(overall_csv),
        "manifest_json": str(manifest_json),
        "report_json": str(report_json),
    }
