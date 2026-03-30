"""
Experiment orchestration and results pipeline.

Runs a sweep over human rates, controller modes, and seeds.
For each run, saves traces, metrics, and plots under Results/.
Generates aggregated summaries and plots across all runs.

Supported controller modes:
- baseline : CACC only
- adaptive : mesoscopic adaptation only
- rl       : legacy alias for PPO residual policy
- ppo      : PPO residual policy
- sac      : SAC residual policy

"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

import torch

from src.simulation.scenario_manager import ScenarioManager
from src.utils.config import get_default_config, merge_config, validate_config
from src.utils.primary_objective import (
    PRIMARY_OBJECTIVE_METRIC,
    SAFE_HEADWAY_M,
    annotate_with_primary_objective,
    coerce_bool,
    mean_last_window,
)
from src.utils.string_stability_metrics import (
    STRING_STABILITY_METRIC,
    compute_string_stability_from_traces,
    string_stability_report_metadata,
)
from src.agents.rl_types import RLConfig
from src.agents.networks import ActorCritic
from src.agents.ppo_trainer import PPOTrainer
from src.agents.observation_builder import ObservationBuilder, RunningNormalizer
from src.agents.reward import RewardFunction
from src.gpu.gpu_sac import SACActorGPU
from src.vehicles.cav_vehicle import CAVVehicle
from src.visualization.analysis_plots import (
    generate_all_from_directory,
    generate_comparison_report,
)


HUMAN_RATES = [1.0, 0.75, 0.5, 0.25, 0.0]
DEFAULT_MODES = ["baseline", "adaptive", "rl"]
ALL_MODES = ["baseline", "adaptive", "rl", "ppo", "sac"]
SEEDS = [0, 1, 2, 3, 4]

# Path to the trained RL checkpoints used for evaluation runs.
DEFAULT_PPO_CHECKPOINT = "output/rl_train/ckpt_final.pt"
DEFAULT_SAC_CHECKPOINT = "output/sac_train/gpu_sac_final.pt"

RESULTS_DIR = "Results"


def format_duration(seconds: float) -> str:
    """Render elapsed wall-clock seconds in a compact human-readable form."""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


# ----------------------------
# Config helpers
# ----------------------------
def load_base_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)
    base_config = merge_config(user_config, get_default_config())
    validate_config(base_config)
    return base_config


def normalize_modes(modes: List[str]) -> List[str]:
    """Validate and de-duplicate requested controller modes."""
    normalized: List[str] = []
    for mode in modes:
        mode_norm = mode.lower()
        if mode_norm not in ALL_MODES:
            known = ", ".join(ALL_MODES)
            raise ValueError(f"Unknown mode '{mode}'. Known modes: {known}.")
        if mode_norm not in normalized:
            normalized.append(mode_norm)
    return normalized


def normalize_human_rates(human_rates: List[float]) -> List[float]:
    """Validate and de-duplicate requested human ratios."""
    normalized: List[float] = []
    for human_rate in human_rates:
        hr = float(human_rate)
        if not 0.0 <= hr <= 1.0:
            raise ValueError(f"human_rate must be in [0, 1], got {human_rate}.")
        if hr not in normalized:
            normalized.append(hr)
    if not normalized:
        raise ValueError("At least one human_rate must be provided.")
    return normalized


def normalize_seeds(seeds: List[int]) -> List[int]:
    """Validate and de-duplicate requested seeds."""
    normalized: List[int] = []
    for seed in seeds:
        seed_int = int(seed)
        if seed_int not in normalized:
            normalized.append(seed_int)
    if not normalized:
        raise ValueError("At least one seed must be provided.")
    return normalized


def human_rate_key(human_rate: float) -> str:
    """Stable string key for float-valued human-rate lookups."""
    return f"{float(human_rate):.8f}".rstrip("0").rstrip(".") or "0"


def human_rate_tag(human_rate: float) -> str:
    """Filesystem-friendly human-rate tag."""
    return human_rate_key(human_rate).replace(".", "p")


def parse_hr_checkpoint_overrides(
    entries: Optional[List[str]],
    label: str,
) -> Dict[str, str]:
    """Parse CLI items of the form HR=PATH into a lookup table."""
    overrides: Dict[str, str] = {}
    if not entries:
        return overrides

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"{label} checkpoint override '{entry}' must use the form HR=PATH."
            )
        hr_text, ckpt_path = entry.split("=", 1)
        hr_key = human_rate_key(float(hr_text))
        if not ckpt_path:
            raise ValueError(f"{label} checkpoint override '{entry}' is missing a path.")
        overrides[hr_key] = ckpt_path

    return overrides


def format_checkpoint_template(template: str, human_rate: float) -> str:
    """Render a checkpoint template with HR / CAV share placeholders."""
    cav_share = 1.0 - human_rate
    return template.format(
        human_rate=human_rate_key(human_rate),
        human_rate_tag=human_rate_tag(human_rate),
        cav_share=human_rate_key(cav_share),
        cav_share_tag=human_rate_tag(cav_share),
    )


def resolve_checkpoint_path(
    default_path: str,
    template: Optional[str],
    overrides: Dict[str, str],
    human_rate: float,
) -> str:
    """Resolve checkpoint path for a given human rate."""
    hr_key = human_rate_key(human_rate)
    if hr_key in overrides:
        return overrides[hr_key]
    if template:
        return format_checkpoint_template(template, human_rate)
    return default_path


def override_config(base: Dict, human_rate: float, mode: str, seed: int) -> Dict:
    cfg = copy.deepcopy(base)
    cfg["human_ratio"] = human_rate
    cfg["seed"] = seed

    # Ensure mesoscopic config exists
    if "mesoscopic" not in cfg or cfg["mesoscopic"] is None:
        cfg["mesoscopic"] = {}
    residual_modes = ("rl", "ppo", "sac")
    cfg["mesoscopic"]["enabled"] = mode in ("adaptive",) + residual_modes

    if mode in residual_modes:
        cfg.setdefault("rl", {})
        cfg["rl"]["rl_mode"] = "residual"
        cfg["rl"]["delta_alpha_max"] = 0.5
        cfg["rl"]["alpha_min"] = 0.5
        cfg["rl"]["alpha_max"] = 2.0
        cfg["rl"]["hidden_dim"] = 128
        cfg["rl"]["num_hidden"] = 2
        cfg.setdefault("dx", 1.0)
        cfg.setdefault("kernel_h", 3.0)

    shuffle_cav_positions = cfg.get("shuffle_cav_positions")
    if shuffle_cav_positions is None:
        shuffle_cav_positions = cfg.get("rl", {}).get("shuffle_cav_positions")
    if shuffle_cav_positions is None:
        shuffle_cav_positions = True
    cfg["shuffle_cav_positions"] = bool(shuffle_cav_positions)

    # FORCE UNIFORM INITIAL CONDITIONS for scientific experiments
    # This ensures clean wave emergence and reproducible dynamics
    cfg["initial_conditions"] = "uniform"
    cfg["perturbation_enabled"] = True
    cfg["perturbation_vehicle"] = 0
    cfg["perturbation_time"] = 3.0  # 3 seconds of perfect uniformity
    cfg["perturbation_delta_v"] = -2.0
    cfg["noise_warmup_time"] = 3.0  # Suppress noise for first 3 seconds

    # DISABLE VISUALIZATION for batch experiments (much faster)
    cfg["enable_live_viz"] = False
    cfg["play_recording"] = False
    cfg["compute_macro_fields"] = False
    cfg["macro_teacher"] = "none"
    cfg["save_macro_dataset"] = False
    cfg["step_log_interval"] = int(cfg.get("step_log_interval", 100) or 100)

    # Optional metadata
    if "controller" not in cfg or cfg["controller"] is None:
        cfg["controller"] = {}
    cfg["controller"]["mode"] = mode
    cfg["controller_mode"] = mode
    cfg["scenario"] = f"HR_{human_rate}_MODE_{mode}_seed_{seed}"
    return cfg


# ----------------------------
# Simulation run
# ----------------------------
def run_single_simulation(config: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config["output_path"] = str(out_dir)

    # Build and run
    manager = ScenarioManager(config)
    sim = manager.build(live_viz=None)
    sim.run()

    # Save effective config
    with open(out_dir / "config_effective.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


def infer_mlp_shape(state_dict: Dict[str, torch.Tensor], prefix: str = "backbone") -> Tuple[int, int, int]:
    """Infer obs_dim, hidden_dim, and num_hidden from a Sequential MLP state dict."""
    weight_keys = []
    for key in state_dict:
        if key.startswith(f"{prefix}.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                weight_keys.append((int(parts[1]), key))
    if not weight_keys:
        raise ValueError(f"Could not infer MLP shape from prefix '{prefix}'.")

    weight_keys.sort()
    first_weight = state_dict[weight_keys[0][1]]
    obs_dim = int(first_weight.shape[1])
    hidden_dim = int(first_weight.shape[0])
    num_hidden = len(weight_keys)
    return obs_dim, hidden_dim, num_hidden


def load_normalizer_from_ckpt(ckpt: Dict, obs_dim: int) -> RunningNormalizer | None:
    """Extract observation normalizer stats from a checkpoint when present."""
    if "normalizer" not in ckpt:
        return None

    norm = RunningNormalizer(obs_dim)
    nd = ckpt["normalizer"]
    if hasattr(nd["mean"], "numpy"):
        norm.mean = nd["mean"].numpy().astype(np.float64)
        norm.var = nd["var"].numpy().astype(np.float64)
        norm._M2 = nd["M2"].numpy().astype(np.float64)
    else:
        norm.mean = np.asarray(nd["mean"], dtype=np.float64)
        norm.var = np.asarray(nd["var"], dtype=np.float64)
        norm._M2 = np.asarray(nd["M2"], dtype=np.float64)
    norm.count = int(nd["count"])
    norm.freeze()
    return norm


def load_ppo_runtime(ckpt_path: str, config: Dict) -> Tuple[ActorCritic, RunningNormalizer | None, int]:
    """Load a PPO-compatible policy and optional normalizer for inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "policy_state_dict" not in ckpt:
        raise KeyError(
            f"PPO checkpoint '{ckpt_path}' does not contain 'policy_state_dict'."
        )

    policy_state = ckpt["policy_state_dict"]
    obs_dim, hidden_dim, num_hidden = infer_mlp_shape(policy_state)
    rl_sub = config.get("rl", {})
    rl_cfg = RLConfig(
        delta_alpha_max=float(rl_sub.get("delta_alpha_max", 0.5)),
        alpha_min=float(rl_sub.get("alpha_min", 0.5)),
        alpha_max=float(rl_sub.get("alpha_max", 2.0)),
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
    )
    policy = ActorCritic(rl_cfg, obs_dim=obs_dim)
    policy.load_state_dict(policy_state)
    policy.eval()

    normalizer = load_normalizer_from_ckpt(ckpt, obs_dim)
    if normalizer is None:
        norm_path = os.path.join(os.path.dirname(ckpt_path), "obs_normalizer.npz")
        if os.path.exists(norm_path):
            normalizer = RunningNormalizer(obs_dim)
            normalizer.load(norm_path)
            normalizer.freeze()

    return policy, normalizer, obs_dim


def load_sac_runtime(config: Dict, ckpt_path: str) -> Tuple[SACActorGPU, RunningNormalizer | None, int]:
    """Load a SAC actor and optional normalizer for inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError(
            f"SAC checkpoint '{ckpt_path}' does not contain 'actor_state_dict'."
        )

    actor_state = ckpt["actor_state_dict"]
    obs_dim, hidden_dim, num_hidden = infer_mlp_shape(actor_state)
    delta_alpha_max = float(config.get("rl", {}).get("delta_alpha_max", 0.5))

    policy = SACActorGPU(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        delta_alpha_max=delta_alpha_max,
    )
    policy.load_state_dict(actor_state)
    policy.eval()

    normalizer = load_normalizer_from_ckpt(ckpt, obs_dim)
    return policy, normalizer, obs_dim


def normalize_obs_batch(obs: np.ndarray, normalizer: RunningNormalizer | None) -> torch.Tensor:
    """Apply a frozen normalizer when available and return a CPU tensor."""
    obs_norm = normalizer.normalize(obs) if normalizer is not None else obs.astype(np.float32)
    return torch.tensor(obs_norm, dtype=torch.float32)


def run_policy_simulation(
    config: Dict,
    out_dir: Path,
    ckpt_path: str,
    algorithm: str,
) -> None:
    """Run a trained PPO or SAC policy deterministically and log the rollout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config["output_path"] = str(out_dir)
    config["enable_live_viz"] = False
    config["play_recording"] = False

    print(
        f"[POLICY] mode={algorithm} checkpoint={ckpt_path} output={out_dir}",
        flush=True,
    )

    manager = ScenarioManager(config)
    sim = manager.build(live_viz=None)

    # Count CAVs and build helpers
    cav_ids = [v.id for v in sim.env.vehicles if isinstance(v, CAVVehicle)]
    num_cav = len(cav_ids)
    if num_cav == 0:
        sim.run()
        with open(out_dir / "config_effective.yml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)
        return

    if algorithm == "ppo":
        policy, normalizer, obs_dim = load_ppo_runtime(ckpt_path, config)
    elif algorithm == "sac":
        policy, normalizer, obs_dim = load_sac_runtime(config, ckpt_path)
    else:
        raise ValueError(f"Unsupported policy algorithm '{algorithm}'.")

    meso_M = config.get("mesoscopic", {}).get("M", 8)
    obs_builder = ObservationBuilder(M=meso_M, normalize=False, obs_dim=obs_dim)

    alpha_prev = {cid: 1.0 for cid in cav_ids}

    # Run one priming step so leaders/gaps are initialised
    sim.step()

    while not sim.done:
        obs, cur_cav_ids = obs_builder.build(sim.env.vehicles, sim.env.L, alpha_prev)
        if len(cur_cav_ids) == 0:
            sim.step()
            continue

        obs_t = normalize_obs_batch(obs, normalizer)
        with torch.no_grad():
            if algorithm == "ppo":
                actions, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
            else:
                actions, _ = policy.get_action(obs_t, deterministic=True)
        actions_np = actions.cpu().numpy()  # (num_cav, 1)

        delta_alphas = {
            cid: float(actions_np[k, 0]) for k, cid in enumerate(cur_cav_ids)
        }
        sim.set_rl_actions(delta_alphas)
        sim.step()

        for v in sim.env.vehicles:
            if isinstance(v, CAVVehicle):
                alpha_prev[v.id] = getattr(v, "_meso_alpha", 1.0)

    sim.update_logger_metadata()
    sim.logger.save()

    with open(out_dir / "config_effective.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


# ----------------------------
# Trace post-processing
# ----------------------------
def _map_vehicle_type(type_name: str) -> str:
    if type_name == "CAVVehicle":
        return "CAV"
    return "HDV"


def build_traces(
    micro_csv: Path,
    metadata_json: Path,
    mode: str,
) -> pd.DataFrame:
    df = pd.read_csv(micro_csv)
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    L = float(meta["L"])

    df = df.rename(
        columns={
            "t": "time",
            "id": "vehicle_id",
            "x": "x",
            "v": "v",
            "a": "a",
            "type": "vehicle_type_raw",
        }
    )
    df["vehicle_type"] = df["vehicle_type_raw"].map(_map_vehicle_type)

    # Compute leader_id, gap_s, rel_speed_dv per time step
    records = []
    for t, group in df.groupby("time", sort=True):
        g = group.sort_values("x").reset_index(drop=True)
        n = len(g)
        leader_ids = []
        gaps = []
        rel_speeds = []
        for i in range(n):
            self_row = g.iloc[i]
            leader_row = g.iloc[(i + 1) % n]
            leader_ids.append(int(leader_row["vehicle_id"]))
            gap = leader_row["x"] - self_row["x"]
            if gap < 0:
                gap += L
            gaps.append(float(gap))
            rel_speeds.append(float(leader_row["v"] - self_row["v"]))

        g = g.assign(
            leader_id=leader_ids,
            gap_s=gaps,
            rel_speed_dv=rel_speeds,
        )
        records.append(g)

    traces = pd.concat(records, ignore_index=True)
    traces = traces[
        [
            "time",
            "vehicle_id",
            "vehicle_type",
            "x",
            "v",
            "a",
            "leader_id",
            "gap_s",
            "rel_speed_dv",
            "alpha",
            "meso_h_c",
            "meso_mu_v",
            "meso_sigma_v",
        ]
    ].copy()

    if mode in ("adaptive", "rl", "ppo", "sac"):
        traces = traces.rename(
            columns={
                "meso_h_c": "headway_hc_effective",
                "meso_mu_v": "mu_v_ahead",
                "meso_sigma_v": "sigma_v_ahead",
            }
        )
        traces = traces[
            [
                "time",
                "vehicle_id",
                "vehicle_type",
                "x",
                "v",
                "a",
                "leader_id",
                "gap_s",
                "rel_speed_dv",
                "alpha",
                "headway_hc_effective",
                "mu_v_ahead",
                "sigma_v_ahead",
            ]
        ]
    else:
        traces = traces[
            [
                "time",
                "vehicle_id",
                "vehicle_type",
                "x",
                "v",
                "a",
                "leader_id",
                "gap_s",
                "rel_speed_dv",
            ]
        ]

    return traces


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(
    traces: pd.DataFrame,
    dt: float,
    mode: str,
    metadata: Optional[Dict] = None,
) -> Dict[str, object]:
    v = traces["v"].to_numpy()
    a = traces["a"].to_numpy()
    mean_speed_by_time = (
        traces.groupby("time", sort=True)["v"].mean().to_numpy(dtype=float).tolist()
    )

    metrics = {
        "speed_var_global": float(np.var(v)),
        "speed_std_time_mean": float(traces.groupby("time")["v"].std().mean()),
        "oscillation_amplitude": float(v.max() - v.min()),
        "min_gap": float(traces["gap_s"].min()),
        "rms_acc": float(np.sqrt(np.mean(a**2))),
        "mean_speed": float(np.mean(v)),
        "mean_speed_last100": mean_last_window(mean_speed_by_time, 100),
    }

    # RMS jerk per vehicle
    jerk_vals = []
    for _, g in traces.groupby("vehicle_id"):
        g_sorted = g.sort_values("time")
        jerk = np.diff(g_sorted["a"].to_numpy()) / dt
        if len(jerk) > 0:
            jerk_vals.append(np.mean(jerk**2))
    rms_jerk = float(np.sqrt(np.mean(jerk_vals))) if jerk_vals else 0.0
    metrics["rms_jerk"] = rms_jerk

    if mode in ("adaptive", "rl", "ppo", "sac"):
        cav = traces[traces["vehicle_type"] == "CAV"]
        metrics["alpha_mean"] = float(cav["alpha"].mean())
        metrics["alpha_std"] = float(cav["alpha"].std())
        metrics["headway_effective_mean"] = float(cav["headway_hc_effective"].mean())

    string_stability_valid_base = True
    string_stability_applicable = False
    if metadata is not None:
        metrics["collision_count"] = int(metadata.get("collision_count", 0))
        metrics["collision_clamp_count"] = int(
            metadata.get("collision_clamp_count", 0)
        )
        metrics["string_stability_valid"] = bool(
            metadata.get("string_stability_valid", True)
        )
        metrics["steps_completed"] = int(metadata.get("steps_completed", 0))
        string_stability_valid_base = bool(metrics["string_stability_valid"])
        string_stability_applicable = bool(metadata.get("perturbation_enabled", False))

    perturb_vehicle_id = 0
    perturbation_time = 0.0
    if metadata is not None:
        perturb_vehicle_id = int(
            metadata.get(
                "perturbation_target_vehicle_actual",
                metadata.get("perturbation_vehicle", 0),
            )
        )
        perturbation_time = float(metadata.get("perturbation_time", 0.0))
        metrics["perturbation_target_vehicle"] = perturb_vehicle_id
        metrics["perturbation_time_s"] = perturbation_time
        metrics["perturbation_delta_v"] = float(
            metadata.get("perturbation_delta_v", 0.0)
        )

    metrics.update(
        compute_string_stability_from_traces(
            traces,
            perturb_vehicle_id=perturb_vehicle_id,
            perturbation_time=perturbation_time,
            valid_base=string_stability_valid_base,
            applicable=string_stability_applicable,
        )
    )

    return annotate_with_primary_objective(
        metrics,
        min_gap_key="min_gap",
        min_gap_threshold=SAFE_HEADWAY_M,
        collision_count_key="collision_count" if metadata is not None else None,
        collision_clamp_count_key=(
            "collision_clamp_count" if metadata is not None else None
        ),
    )


# ----------------------------
# Plotting helpers
# ----------------------------
def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_speed_traces(traces: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 4))
    for vid, g in traces.groupby("vehicle_id"):
        plt.plot(g["time"], g["v"], linewidth=0.8, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed Traces")
    _save_plot(out_dir / "plots" / "speed_traces.png")


def plot_spacetime_heatmap(traces: pd.DataFrame, out_dir: Path) -> None:
    # Order vehicles by initial position
    t0 = traces["time"].min()
    order = traces[traces["time"] == t0].sort_values("x")["vehicle_id"].tolist()
    pivot = traces.pivot(index="time", columns="vehicle_id", values="v")
    pivot = pivot[order]
    plt.figure(figsize=(9, 5))
    plt.imshow(
        pivot.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[pivot.index.min(), pivot.index.max(), 0, len(order)],
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Vehicle (ordered)")
    plt.title("Spacetime Heatmap (Speed)")
    plt.colorbar(label="Speed [m/s]")
    _save_plot(out_dir / "plots" / "spacetime_heatmap_speed.png")


def plot_headway_traces(traces: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 4))
    for vid, g in traces.groupby("vehicle_id"):
        plt.plot(g["time"], g["gap_s"], linewidth=0.8, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Gap [m]")
    plt.title("Headway (Gap) Traces")
    _save_plot(out_dir / "plots" / "headway_traces.png")


def plot_alpha_traces(traces: pd.DataFrame, out_dir: Path) -> None:
    cav = traces[traces["vehicle_type"] == "CAV"]
    if cav.empty:
        return

    plt.figure(figsize=(9, 4))
    for vid, g in cav.groupby("vehicle_id"):
        plt.plot(g["time"], g["alpha"], linewidth=0.8, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Alpha")
    plt.title("Alpha Traces (CAVs)")

    # Overlay sigma_v_ahead for a representative CAV (std of vehicles ahead)
    rep_id = cav["vehicle_id"].iloc[0]
    rep = cav[cav["vehicle_id"] == rep_id]
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(rep["time"], rep["sigma_v_ahead"], color="black", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Sigma_v_ahead (Leaders)")
    _save_plot(out_dir / "plots" / "alpha_traces.png")


# ----------------------------
# Aggregation
# ----------------------------
def write_summary_runs(rows: List[Dict], out_dir: Path) -> Path:
    df = pd.DataFrame(rows)
    path = out_dir / "summary_runs.csv"
    df.to_csv(path, index=False)
    return path


def write_summary_by_human_rate(summary_runs: pd.DataFrame, out_dir: Path) -> Path:
    group_cols = ["mode", "human_rate", "cav_share"]
    agg_cols = [
        "speed_var_global",
        "speed_std_time_mean",
        "oscillation_amplitude",
        "min_gap",
        "rms_acc",
        "rms_jerk",
        "mean_speed",
        "mean_speed_last100",
        "primary_objective_value",
        "string_stability_value",
        "string_stability_max_pairwise_speed_gain",
        "string_stability_max_downstream_spacing_error_amplification",
        "string_stability_source_peak_speed_error",
    ]
    agg_cols = [col for col in agg_cols if col in summary_runs.columns]
    adaptive_cols = ["alpha_mean", "alpha_std", "headway_effective_mean"]
    diagnostic_cols = [
        "collision_count",
        "collision_clamp_count",
        "string_stability_valid",
        "string_stability_metric_valid",
        "string_stability_is_stable",
        "string_stability_applicable",
        "steps_completed",
    ]
    safety_rate_cols = [
        "safety_min_gap_ok",
        "safety_collision_free",
        "safety_no_collision_clamps",
        "safety_constraint_satisfied",
        "string_stability_valid",
        "string_stability_metric_valid",
        "string_stability_is_stable",
        "string_stability_applicable",
    ]
    for col in adaptive_cols:
        if col in summary_runs.columns:
            agg_cols.append(col)
    for col in diagnostic_cols:
        if col in summary_runs.columns:
            agg_cols.append(col)

    grouped = (
        summary_runs.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index()
    )

    # Flatten columns
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            if col[1]:
                flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col[0])
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    std_cols = [col for col in grouped.columns if str(col).endswith("_std")]
    if std_cols:
        grouped[std_cols] = grouped[std_cols].fillna(0.0)
    grouped = grouped.rename(
        columns={
            "primary_objective_value_mean": "primary_objective_mean",
            "primary_objective_value_std": "primary_objective_std",
        }
    )

    present_rate_cols = [col for col in safety_rate_cols if col in summary_runs.columns]
    if present_rate_cols:
        rates_input = summary_runs[group_cols + present_rate_cols].copy()
        for col in present_rate_cols:
            rates_input[col] = rates_input[col].map(
                lambda value: (
                    np.nan
                    if pd.isna(value)
                    or (isinstance(value, str) and not value.strip())
                    else float(coerce_bool(value))
                )
            )
        rates = rates_input.groupby(group_cols)[present_rate_cols].mean().reset_index()
        rates = rates.rename(
            columns={col: f"{col}_rate" for col in present_rate_cols}
        )
        grouped = grouped.merge(rates, on=group_cols, how="left")

    grouped["primary_objective_metric"] = PRIMARY_OBJECTIVE_METRIC
    if "string_stability_metric" in summary_runs.columns:
        grouped["string_stability_metric"] = STRING_STABILITY_METRIC
    if "safety_min_gap_threshold" in summary_runs.columns:
        grouped["safety_min_gap_threshold"] = float(
            pd.to_numeric(summary_runs["safety_min_gap_threshold"], errors="coerce")
            .dropna()
            .iloc[0]
        )
    if "string_stability_threshold" in summary_runs.columns:
        grouped["string_stability_threshold"] = float(
            pd.to_numeric(summary_runs["string_stability_threshold"], errors="coerce")
            .dropna()
            .iloc[0]
        )

    path = out_dir / "summary_by_human_rate.csv"
    grouped.to_csv(path, index=False)
    return path


def plot_summary_metric(
    summary_by: pd.DataFrame,
    metric: str,
    out_dir: Path,
    title: str,
    ylabel: str,
    filename: str,
    modes: List[str],
) -> None:
    palette = {
        "baseline": "C0",
        "adaptive": "C1",
        "rl": "C2",
        "ppo": "C3",
        "sac": "C4",
    }
    plt.figure(figsize=(7, 4))
    for mode in modes:
        sub = summary_by[summary_by["mode"] == mode].sort_values("cav_share")
        x = sub["cav_share"]
        y = sub[f"{metric}_mean"]
        yerr = sub[f"{metric}_std"].fillna(0.0)
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=mode,
            color=palette.get(mode),
        )
    plt.xlabel("CAV share")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save_plot(out_dir / filename)


def plot_insight_metric_groups(
    summary_runs: pd.DataFrame,
    metric: str,
    out_dir: Path,
    title: str,
    ylabel: str,
    filename: str,
    human_rates: List[float],
    modes: List[str],
) -> None:
    """Plot one dot per run, grouped by (human_rate, mode)."""
    palette = {
        "baseline": "C0",
        "adaptive": "C1",
        "rl": "C2",
        "ppo": "C3",
        "sac": "C4",
    }
    subset = summary_runs[
        summary_runs["human_rate"].isin(human_rates) & summary_runs["mode"].isin(modes)
    ].copy()
    if subset.empty:
        return

    groups = [(hr, mode) for hr in human_rates for mode in modes]
    fig_width = max(8.0, 1.1 * len(groups))
    plt.figure(figsize=(fig_width, 4.8))

    xticks = []
    xlabels = []
    for hr_idx, human_rate in enumerate(human_rates):
        group_start = hr_idx * len(modes)
        group_end = group_start + len(modes) - 1
        plt.axvspan(group_start - 0.5, group_end + 0.5, color="0.97", zorder=0)

    for group_idx, (human_rate, mode) in enumerate(groups):
        group = subset[
            (subset["human_rate"] == human_rate) & (subset["mode"] == mode)
        ].sort_values("seed")
        if group.empty:
            continue

        values = group[metric].to_numpy(dtype=float)
        if len(values) == 1:
            x = np.array([group_idx], dtype=float)
        else:
            x = group_idx + np.linspace(-0.12, 0.12, len(values))
        plt.scatter(
            x,
            values,
            s=55,
            alpha=0.85,
            color=palette.get(mode),
            edgecolors="white",
            linewidth=0.6,
            zorder=3,
        )
        plt.hlines(
            float(values.mean()),
            group_idx - 0.18,
            group_idx + 0.18,
            colors=palette.get(mode),
            linewidth=2.0,
            zorder=4,
        )
        xticks.append(group_idx)
        xlabels.append(f"HR={human_rate_key(human_rate)}\n{mode}")

    plt.xticks(xticks, xlabels, fontsize=9)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.figtext(
        0.99,
        0.01,
        "Each dot is one run; short horizontal bars show the mean over seeds.",
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.4",
    )
    _save_plot(out_dir / filename)


def plot_insight_tradeoff(
    summary_runs: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    out_dir: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    human_rates: List[float],
    modes: List[str],
) -> None:
    """Scatter all runs in a 2-metric tradeoff plane."""
    palette = {
        "baseline": "C0",
        "adaptive": "C1",
        "rl": "C2",
        "ppo": "C3",
        "sac": "C4",
    }
    marker_map = {
        1.0: "o",
        0.75: "s",
        0.5: "^",
        0.25: "D",
        0.0: "P",
    }
    subset = summary_runs[
        summary_runs["human_rate"].isin(human_rates) & summary_runs["mode"].isin(modes)
    ].copy()
    if subset.empty:
        return

    plt.figure(figsize=(8.2, 5.6))
    for mode in modes:
        for human_rate in human_rates:
            group = subset[
                (subset["mode"] == mode) & (subset["human_rate"] == human_rate)
            ]
            if group.empty:
                continue
            plt.scatter(
                group[x_metric],
                group[y_metric],
                s=70,
                alpha=0.82,
                color=palette.get(mode),
                marker=marker_map.get(human_rate, "o"),
                edgecolors="white",
                linewidth=0.6,
            )

    mode_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=palette.get(mode),
            markeredgecolor="white",
            markersize=8,
            label=mode,
        )
        for mode in modes
    ]
    hr_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(human_rate, "o"),
            linestyle="None",
            color="0.35",
            markersize=8,
            label=f"HR={human_rate_key(human_rate)}",
        )
        for human_rate in human_rates
    ]
    ax = plt.gca()
    legend_modes = ax.legend(handles=mode_handles, title="Mode", loc="upper left")
    ax.add_artist(legend_modes)
    ax.legend(handles=hr_handles, title="Human rate", loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    _save_plot(out_dir / filename)


def plot_insight_heatmap(
    summary_runs: pd.DataFrame,
    metric: str,
    out_dir: Path,
    title: str,
    filename: str,
    human_rates: List[float],
    modes: List[str],
    seeds: List[int],
    colorbar_label: str | None = None,
) -> None:
    """Heatmap of a metric across (human_rate, mode) rows and seed columns."""
    subset = summary_runs[
        summary_runs["human_rate"].isin(human_rates)
        & summary_runs["mode"].isin(modes)
        & summary_runs["seed"].isin(seeds)
    ].copy()
    if subset.empty:
        return

    row_index = pd.MultiIndex.from_product([human_rates, modes], names=["human_rate", "mode"])
    pivot = subset.pivot_table(index=["human_rate", "mode"], columns="seed", values=metric)
    pivot = pivot.reindex(index=row_index, columns=seeds)

    values = pivot.to_numpy(dtype=float)
    plt.figure(figsize=(max(5.0, 1.15 * len(seeds) + 2.0), max(4.2, 0.45 * len(row_index) + 1.0)))
    im = plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label=colorbar_label or metric)
    plt.xticks(np.arange(len(seeds)), [f"seed {seed}" for seed in seeds])
    ylabels = [
        f"HR={human_rate_key(human_rate)} | {mode}"
        for human_rate, mode in pivot.index.to_list()
    ]
    plt.yticks(np.arange(len(ylabels)), ylabels)

    mean_val = float(np.nanmean(values)) if values.size else 0.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            val = values[row_idx, col_idx]
            if np.isnan(val):
                text = "NA"
                text_color = "black"
            else:
                text = f"{val:.2f}"
                text_color = "white" if val >= mean_val else "black"
            plt.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    plt.title(title)
    _save_plot(out_dir / filename)


def write_experiment_manifest(
    out_dir: Path,
    human_rates: List[float],
    modes: List[str],
    seeds: List[int],
    ppo_ckpt_template: Optional[str],
    sac_ckpt_template: Optional[str],
    ppo_ckpt_overrides: Dict[str, str],
    sac_ckpt_overrides: Dict[str, str],
) -> Path:
    """Persist the exact sweep settings for reproducibility."""
    manifest = {
        "human_rates": human_rates,
        "modes": modes,
        "seeds": seeds,
        "ppo_checkpoint_template": ppo_ckpt_template,
        "sac_checkpoint_template": sac_ckpt_template,
        "ppo_checkpoint_overrides": ppo_ckpt_overrides,
        "sac_checkpoint_overrides": sac_ckpt_overrides,
        "string_stability": string_stability_report_metadata(),
    }
    path = out_dir / "experiment_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


# ----------------------------
# Main entry
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to base YAML config",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help=(
            "Controller modes to run. Choices: baseline adaptive rl ppo sac. "
            "'rl' is a backward-compatible alias for PPO."
        ),
    )
    parser.add_argument(
        "--human_rates",
        nargs="+",
        type=float,
        default=HUMAN_RATES,
        help="Human ratios to evaluate (Cartesian product with modes and seeds).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to evaluate (Cartesian product with human_rates and modes).",
    )
    parser.add_argument(
        "--ppo_checkpoint",
        type=str,
        default=DEFAULT_PPO_CHECKPOINT,
        help="Checkpoint used for PPO / rl evaluation modes.",
    )
    parser.add_argument(
        "--sac_checkpoint",
        type=str,
        default=DEFAULT_SAC_CHECKPOINT,
        help="Checkpoint used for SAC evaluation mode.",
    )
    parser.add_argument(
        "--ppo_checkpoint_template",
        type=str,
        default=None,
        help=(
            "Optional PPO checkpoint template with placeholders "
            "{human_rate}, {human_rate_tag}, {cav_share}, {cav_share_tag}."
        ),
    )
    parser.add_argument(
        "--sac_checkpoint_template",
        type=str,
        default=None,
        help=(
            "Optional SAC checkpoint template with placeholders "
            "{human_rate}, {human_rate_tag}, {cav_share}, {cav_share_tag}."
        ),
    )
    parser.add_argument(
        "--ppo_checkpoint_map",
        nargs="*",
        default=None,
        help="Optional per-HR PPO overrides of the form HR=PATH.",
    )
    parser.add_argument(
        "--sac_checkpoint_map",
        nargs="*",
        default=None,
        help="Optional per-HR SAC overrides of the form HR=PATH.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help="Output root directory for aggregated experiment results.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already have metrics.json and traces.csv in the target results directory.",
    )
    parser.add_argument(
        "--insight_plots",
        action="store_true",
        help="Generate detailed comparison plots from summary_runs.csv after the sweep.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    selected_modes = normalize_modes(args.modes)
    selected_human_rates = normalize_human_rates(args.human_rates)
    selected_seeds = normalize_seeds(args.seeds)
    ppo_ckpt_overrides = parse_hr_checkpoint_overrides(
        args.ppo_checkpoint_map, "PPO"
    )
    sac_ckpt_overrides = parse_hr_checkpoint_overrides(
        args.sac_checkpoint_map, "SAC"
    )
    base_config = load_base_config(args.config)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    write_experiment_manifest(
        results_root,
        human_rates=selected_human_rates,
        modes=selected_modes,
        seeds=selected_seeds,
        ppo_ckpt_template=args.ppo_checkpoint_template,
        sac_ckpt_template=args.sac_checkpoint_template,
        ppo_ckpt_overrides=ppo_ckpt_overrides,
        sac_ckpt_overrides=sac_ckpt_overrides,
    )

    for human_rate in selected_human_rates:
        cav_share = 1.0 - human_rate
        if cav_share <= 0.0:
            continue
        if any(mode in ("rl", "ppo") for mode in selected_modes):
            ppo_path = resolve_checkpoint_path(
                args.ppo_checkpoint,
                args.ppo_checkpoint_template,
                ppo_ckpt_overrides,
                human_rate,
            )
            if not os.path.exists(ppo_path):
                raise FileNotFoundError(
                    f"PPO checkpoint not found for human_rate={human_rate_key(human_rate)}: {ppo_path}"
                )
        if "sac" in selected_modes:
            sac_path = resolve_checkpoint_path(
                args.sac_checkpoint,
                args.sac_checkpoint_template,
                sac_ckpt_overrides,
                human_rate,
            )
            if not os.path.exists(sac_path):
                raise FileNotFoundError(
                    f"SAC checkpoint not found for human_rate={human_rate_key(human_rate)}: {sac_path}"
                )

    summary_rows: List[Dict] = []
    total_runs = (
        len(selected_human_rates) * len(selected_modes) * len(selected_seeds)
    )
    completed_runs = 0
    case_durations: List[float] = []
    sweep_start = time.time()

    for human_rate in selected_human_rates:
        for mode in selected_modes:
            for seed in selected_seeds:
                case_start = time.time()
                case_number = completed_runs + 1
                run_dir = (
                    results_root / f"HR_{human_rate}" / f"MODE_{mode}" / f"seed_{seed}"
                )
                print(
                    f"[RUN {case_number}/{total_runs}] "
                    f"HR={human_rate} mode={mode} seed={seed} -> {run_dir}",
                    flush=True,
                )

                metrics_path = run_dir / "metrics.json"
                traces_path = run_dir / "traces.csv"
                if args.skip_existing and metrics_path.exists() and traces_path.exists():
                    print(f"[SKIP] Existing results found for {run_dir}", flush=True)
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    if (
                        "primary_objective_value" not in metrics
                        or PRIMARY_OBJECTIVE_METRIC not in metrics
                    ):
                        print(
                            "[SKIP] Backfilling canonical objective metrics for "
                            f"{run_dir}",
                            flush=True,
                        )
                        metadata_path = run_dir / "metadata.json"
                        traces = pd.read_csv(traces_path)
                        metadata = None
                        if metadata_path.exists():
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                metadata = json.load(f)
                        backfilled = compute_metrics(
                            traces,
                            dt=float(base_config["dt"]),
                            mode=mode,
                            metadata=metadata,
                        )
                        metrics.update(backfilled)
                        with open(metrics_path, "w", encoding="utf-8") as f:
                            json.dump(metrics, f, indent=2)
                    row = {
                        "human_rate": human_rate,
                        "cav_share": 1.0 - human_rate,
                        "mode": mode,
                        "seed": seed,
                    }
                    row.update(metrics)
                    summary_rows.append(row)
                    if not metrics.get("string_stability_valid", True):
                        print(
                            f"[WARN] Existing run marked invalid for string stability: {run_dir}",
                            flush=True,
                        )
                    completed_runs += 1
                    elapsed = time.time() - sweep_start
                    case_durations.append(time.time() - case_start)
                    avg_case = elapsed / max(completed_runs, 1)
                    remaining = total_runs - completed_runs
                    eta = avg_case * remaining
                    print(
                        f"[PROGRESS] completed={completed_runs}/{total_runs} "
                        f"elapsed={format_duration(elapsed)} "
                        f"eta={format_duration(eta)}",
                        flush=True,
                    )
                    continue

                cfg = override_config(base_config, human_rate, mode, seed)
                print("[STAGE] simulation", flush=True)
                if mode in ("rl", "ppo"):
                    ppo_path = resolve_checkpoint_path(
                        args.ppo_checkpoint,
                        args.ppo_checkpoint_template,
                        ppo_ckpt_overrides,
                        human_rate,
                    )
                    run_policy_simulation(cfg, run_dir, ppo_path, algorithm="ppo")
                elif mode == "sac":
                    sac_path = resolve_checkpoint_path(
                        args.sac_checkpoint,
                        args.sac_checkpoint_template,
                        sac_ckpt_overrides,
                        human_rate,
                    )
                    run_policy_simulation(cfg, run_dir, sac_path, algorithm="sac")
                else:
                    run_single_simulation(cfg, run_dir)

                micro_csv = run_dir / "micro.csv"
                metadata_json = run_dir / "metadata.json"
                print("[STAGE] trace post-processing", flush=True)
                traces = build_traces(micro_csv, metadata_json, mode)
                traces.to_csv(run_dir / "traces.csv", index=False)

                dt = float(cfg["dt"])
                with open(metadata_json, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metrics = compute_metrics(traces, dt, mode, metadata=metadata)
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                print("[STAGE] plot generation", flush=True)
                generate_all_from_directory(run_dir)

                row = {
                    "human_rate": human_rate,
                    "cav_share": 1.0 - human_rate,
                    "mode": mode,
                    "seed": seed,
                }
                row.update(metrics)
                summary_rows.append(row)
                if not metrics.get("string_stability_valid", True):
                    print(
                        f"[WARN] Collision clamps detected; "
                        f"string-stability metrics are invalid for {run_dir}",
                        flush=True,
                    )
                completed_runs += 1
                case_duration = time.time() - case_start
                case_durations.append(case_duration)
                elapsed = time.time() - sweep_start
                avg_case = elapsed / max(completed_runs, 1)
                remaining = total_runs - completed_runs
                eta = avg_case * remaining
                print(
                    f"[DONE {completed_runs}/{total_runs}] "
                    f"duration={format_duration(case_duration)} "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)}",
                    flush=True,
                )

    print("[SUMMARY] Writing aggregate tables", flush=True)
    summary_runs_path = write_summary_runs(summary_rows, results_root)
    summary_runs = pd.read_csv(summary_runs_path)
    summary_by_path = write_summary_by_human_rate(summary_runs, results_root)
    summary_by = pd.read_csv(summary_by_path)

    summary_plots_dir = results_root / "summary_plots"
    summary_plots_dir.mkdir(parents=True, exist_ok=True)

    print("[SUMMARY] Generating aggregate plots", flush=True)
    generate_comparison_report(summary_runs_path, summary_plots_dir)

    if args.insight_plots:
        insight_dir = results_root / "summary_insights"
        insight_dir.mkdir(parents=True, exist_ok=True)
        if "primary_objective_value" in summary_runs.columns:
            plot_insight_metric_groups(
                summary_runs,
                metric="primary_objective_value",
                out_dir=insight_dir,
                title=f"Per-Run Primary Objective ({PRIMARY_OBJECTIVE_METRIC})",
                ylabel="Mean speed over last 100 steps [m/s]",
                filename="all_runs_primary_objective.png",
                human_rates=selected_human_rates,
                modes=selected_modes,
            )
        plot_insight_metric_groups(
            summary_runs,
            metric="speed_var_global",
            out_dir=insight_dir,
            title="Per-Run Speed Variance",
            ylabel="Speed Variance [m^2/s^2]",
            filename="all_runs_speed_variance.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
        )
        plot_insight_metric_groups(
            summary_runs,
            metric="min_gap",
            out_dir=insight_dir,
            title="Per-Run Minimum Gap",
            ylabel="Minimum Gap [m]",
            filename="all_runs_min_gap.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
        )
        if (
            "string_stability_value" in summary_runs.columns
            and pd.to_numeric(summary_runs["string_stability_value"], errors="coerce")
            .notna()
            .any()
        ):
            plot_insight_metric_groups(
                summary_runs,
                metric="string_stability_value",
                out_dir=insight_dir,
                title=f"Per-Run String Stability ({STRING_STABILITY_METRIC})",
                ylabel="Amplification factor",
                filename="all_runs_string_stability.png",
                human_rates=selected_human_rates,
                modes=selected_modes,
            )
        if "primary_objective_value" in summary_runs.columns:
            plot_insight_tradeoff(
                summary_runs,
                x_metric="primary_objective_value",
                y_metric="speed_var_global",
                out_dir=insight_dir,
                title=f"Primary Objective ({PRIMARY_OBJECTIVE_METRIC}) vs Stability",
                xlabel="Mean speed over last 100 steps [m/s]",
                ylabel="Speed Variance [m^2/s^2]",
                filename="tradeoff_primary_objective_vs_variance.png",
                human_rates=selected_human_rates,
                modes=selected_modes,
            )
            if (
                "string_stability_value" in summary_runs.columns
                and pd.to_numeric(summary_runs["string_stability_value"], errors="coerce")
                .notna()
                .any()
            ):
                plot_insight_tradeoff(
                    summary_runs,
                    x_metric="primary_objective_value",
                    y_metric="string_stability_value",
                    out_dir=insight_dir,
                    title=f"Primary Objective ({PRIMARY_OBJECTIVE_METRIC}) vs String Stability",
                    xlabel="Mean speed over last 100 steps [m/s]",
                    ylabel="Amplification factor",
                    filename="tradeoff_primary_objective_vs_string_stability.png",
                    human_rates=selected_human_rates,
                    modes=selected_modes,
                )
        plot_insight_tradeoff(
            summary_runs,
            x_metric="mean_speed",
            y_metric="speed_var_global",
            out_dir=insight_dir,
            title="Speed vs Stability Tradeoff",
            xlabel="Mean Speed [m/s]",
            ylabel="Speed Variance [m^2/s^2]",
            filename="tradeoff_speed_vs_variance.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
        )
        plot_insight_tradeoff(
            summary_runs,
            x_metric="min_gap",
            y_metric="rms_jerk",
            out_dir=insight_dir,
            title="Safety vs Comfort Tradeoff",
            xlabel="Minimum Gap [m]",
            ylabel="RMS Jerk [m/s^3]",
            filename="tradeoff_gap_vs_jerk.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
        )
        if "primary_objective_value" in summary_runs.columns:
            plot_insight_heatmap(
                summary_runs,
                metric="primary_objective_value",
                out_dir=insight_dir,
                title=f"Seed-to-Seed Primary Objective ({PRIMARY_OBJECTIVE_METRIC})",
                filename="heatmap_primary_objective.png",
                human_rates=selected_human_rates,
                modes=selected_modes,
                seeds=selected_seeds,
                colorbar_label="Mean speed over last 100 steps [m/s]",
            )
        plot_insight_heatmap(
            summary_runs,
            metric="speed_var_global",
            out_dir=insight_dir,
            title="Seed-to-Seed Speed Variance",
            filename="heatmap_speed_variance.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
            seeds=selected_seeds,
        )
        plot_insight_heatmap(
            summary_runs,
            metric="mean_speed",
            out_dir=insight_dir,
            title="Seed-to-Seed Mean Speed",
            filename="heatmap_mean_speed.png",
            human_rates=selected_human_rates,
            modes=selected_modes,
            seeds=selected_seeds,
        )

    total_elapsed = time.time() - sweep_start
    print(
        f"[COMPLETE] runs={completed_runs}/{total_runs} "
        f"elapsed={format_duration(total_elapsed)} "
        f"results={results_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
