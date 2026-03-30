"""
Experiment Orchestration and Results Pipeline

Runs a sweep over human rates, controller modes, and seeds.
For each run, saves traces, metrics, and plots under Results/.
Generates aggregated summaries and plots across all runs.

"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.simulation.scenario_manager import ScenarioManager
from src.utils.config import get_default_config, merge_config, validate_config


HUMAN_RATES = [1.0, 0.75, 0.5, 0.25, 0.0]
MODES = ["baseline", "adaptive"]
SEEDS = [0, 1, 2, 3, 4]

RESULTS_DIR = "Results"


# ----------------------------
# Config helpers
# ----------------------------
def load_base_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)
    base_config = merge_config(user_config, get_default_config())
    validate_config(base_config)
    return base_config


def override_config(base: Dict, human_rate: float, mode: str, seed: int) -> Dict:
    cfg = copy.deepcopy(base)
    cfg["human_ratio"] = human_rate
    cfg["seed"] = seed

    # Ensure mesoscopic config exists
    if "mesoscopic" not in cfg or cfg["mesoscopic"] is None:
        cfg["mesoscopic"] = {}
    cfg["mesoscopic"]["enabled"] = (mode == "adaptive")

    # FORCE UNIFORM INITIAL CONDITIONS for scientific experiments
    # This ensures clean wave emergence and reproducible dynamics
    cfg["initial_conditions"] = "uniform"
    cfg["perturbation_enabled"] = True
    cfg["perturbation_vehicle"] = 0
    cfg["perturbation_time"] = 3.0         # 3 seconds of perfect uniformity
    cfg["perturbation_delta_v"] = -2.0
    cfg["noise_warmup_time"] = 3.0         # Suppress noise for first 3 seconds
    
    # DISABLE VISUALIZATION for batch experiments (much faster)
    cfg["enable_live_viz"] = False
    cfg["play_recording"] = False

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
    sim = manager.build()
    sim.run()

    # Save effective config
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

    if mode == "adaptive":
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
def compute_metrics(traces: pd.DataFrame, dt: float, mode: str) -> Dict[str, float]:
    v = traces["v"].to_numpy()
    a = traces["a"].to_numpy()

    metrics = {
        "speed_var_global": float(np.var(v)),
        "speed_std_time_mean": float(
            traces.groupby("time")["v"].std().mean()
        ),
        "oscillation_amplitude": float(v.max() - v.min()),
        "min_gap": float(traces["gap_s"].min()),
        "rms_acc": float(np.sqrt(np.mean(a ** 2))),
        "mean_speed": float(np.mean(v)),
    }

    # RMS jerk per vehicle
    jerk_vals = []
    for _, g in traces.groupby("vehicle_id"):
        g_sorted = g.sort_values("time")
        jerk = np.diff(g_sorted["a"].to_numpy()) / dt
        if len(jerk) > 0:
            jerk_vals.append(np.mean(jerk ** 2))
    rms_jerk = float(np.sqrt(np.mean(jerk_vals))) if jerk_vals else 0.0
    metrics["rms_jerk"] = rms_jerk

    if mode == "adaptive":
        cav = traces[traces["vehicle_type"] == "CAV"]
        metrics["alpha_mean"] = float(cav["alpha"].mean())
        metrics["alpha_std"] = float(cav["alpha"].std())
        metrics["headway_effective_mean"] = float(cav["headway_hc_effective"].mean())

    return metrics


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
    order = (
        traces[traces["time"] == t0]
        .sort_values("x")["vehicle_id"]
        .tolist()
    )
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
    ]
    adaptive_cols = ["alpha_mean", "alpha_std", "headway_effective_mean"]
    for col in adaptive_cols:
        if col in summary_runs.columns:
            agg_cols.append(col)

    grouped = summary_runs.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index()

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
) -> None:
    plt.figure(figsize=(7, 4))
    for mode in MODES:
        sub = summary_by[summary_by["mode"] == mode].sort_values("cav_share")
        x = sub["cav_share"]
        y = sub[f"{metric}_mean"]
        yerr = sub[f"{metric}_std"]
        plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=mode)
    plt.xlabel("CAV share")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save_plot(out_dir / filename)


# ----------------------------
# Main entry
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to base YAML config",
    )
    args = parser.parse_args()

    base_config = load_base_config(args.config)
    results_root = Path(RESULTS_DIR)
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []

    for human_rate in HUMAN_RATES:
        for mode in MODES:
            for seed in SEEDS:
                run_dir = results_root / f"HR_{human_rate}" / f"MODE_{mode}" / f"seed_{seed}"
                print(f"[RUN] HR={human_rate} mode={mode} seed={seed} -> {run_dir}")

                cfg = override_config(base_config, human_rate, mode, seed)
                run_single_simulation(cfg, run_dir)

                micro_csv = run_dir / "micro.csv"
                metadata_json = run_dir / "metadata.json"
                traces = build_traces(micro_csv, metadata_json, mode)
                traces.to_csv(run_dir / "traces.csv", index=False)

                dt = float(cfg["dt"])
                metrics = compute_metrics(traces, dt, mode)
                with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                plot_speed_traces(traces, run_dir)
                plot_spacetime_heatmap(traces, run_dir)
                plot_headway_traces(traces, run_dir)
                if mode == "adaptive":
                    plot_alpha_traces(traces, run_dir)

                row = {
                    "human_rate": human_rate,
                    "cav_share": 1.0 - human_rate,
                    "mode": mode,
                    "seed": seed,
                }
                row.update(metrics)
                summary_rows.append(row)

    summary_runs_path = write_summary_runs(summary_rows, results_root)
    summary_runs = pd.read_csv(summary_runs_path)
    summary_by_path = write_summary_by_human_rate(summary_runs, results_root)
    summary_by = pd.read_csv(summary_by_path)

    summary_plots_dir = results_root / "summary_plots"
    summary_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_summary_metric(
        summary_by,
        metric="speed_std_time_mean",
        out_dir=summary_plots_dir,
        title="Speed Std (Mean over Time) vs CAV Share",
        ylabel="Speed Std [m/s]",
        filename="stability_vs_cavshare.png",
    )
    plot_summary_metric(
        summary_by,
        metric="speed_var_global",
        out_dir=summary_plots_dir,
        title="Speed Variance (Global) vs CAV Share",
        ylabel="Speed Variance [m^2/s^2]",
        filename="speedvar_vs_cavshare.png",
    )
    plot_summary_metric(
        summary_by,
        metric="min_gap",
        out_dir=summary_plots_dir,
        title="Minimum Gap vs CAV Share",
        ylabel="Min Gap [m]",
        filename="min_gap_vs_cavshare.png",
    )
    plot_summary_metric(
        summary_by,
        metric="rms_jerk",
        out_dir=summary_plots_dir,
        title="RMS Jerk vs CAV Share",
        ylabel="RMS Jerk [m/s^3]",
        filename="rms_jerk_vs_cavshare.png",
    )


if __name__ == "__main__":
    main()
