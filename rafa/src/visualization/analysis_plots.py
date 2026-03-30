"""Analysis plotting utilities for simulator runs, training logs, and comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils.primary_objective import (
    PRIMARY_OBJECTIVE_LABEL,
    PRIMARY_OBJECTIVE_METRIC,
)
from src.utils.string_stability_metrics import (
    STRING_STABILITY_LABEL,
    STRING_STABILITY_METRIC,
    STRING_STABILITY_THRESHOLD,
)


MODE_ORDER = ["baseline", "adaptive", "rl", "ppo", "sac"]
MODE_LABELS = {
    "baseline": "Baseline",
    "adaptive": "Adaptive",
    "rl": "RL",
    "ppo": "PPO",
    "sac": "SAC",
}
MODE_COLORS = {
    "baseline": "#2B6CB0",
    "adaptive": "#DD6B20",
    "rl": "#805AD5",
    "ppo": "#D53F8C",
    "sac": "#2F855A",
}
MODE_MARKERS = {
    "baseline": "o",
    "adaptive": "s",
    "rl": "^",
    "ppo": "P",
    "sac": "D",
}
VEHICLE_COLORS = {
    "HDV": "#E53E3E",
    "CAV": "#2F855A",
}
SHARE_LABELS = {
    0.0: "0%",
    0.25: "25%",
    0.5: "50%",
    0.75: "75%",
    1.0: "100%",
}


def configure_style() -> None:
    """Apply the shared plotting style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 240,
        }
    )


def _base_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D7DEE8", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="x", color="#EDF2F7", linewidth=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A0AEC0")
    ax.spines["bottom"].set_color("#A0AEC0")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in (".png", ".svg"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved.append(path)
    plt.close(fig)
    return saved


def _ensure_vehicle_type(df: pd.DataFrame) -> pd.DataFrame:
    if "vehicle_type" in df.columns:
        return df
    if "type" in df.columns:
        out = df.copy()
        out["vehicle_type"] = out["type"].replace({"CAVVehicle": "CAV", "HumanVehicle": "HDV"})
        return out
    return df


def _load_trace_frame(path: Path) -> pd.DataFrame:
    if path.is_dir():
        if (path / "traces.csv").exists():
            return pd.read_csv(path / "traces.csv")
        if (path / "micro.csv").exists():
            return pd.read_csv(path / "micro.csv")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".pt":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("Reading .pt trajectory files requires torch.") from exc
        records = torch.load(path)
        return pd.DataFrame(records)
    raise FileNotFoundError(f"Could not locate trajectory data at {path}")


def _infer_step_column(df: pd.DataFrame) -> str | None:
    for name in ("step", "global_step", "total_steps", "timesteps", "timestep", "iteration", "update"):
        if name in df.columns:
            return name
    return None


def _infer_reward_column(df: pd.DataFrame) -> str | None:
    for name in ("total_reward", "reward", "episode_reward", "avg_reward", "return"):
        if name in df.columns:
            return name
    return None


def plot_speed_traces(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot speed over time for all vehicles, colored by type."""
    configure_style()
    traces = _ensure_vehicle_type(traces)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for _, group in traces.groupby("vehicle_id"):
        vehicle_type = str(group.get("vehicle_type", pd.Series(["HDV"])).iloc[0])
        ax.plot(group["time"], group["v"], color=VEHICLE_COLORS.get(vehicle_type, "#4A5568"), alpha=0.72, linewidth=1.0)
    ax.set_title("Vehicle Speed Traces", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    legend_handles = [
        Line2D([0], [0], color=VEHICLE_COLORS["HDV"], linewidth=2, label="HDV"),
        Line2D([0], [0], color=VEHICLE_COLORS["CAV"], linewidth=2, label="CAV"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    return _save_figure(fig, output_dir, "speed_traces")


def plot_spacetime_heatmap(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot ordered vehicle speed as a space-time heatmap."""
    configure_style()
    t0 = traces["time"].min()
    order = traces[traces["time"] == t0].sort_values("x")["vehicle_id"].tolist()
    pivot = traces.pivot(index="time", columns="vehicle_id", values="v")
    pivot = pivot[[col for col in order if col in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    image = ax.imshow(
        pivot.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[pivot.index.min(), pivot.index.max(), 0, len(pivot.columns)],
    )
    ax.set_title("Spacetime Heatmap (Speed)", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vehicle order on ring")
    fig.colorbar(image, ax=ax, label="Speed [m/s]")
    return _save_figure(fig, output_dir, "spacetime_heatmap_speed")


def plot_gap_distribution_over_time(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot gap quantile bands over time."""
    if "gap_s" not in traces.columns:
        return []
    configure_style()
    quantiles = (
        traces.groupby("time")["gap_s"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    ax.fill_between(quantiles["time"], quantiles[0.1], quantiles[0.9], color="#BEE3F8", alpha=0.45, label="10-90%")
    ax.fill_between(quantiles["time"], quantiles[0.25], quantiles[0.75], color="#63B3ED", alpha=0.35, label="25-75%")
    ax.plot(quantiles["time"], quantiles[0.5], color="#1A365D", linewidth=2.0, label="median")
    ax.set_title("Gap Distribution Over Time", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gap [m]")
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "gap_distribution_over_time")


def plot_alpha_evolution(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot alpha evolution for CAVs."""
    if "alpha" not in traces.columns:
        return []
    configure_style()
    traces = _ensure_vehicle_type(traces)
    cav = traces[traces["vehicle_type"] == "CAV"].copy()
    if cav.empty:
        return []
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for _, group in cav.groupby("vehicle_id"):
        ax.plot(group["time"], group["alpha"], color=VEHICLE_COLORS["CAV"], alpha=0.7, linewidth=1.0)
    ax.set_title("Alpha Evolution for CAVs", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Alpha")
    return _save_figure(fig, output_dir, "alpha_traces")


def plot_acceleration_distribution(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot acceleration distributions for HDVs and CAVs."""
    if "a" not in traces.columns:
        return []
    configure_style()
    traces = _ensure_vehicle_type(traces)
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    bins = 28
    for vehicle_type in ("HDV", "CAV"):
        subset = traces[traces["vehicle_type"] == vehicle_type]["a"].to_numpy()
        if subset.size:
            ax.hist(subset, bins=bins, alpha=0.48, density=True, color=VEHICLE_COLORS[vehicle_type], label=vehicle_type)
    ax.set_title("Acceleration Distribution", fontweight="bold")
    ax.set_xlabel("Acceleration [m/s^2]")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "acceleration_distribution")


def plot_speed_histograms(traces: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot per-vehicle speed histograms in a small-multiples grid."""
    configure_style()
    vehicle_ids = sorted(traces["vehicle_id"].unique().tolist())
    if not vehicle_ids:
        return []
    cols = min(4, len(vehicle_ids))
    rows = math.ceil(len(vehicle_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    traces = _ensure_vehicle_type(traces)
    for ax in axes_arr.flat:
        ax.set_visible(False)
    for ax, vehicle_id in zip(axes_arr.flat, vehicle_ids):
        group = traces[traces["vehicle_id"] == vehicle_id]
        vehicle_type = str(group.get("vehicle_type", pd.Series(["HDV"])).iloc[0])
        ax.set_visible(True)
        _base_axis_style(ax)
        ax.hist(group["v"], bins=20, color=VEHICLE_COLORS.get(vehicle_type, "#4A5568"), alpha=0.65)
        ax.set_title(f"Vehicle {vehicle_id} ({vehicle_type})", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Speed [m/s]")
        ax.set_ylabel("Count")
    return _save_figure(fig, output_dir, "per_vehicle_speed_histograms")


def plot_training_reward_curve(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the main reward/return curve if present."""
    step_col = _infer_step_column(log_df)
    reward_col = _infer_reward_column(log_df)
    if step_col is None or reward_col is None:
        return []
    configure_style()
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    ax.plot(log_df[step_col], log_df[reward_col], color=MODE_COLORS["sac"], linewidth=2.0)
    ax.set_title("Training Reward Curve", fontweight="bold")
    ax.set_xlabel(step_col.replace("_", " ").title())
    ax.set_ylabel(reward_col.replace("_", " ").title())
    return _save_figure(fig, output_dir, "training_reward_curve")


def plot_reward_components(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot available reward component columns."""
    step_col = _infer_step_column(log_df)
    component_cols = [col for col in log_df.columns if col.startswith("reward_") and pd.api.types.is_numeric_dtype(log_df[col])]
    if step_col is None or not component_cols:
        return []
    configure_style()
    fig, ax = plt.subplots(figsize=(10, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for col in component_cols:
        ax.plot(log_df[step_col], log_df[col], linewidth=1.6, label=col.replace("reward_", ""))
    ax.set_title("Reward Components", fontweight="bold")
    ax.set_xlabel(step_col.replace("_", " ").title())
    ax.set_ylabel("Value")
    ax.legend(frameon=False, ncol=min(4, len(component_cols)))
    return _save_figure(fig, output_dir, "reward_components")


def plot_learning_rate_schedule(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot learning-rate columns if present."""
    step_col = _infer_step_column(log_df)
    lr_cols = [col for col in log_df.columns if "lr" in col.lower() and pd.api.types.is_numeric_dtype(log_df[col])]
    if step_col is None or not lr_cols:
        return []
    configure_style()
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for col in lr_cols:
        ax.plot(log_df[step_col], log_df[col], linewidth=1.8, label=col)
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_xlabel(step_col.replace("_", " ").title())
    ax.set_ylabel("Learning rate")
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "learning_rate_schedule")


def plot_policy_entropy(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot entropy-like columns if present."""
    step_col = _infer_step_column(log_df)
    entropy_cols = [col for col in log_df.columns if "entropy" in col.lower() and pd.api.types.is_numeric_dtype(log_df[col])]
    if step_col is None or not entropy_cols:
        return []
    configure_style()
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for col in entropy_cols:
        ax.plot(log_df[step_col], log_df[col], linewidth=1.8, label=col)
    ax.set_title("Policy Entropy", fontweight="bold")
    ax.set_xlabel(step_col.replace("_", " ").title())
    ax.set_ylabel("Entropy")
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "policy_entropy")


def plot_training_diagnostics(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the main training scalars in a compact diagnostics grid."""
    step_col = _infer_step_column(log_df)
    if step_col is None:
        return []

    panels = [
        ("reward", "Reward", "Reward"),
        ("pg_loss", "Policy Loss", "Loss"),
        ("v_loss", "Value Loss", "Loss"),
        ("entropy", "Policy Entropy", "Entropy"),
        ("q1_loss", "Critic Loss (Q1)", "Loss"),
        ("actor_loss", "Actor Loss", "Loss"),
        ("alpha", "Entropy Temperature", "Alpha"),
        ("steps_per_second", "Throughput", "Steps / s"),
    ]
    available = [
        (column, title, ylabel)
        for column, title, ylabel in panels
        if column in log_df.columns
        and pd.to_numeric(log_df[column], errors="coerce").notna().any()
    ]
    if not available:
        return []

    configure_style()
    cols = 2
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.6 * cols, 3.6 * rows),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FBFCFE")
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.set_visible(False)

    for ax, (column, title, ylabel) in zip(axes_arr.flat, available):
        series = pd.to_numeric(log_df[column], errors="coerce")
        valid = series.notna()
        if not valid.any():
            continue
        ax.set_visible(True)
        _base_axis_style(ax)
        ax.plot(
            pd.to_numeric(log_df.loc[valid, step_col], errors="coerce"),
            series.loc[valid],
            linewidth=1.9,
            color="#1A365D",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(step_col.replace("_", " ").title())
        ax.set_ylabel(ylabel)
    return _save_figure(fig, output_dir, "training_diagnostics_grid")


def plot_eval_convergence(log_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot evaluation primary-objective convergence across human rates."""
    required = {"step", "human_rate", "eval_mean_speed"}
    if not required.issubset(log_df.columns):
        return []

    eval_df = log_df.copy()
    eval_df["step"] = pd.to_numeric(eval_df["step"], errors="coerce")
    eval_df["human_rate"] = pd.to_numeric(eval_df["human_rate"], errors="coerce")
    eval_df["eval_mean_speed"] = pd.to_numeric(
        eval_df["eval_mean_speed"], errors="coerce"
    )
    eval_df = eval_df.dropna(subset=["step", "human_rate", "eval_mean_speed"])
    if eval_df.empty:
        return []

    configure_style()
    fig, ax = plt.subplots(figsize=(10.4, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)

    unique_rates = sorted(eval_df["human_rate"].unique().tolist())
    cmap = plt.get_cmap("viridis")
    denom = max(len(unique_rates) - 1, 1)
    for index, human_rate in enumerate(unique_rates):
        subset = eval_df[eval_df["human_rate"] == human_rate].sort_values("step")
        cav_share = 1.0 - float(human_rate)
        color = cmap(index / denom)
        ax.plot(
            subset["step"],
            subset["eval_mean_speed"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=f"HR={human_rate:.2f} | CAV={cav_share:.2f}",
        )

    overall = (
        eval_df.groupby("step", observed=True)["eval_mean_speed"].mean().reset_index()
    )
    ax.plot(
        overall["step"],
        overall["eval_mean_speed"],
        color="#111827",
        linewidth=2.6,
        linestyle="--",
        label="Mean across human rates",
    )

    metric_name = PRIMARY_OBJECTIVE_METRIC
    if "primary_objective_metric" in eval_df.columns:
        candidates = eval_df["primary_objective_metric"].dropna().astype(str)
        if not candidates.empty:
            metric_name = candidates.iloc[0]

    ax.set_title(f"Evaluation Convergence ({metric_name})", fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"{PRIMARY_OBJECTIVE_LABEL} [m/s]")
    ax.legend(frameon=False, ncol=2)
    return _save_figure(fig, output_dir, "eval_convergence")


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mean_speed",
        "mean_speed_last100",
        "min_gap",
        "rms_jerk",
        "speed_var_global",
        "speed_std_time_mean",
        "oscillation_amplitude",
        "string_stability_value",
    ]
    if "primary_objective_value" in df.columns:
        metric_columns.append("primary_objective_value")
    metric_columns = [col for col in metric_columns if col in df.columns]
    agg = (
        df.groupby(["mode", "cav_share"], observed=True)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join(part for part in col if part).strip("_") for col in agg.columns.to_flat_index()]
    return agg


def _with_comparison_metric_aliases(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill comparison-plot metrics from alternate evaluation schemas."""
    df = summary_df.copy()
    aliases = {
        "mean_speed": ("mean_speed_all",),
        "min_gap": ("min_gap_episode",),
        "speed_var_global": ("speed_var_last100",),
    }
    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df[canonical] = pd.to_numeric(df[candidate], errors="coerce")
                break
    return df


def plot_overview_grid(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot a 2x2 grid of key metrics with mean and 1-sigma bands."""
    configure_style()
    agg = _aggregate_summary(summary_df)
    available_metrics = {
        col.removesuffix("_mean")
        for col in agg.columns
        if col.endswith("_mean")
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    panels = [
        ("mean_speed", "Mean Speed", "Speed [m/s]"),
        ("min_gap", "Minimum Gap", "Gap [m]"),
        ("rms_jerk", "RMS Jerk", "Jerk [m/s^3]"),
        ("speed_var_global", "Global Speed Variance", "Variance [m^2/s^2]"),
    ]
    plotted = 0
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        if metric not in available_metrics:
            continue
        ax.set_visible(True)
        _base_axis_style(ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("CAV share")
        ax.set_ylabel(ylabel)
        for mode in MODE_ORDER:
            grouped = agg[agg["mode"] == mode].sort_values("cav_share")
            if grouped.empty:
                continue
            x = grouped["cav_share"].to_numpy()
            mean = grouped[f"{metric}_mean"].to_numpy()
            std = np.nan_to_num(grouped[f"{metric}_std"].to_numpy())
            ax.fill_between(x, mean - std, mean + std, color=MODE_COLORS[mode], alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], linewidth=2.2, label=MODE_LABELS[mode])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []
    legend_axis = next((ax for ax in axes.flat if ax.get_visible()), None)
    handles, labels = (
        legend_axis.get_legend_handles_labels() if legend_axis is not None else ([], [])
    )
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    return _save_figure(fig, output_dir, "overview_metrics_grid")


def plot_tradeoff_scatter(summary_df: pd.DataFrame, output_dir: Path, x_metric: str, y_metric: str, stem: str, title: str, xlabel: str, ylabel: str) -> list[Path]:
    """Plot controller tradeoff scatter from aggregated summary data."""
    configure_style()
    agg = _aggregate_summary(summary_df)
    if f"{x_metric}_mean" not in agg.columns or f"{y_metric}_mean" not in agg.columns:
        return []
    fig, ax = plt.subplots(figsize=(10.4, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    for mode in MODE_ORDER:
        grouped = agg[agg["mode"] == mode].sort_values("cav_share")
        if grouped.empty:
            continue
        shares = grouped["cav_share"].to_numpy()
        x = grouped[f"{x_metric}_mean"].to_numpy()
        y = grouped[f"{y_metric}_mean"].to_numpy()
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if not finite_mask.any():
            continue
        shares = shares[finite_mask]
        x = x[finite_mask]
        y = y[finite_mask]
        ax.plot(x, y, color=MODE_COLORS[mode], linewidth=2.0, alpha=0.9)
        ax.scatter(x, y, color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], s=70 + 80 * shares, edgecolors="white", linewidths=0.8, label=MODE_LABELS[mode])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, stem)


def plot_metric_boxplot(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "mean_speed",
    *,
    stem: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> list[Path]:
    """Plot per-mode distributions grouped by CAV share."""
    if metric not in summary_df.columns:
        return []
    configure_style()
    fig, ax = plt.subplots(figsize=(12.2, 7.0), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    _base_axis_style(ax)
    shares = sorted(summary_df["cav_share"].unique())
    centers = np.arange(len(shares), dtype=float)
    width = 0.16
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODE_ORDER))
    for offset, mode in zip(offsets, MODE_ORDER):
        data = []
        for share in shares:
            values = summary_df[
                (summary_df["mode"] == mode) & (summary_df["cav_share"] == share)
            ][metric].to_numpy(dtype=float)
            data.append(values[np.isfinite(values)])
        if not any(values.size for values in data):
            continue
        positions = centers + offset
        bp = ax.boxplot(data, positions=positions, widths=width * 0.92, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(MODE_COLORS[mode])
            patch.set_alpha(0.28)
            patch.set_edgecolor(MODE_COLORS[mode])
        for pos, values in zip(positions, data):
            if values.size:
                jitter = np.linspace(-0.025, 0.025, len(values))
                ax.scatter(np.full_like(values, pos, dtype=float) + jitter, values, color=MODE_COLORS[mode], s=24, alpha=0.62, edgecolors="white", linewidths=0.3)
    ax.set_xticks(centers)
    ax.set_xticklabels([SHARE_LABELS.get(float(share), str(share)) for share in shares])
    ax.set_title(title or f"{metric.replace('_', ' ').title()} Distribution", fontweight="bold")
    ax.set_xlabel("CAV share")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    handles = [Line2D([0], [0], color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], linestyle="", markersize=8, label=MODE_LABELS[mode]) for mode in MODE_ORDER]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    return _save_figure(fig, output_dir, stem or f"{metric}_boxplot")


def plot_radar_chart(summary_df: pd.DataFrame, output_dir: Path, cav_share: float | None = None) -> list[Path]:
    """Plot a radar chart comparing controllers at a fixed CAV share."""
    configure_style()
    if cav_share is None:
        cav_share = float(sorted(summary_df["cav_share"].unique())[-1])
    subset = summary_df[summary_df["cav_share"] == cav_share]
    if subset.empty:
        return []
    metrics = ["mean_speed", "min_gap", "rms_jerk", "speed_var_global"]
    if not set(metrics).issubset(subset.columns):
        return []
    display = {
        "mean_speed": "Mean Speed",
        "min_gap": "Min Gap",
        "rms_jerk": "RMS Jerk",
        "speed_var_global": "Speed Var",
    }
    means = subset.groupby("mode")[metrics].mean()
    if means.empty:
        return []
    scaled = means.copy()
    for metric in metrics:
        values = means[metric].to_numpy(dtype=float)
        lo = values.min()
        hi = values.max()
        if hi - lo < 1e-9:
            scaled[metric] = 1.0
        elif metric in ("rms_jerk", "speed_var_global"):
            scaled[metric] = 1.0 - (means[metric] - lo) / (hi - lo)
        else:
            scaled[metric] = (means[metric] - lo) / (hi - lo)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.2, 7.4))
    fig.patch.set_facecolor("#FBFCFE")
    ax = fig.add_subplot(111, polar=True)
    for mode in MODE_ORDER:
        if mode not in scaled.index:
            continue
        values = scaled.loc[mode, metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, color=MODE_COLORS[mode], linewidth=2.0, label=MODE_LABELS[mode])
        ax.fill(angles, values, color=MODE_COLORS[mode], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([display[m] for m in metrics])
    ax.set_yticklabels([])
    ax.set_title(f"Controller radar at {SHARE_LABELS.get(float(cav_share), cav_share)} CAV share", fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.12))
    return _save_figure(fig, output_dir, f"radar_cav_share_{str(cav_share).replace('.', 'p')}")


def plot_mode_share_heatmap(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "mean_speed",
    *,
    stem: str | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
) -> list[Path]:
    """Plot a mode x CAV-share heatmap of a chosen metric."""
    if metric not in summary_df.columns:
        return []
    configure_style()
    pivot = summary_df.pivot_table(index="mode", columns="cav_share", values=metric, aggfunc="mean").reindex(index=[mode for mode in MODE_ORDER if mode in summary_df["mode"].unique()])
    if pivot.empty:
        return []
    if not np.isfinite(pivot.to_numpy(dtype=float)).any():
        return []
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([MODE_LABELS.get(mode, mode) for mode in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([SHARE_LABELS.get(float(col), str(col)) for col in pivot.columns])
    ax.set_title(title or f"{metric.replace('_', ' ').title()} Heatmap", fontweight="bold")
    ax.set_xlabel("CAV share")
    ax.set_ylabel("Mode")
    fig.colorbar(image, ax=ax, label=colorbar_label or metric.replace("_", " ").title())
    return _save_figure(fig, output_dir, stem or f"{metric}_heatmap")


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: str(value).strip().lower() in {"1", "1.0", "true", "yes", "on"}
    )


def _condition_title(shuffle_cav_positions: bool, perturbation_enabled: bool) -> str:
    layout = "Shuffled Layout" if shuffle_cav_positions else "Ordered Layout"
    perturbation = "Perturbation On" if perturbation_enabled else "Perturbation Off"
    return f"{layout} | {perturbation}"


def _ordered_modes(summary_df: pd.DataFrame) -> list[str]:
    present = summary_df["mode"].astype(str).unique().tolist()
    ordered = [mode for mode in MODE_ORDER if mode in present]
    ordered.extend(mode for mode in present if mode not in ordered)
    return ordered


def plot_generalization_metric_grid(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    value_col: str,
    stem: str,
    title: str,
    ylabel: str,
    std_col: str | None = None,
    y_limits: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot human-rate curves faceted by shuffle/perturbation condition."""
    required = {"mode", "human_rate", "shuffle_cav_positions", "perturbation_enabled", value_col}
    if not required.issubset(summary_df.columns):
        return []

    configure_style()
    df = summary_df.copy()
    df["shuffle_cav_positions"] = _coerce_bool_series(df["shuffle_cav_positions"])
    df["perturbation_enabled"] = _coerce_bool_series(df["perturbation_enabled"])

    shuffles = sorted(df["shuffle_cav_positions"].unique().tolist())
    perturbations = sorted(df["perturbation_enabled"].unique().tolist())
    human_rates = sorted(df["human_rate"].astype(float).unique().tolist())
    modes = _ordered_modes(df)

    fig, axes = plt.subplots(
        len(perturbations),
        len(shuffles),
        figsize=(5.9 * len(shuffles), 4.4 * len(perturbations)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("#FBFCFE")

    for row_idx, perturbation_enabled in enumerate(perturbations):
        for col_idx, shuffle_cav_positions in enumerate(shuffles):
            ax = axes[row_idx, col_idx]
            _base_axis_style(ax)
            subset = df[
                (df["shuffle_cav_positions"] == shuffle_cav_positions)
                & (df["perturbation_enabled"] == perturbation_enabled)
            ]
            for mode in modes:
                grouped = subset[subset["mode"].astype(str) == mode].sort_values("human_rate")
                if grouped.empty:
                    continue
                x = grouped["human_rate"].to_numpy(dtype=float)
                y = grouped[value_col].to_numpy(dtype=float)
                finite_mask = np.isfinite(x) & np.isfinite(y)
                if not finite_mask.any():
                    continue
                x = x[finite_mask]
                y = y[finite_mask]
                if std_col and std_col in grouped.columns:
                    yerr = np.nan_to_num(grouped[std_col].to_numpy(dtype=float))[finite_mask]
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        color=MODE_COLORS.get(mode, "#4A5568"),
                        marker=MODE_MARKERS.get(mode, "o"),
                        linewidth=2.0,
                        capsize=3.5,
                        label=MODE_LABELS.get(mode, mode.title()),
                    )
                else:
                    ax.plot(
                        x,
                        y,
                        color=MODE_COLORS.get(mode, "#4A5568"),
                        marker=MODE_MARKERS.get(mode, "o"),
                        linewidth=2.0,
                        label=MODE_LABELS.get(mode, mode.title()),
                    )
            ax.set_title(
                _condition_title(
                    shuffle_cav_positions=shuffle_cav_positions,
                    perturbation_enabled=perturbation_enabled,
                ),
                fontweight="bold",
            )
            ax.set_xlabel("Human ratio")
            ax.set_ylabel(ylabel)
            ax.set_xticks(human_rates)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

    handles: list[Line2D] = []
    labels: list[str] = []
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=min(4, len(handles)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontweight="bold")
    return _save_figure(fig, output_dir, stem)


def plot_generalization_rl_heatmap(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    value_col: str,
    stem: str,
    title: str,
    colorbar_label: str,
) -> list[Path]:
    """Plot an RL-only heatmap over human ratio and evaluation condition."""
    required = {
        "mode",
        "human_rate",
        "shuffle_cav_positions",
        "perturbation_enabled",
        value_col,
    }
    if not required.issubset(summary_df.columns):
        return []

    configure_style()
    df = summary_df.copy()
    df["shuffle_cav_positions"] = _coerce_bool_series(df["shuffle_cav_positions"])
    df["perturbation_enabled"] = _coerce_bool_series(df["perturbation_enabled"])
    rl_df = df[df["mode"].astype(str) == "rl"].copy()
    if rl_df.empty:
        return []

    shuffles = sorted(rl_df["shuffle_cav_positions"].unique().tolist())
    perturbations = sorted(rl_df["perturbation_enabled"].unique().tolist())
    condition_order = [
        _condition_title(shuffle, perturbation)
        for perturbation in perturbations
        for shuffle in shuffles
    ]
    rl_df["condition"] = rl_df.apply(
        lambda row: _condition_title(
            shuffle_cav_positions=bool(row["shuffle_cav_positions"]),
            perturbation_enabled=bool(row["perturbation_enabled"]),
        ),
        axis=1,
    )
    pivot = (
        rl_df.pivot_table(
            index="human_rate",
            columns="condition",
            values=value_col,
            aggfunc="mean",
        )
        .reindex(index=sorted(rl_df["human_rate"].astype(float).unique().tolist()))
        .reindex(columns=condition_order)
    )
    if pivot.empty:
        return []

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{float(hr):.2f}" for hr in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel("Human ratio")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(image, ax=ax, label=colorbar_label)

    values = pivot.to_numpy(dtype=float)
    mean_val = float(np.nanmean(values)) if values.size else 0.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
                color = "black"
            else:
                text = f"{value:.2f}"
                color = "white" if value >= mean_val else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )
    return _save_figure(fig, output_dir, stem)


def _resolve_primary_objective(summary_df: pd.DataFrame) -> tuple[str, str, str]:
    metric_name = PRIMARY_OBJECTIVE_METRIC
    if "primary_objective_metric" in summary_df.columns:
        candidates = summary_df["primary_objective_metric"].dropna().astype(str)
        if not candidates.empty:
            metric_name = candidates.iloc[0]
    if "primary_objective_value" in summary_df.columns:
        return "primary_objective_value", metric_name, f"{PRIMARY_OBJECTIVE_LABEL} [m/s]"
    if "mean_speed_last100" in summary_df.columns:
        return "mean_speed_last100", metric_name, f"{PRIMARY_OBJECTIVE_LABEL} [m/s]"
    return "mean_speed", "mean_speed", "Mean Speed [m/s]"


def _has_finite_metric(summary_df: pd.DataFrame, column: str) -> bool:
    return column in summary_df.columns and bool(
        np.isfinite(pd.to_numeric(summary_df[column], errors="coerce")).any()
    )


def generate_generalization_report(
    summary_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate factor-aware plots for the formal RL generalization sweep."""
    summary_df = pd.read_csv(summary_csv)
    output_path = Path(output_dir)
    objective_metric = PRIMARY_OBJECTIVE_METRIC
    if "primary_objective_metric" in summary_df.columns:
        candidates = summary_df["primary_objective_metric"].dropna().astype(str)
        if not candidates.empty:
            objective_metric = candidates.iloc[0]

    saved: list[Path] = []
    saved += plot_generalization_metric_grid(
        summary_df,
        output_path,
        value_col="primary_objective_mean",
        std_col="primary_objective_std",
        stem="generalization_primary_objective_grid",
        title=f"Primary Objective Across Generalization Conditions ({objective_metric})",
        ylabel=f"{PRIMARY_OBJECTIVE_LABEL} [m/s]",
    )
    saved += plot_generalization_metric_grid(
        summary_df,
        output_path,
        value_col="min_gap_episode_mean",
        std_col="min_gap_episode_std",
        stem="generalization_min_gap_grid",
        title="Minimum Gap Across Generalization Conditions",
        ylabel="Minimum gap [m]",
    )
    saved += plot_generalization_metric_grid(
        summary_df,
        output_path,
        value_col="safety_constraint_satisfied_rate",
        stem="generalization_safety_rate_grid",
        title="Safety Pass Rate Across Generalization Conditions",
        ylabel="Safety pass rate",
        y_limits=(0.0, 1.05),
    )
    saved += plot_generalization_rl_heatmap(
        summary_df,
        output_path,
        value_col="primary_objective_mean",
        stem="generalization_rl_primary_objective_heatmap",
        title=f"RL Primary Objective Heatmap ({objective_metric})",
        colorbar_label=f"{PRIMARY_OBJECTIVE_LABEL} [m/s]",
    )
    saved += plot_generalization_rl_heatmap(
        summary_df,
        output_path,
        value_col="safety_constraint_satisfied_rate",
        stem="generalization_rl_safety_heatmap",
        title="RL Safety Pass Rate Heatmap",
        colorbar_label="Safety pass rate",
    )
    if "perturbation_enabled" in summary_df.columns:
        perturb_on_df = summary_df.copy()
        perturb_on_df["perturbation_enabled"] = _coerce_bool_series(
            perturb_on_df["perturbation_enabled"]
        )
        perturb_on_df = perturb_on_df[perturb_on_df["perturbation_enabled"]]
        if _has_finite_metric(perturb_on_df, "string_stability_value_mean"):
            saved += plot_generalization_metric_grid(
                perturb_on_df,
                output_path,
                value_col="string_stability_value_mean",
                std_col="string_stability_value_std",
                stem="generalization_string_stability_grid",
                title=f"String Stability Across Controlled Perturbation Cases ({STRING_STABILITY_METRIC})",
                ylabel="Amplification factor",
            )
            saved += plot_generalization_metric_grid(
                perturb_on_df,
                output_path,
                value_col="string_stability_metric_valid_rate",
                stem="generalization_string_stability_validity_grid",
                title="String-Stability Metric Validity Rate",
                ylabel="Validity rate",
                y_limits=(0.0, 1.05),
            )
            saved += plot_generalization_rl_heatmap(
                perturb_on_df,
                output_path,
                value_col="string_stability_value_mean",
                stem="generalization_rl_string_stability_heatmap",
                title=f"RL String Stability Heatmap ({STRING_STABILITY_METRIC})",
                colorbar_label="Amplification factor",
            )
    return saved


def generate_comparison_report(summary_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate the full suite of comparison plots from a summary CSV."""
    summary_df = _with_comparison_metric_aliases(pd.read_csv(summary_csv))
    if "mode" in summary_df.columns:
        summary_df["mode"] = pd.Categorical(summary_df["mode"], categories=MODE_ORDER, ordered=True)
    output_path = Path(output_dir)
    objective_col, objective_metric, objective_ylabel = _resolve_primary_objective(summary_df)
    saved: list[Path] = []
    saved += plot_overview_grid(summary_df, output_path)
    saved += plot_tradeoff_scatter(summary_df, output_path, "mean_speed", "speed_var_global", "tradeoff_speed_vs_variance", "Speed vs Stability Tradeoff", "Mean Speed [m/s]", "Global Speed Variance [m^2/s^2]")
    saved += plot_tradeoff_scatter(summary_df, output_path, "min_gap", "rms_jerk", "tradeoff_gap_vs_jerk", "Safety vs Comfort Tradeoff", "Minimum Gap [m]", "RMS Jerk [m/s^3]")
    saved += plot_metric_boxplot(summary_df, output_path, "mean_speed")
    saved += plot_radar_chart(summary_df, output_path)
    saved += plot_mode_share_heatmap(summary_df, output_path, "mean_speed")
    saved += plot_mode_share_heatmap(summary_df, output_path, "speed_var_global")
    if objective_col != "mean_speed":
        saved += plot_tradeoff_scatter(
            summary_df,
            output_path,
            objective_col,
            "speed_var_global",
            "tradeoff_primary_objective_vs_variance",
            f"Primary Objective ({objective_metric}) vs Stability",
            objective_ylabel,
            "Global Speed Variance [m^2/s^2]",
        )
        saved += plot_metric_boxplot(
            summary_df,
            output_path,
            objective_col,
            stem="primary_objective_boxplot",
            title=f"Primary Objective ({objective_metric}) Distribution",
            ylabel=objective_ylabel,
        )
        saved += plot_mode_share_heatmap(
            summary_df,
            output_path,
            objective_col,
            stem="primary_objective_heatmap",
            title=f"Primary Objective ({objective_metric}) Heatmap",
            colorbar_label=objective_ylabel,
        )
    if _has_finite_metric(summary_df, "string_stability_value"):
        saved += plot_tradeoff_scatter(
            summary_df,
            output_path,
            objective_col,
            "string_stability_value",
            "tradeoff_primary_objective_vs_string_stability",
            f"Primary Objective ({objective_metric}) vs String Stability",
            objective_ylabel,
            "Amplification factor",
        )
        saved += plot_metric_boxplot(
            summary_df,
            output_path,
            "string_stability_value",
            stem="string_stability_boxplot",
            title=f"String Stability ({STRING_STABILITY_METRIC}) Distribution",
            ylabel="Amplification factor",
        )
        saved += plot_mode_share_heatmap(
            summary_df,
            output_path,
            "string_stability_value",
            stem="string_stability_heatmap",
            title=f"String Stability ({STRING_STABILITY_METRIC}) Heatmap",
            colorbar_label="Amplification factor",
        )
    return saved


def generate_all_from_directory(run_dir: str | Path) -> list[Path]:
    """Autodetect available artifacts in a run directory and generate plots."""
    run_path = Path(run_dir)
    saved: list[Path] = []
    plots_dir = run_path / "plots"
    if (run_path / "gpu_eval_summary.csv").exists():
        saved += generate_generalization_report(run_path / "gpu_eval_summary.csv", plots_dir)
    elif (run_path / "summary_runs.csv").exists():
        saved += generate_comparison_report(run_path / "summary_runs.csv", plots_dir)
    else:
        traces_source = None
        for candidate in ("traces.csv", "micro.csv", "micro.pt"):
            if (run_path / candidate).exists():
                traces_source = run_path / candidate
                break
        if traces_source is not None:
            traces = _load_trace_frame(traces_source)
            if "t" in traces.columns and "time" not in traces.columns:
                traces = traces.rename(columns={"t": "time", "id": "vehicle_id"})
            traces = _ensure_vehicle_type(traces)
            saved += plot_speed_traces(traces, plots_dir)
            saved += plot_spacetime_heatmap(traces, plots_dir)
            saved += plot_gap_distribution_over_time(traces, plots_dir)
            saved += plot_alpha_evolution(traces, plots_dir)
            saved += plot_acceleration_distribution(traces, plots_dir)
            saved += plot_speed_histograms(traces, plots_dir)
        training_log = run_path / "training_log.csv"
        if training_log.exists():
            log_df = pd.read_csv(training_log)
            saved += plot_training_reward_curve(log_df, plots_dir)
            saved += plot_training_diagnostics(log_df, plots_dir)
            saved += plot_reward_components(log_df, plots_dir)
            saved += plot_learning_rate_schedule(log_df, plots_dir)
            saved += plot_policy_entropy(log_df, plots_dir)
        eval_log = run_path / "eval_log.csv"
        if eval_log.exists():
            eval_df = pd.read_csv(eval_log)
            saved += plot_eval_convergence(eval_df, plots_dir)
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate analysis plots from run directories or summary CSV files.")
    parser.add_argument("path", nargs="?", default=".", help="Run directory or summary CSV")
    parser.add_argument("--output_dir", default=None, help="Optional explicit output directory")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        saved_paths = generate_comparison_report(target, Path(args.output_dir) if args.output_dir else target.parent / "summary_plots")
    else:
        saved_paths = generate_all_from_directory(target)
        if args.output_dir and saved_paths:
            # Output directory overrides are handled in direct API calls; keep CLI simple.
            pass

    for path in saved_paths:
        print(path)
