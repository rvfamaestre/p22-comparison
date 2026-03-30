"""Generate clean, presentation-ready plots for a general audience.

Usage:
    python generate_presentation_plots.py \
        --summary_csv output/plot_smoke_input/summary_runs.csv \
        --output_dir output/presentation_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Visual identity ──────────────────────────────────────────────────
PALETTE = {
    "baseline": "#5B8DEE",   # calm blue
    "adaptive": "#F5A623",   # warm amber
    "sac":      "#2ECC71",   # fresh green
    "ppo":      "#E74C8B",   # magenta
    "rl":       "#9B59B6",   # purple
}
MODE_NICE = {
    "baseline": "Baseline (IDM)",
    "adaptive": "Adaptive",
    "sac":      "SAC  (ours)",
    "ppo":      "PPO",
    "rl":       "RL",
}
CAV_NICE = {0.25: "25 % CAVs", 0.5: "50 % CAVs", 0.75: "75 % CAVs", 1.0: "100 % CAVs"}

# Metrics we care about, with audience-friendly labels & "higher is better" flag
METRICS = {
    "mean_speed":      ("Mean Speed (m/s)",            True),
    "min_gap":         ("Safety Gap (m)",               True),
    "rms_jerk":        ("Ride Smoothness (jerk)",       False),  # lower = better
    "speed_var_global": ("Speed Consistency (variance)", False),  # lower = better
}


def _presentation_style() -> None:
    """Large fonts, white background, no clutter – ready for beamer / slides."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(out / f"{stem}{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {stem}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 1 – Grouped bar chart: one subplot per metric
# ═══════════════════════════════════════════════════════════════════════
def plot_grouped_bars(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart – one panel per metric, grouped by CAV share."""
    _presentation_style()

    agg = df.groupby(["mode", "cav_share"]).agg("mean").reset_index()
    modes = [m for m in ["baseline", "adaptive", "sac", "ppo", "rl"] if m in agg["mode"].unique()]
    shares = sorted(agg["cav_share"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)

    for ax, (metric, (label, higher_better)) in zip(axes, METRICS.items()):
        x = np.arange(len(shares))
        width = 0.8 / len(modes)
        for i, mode in enumerate(modes):
            sub = agg[agg["mode"] == mode]
            vals = [sub[sub["cav_share"] == s][metric].values[0] for s in shares]
            bars = ax.bar(
                x + i * width - 0.4 + width / 2,
                vals,
                width * 0.88,
                label=MODE_NICE.get(mode, mode),
                color=PALETTE[mode],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            # value labels on top
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.001,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10, color="#444",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([CAV_NICE.get(s, f"{s:.0%}") for s in shares])
        ax.set_title(label)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)

        # Arrow indicating "better" direction
        arrow = "↑ higher = better" if higher_better else "↓ lower = better"
        ax.annotate(
            arrow, xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=9, color="#888", va="top",
        )

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Controller Performance across CAV Penetration Rates",
                 fontsize=20, fontweight="bold", y=1.02)
    _save(fig, out, "bars_all_metrics")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2 – Radar / spider chart  (bigger = better, inverted where needed)
# ═══════════════════════════════════════════════════════════════════════
def plot_radar(df: pd.DataFrame, out: Path, cav_share: float = 0.75) -> None:
    """Radar chart at a given CAV share – all axes normalised so bigger = better."""
    _presentation_style()

    agg = df[df["cav_share"] == cav_share].groupby("mode").mean(numeric_only=True)
    modes = [m for m in ["baseline", "adaptive", "sac", "ppo", "rl"] if m in agg.index]

    labels_short = []
    values_per_mode: dict[str, list[float]] = {m: [] for m in modes}

    for metric, (label, higher_better) in METRICS.items():
        short = label.split("(")[0].strip()
        labels_short.append(short)
        col = agg[metric]
        lo, hi = col.min(), col.max()
        span = hi - lo if hi != lo else 1.0
        for m in modes:
            raw = col[m]
            normed = (raw - lo) / span  # 0..1
            if not higher_better:
                normed = 1.0 - normed  # flip so bigger = better
            values_per_mode[m].append(normed)

    n = len(labels_short)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for mode in modes:
        vals = values_per_mode[mode] + values_per_mode[mode][:1]
        ax.plot(angles, vals, linewidth=2.4, label=MODE_NICE.get(mode, mode),
                color=PALETTE[mode])
        ax.fill(angles, vals, alpha=0.12, color=PALETTE[mode])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_short, fontsize=14, fontweight="bold")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", "best"], fontsize=9, color="#aaa")
    ax.set_ylim(0, 1.1)
    ax.spines["polar"].set_color("#ddd")
    ax.grid(color="#e0e0e0")

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    ax.set_title(f"Normalised Performance at {CAV_NICE.get(cav_share, f'{cav_share:.0%}')}",
                 fontsize=18, fontweight="bold", pad=24)
    _save(fig, out, f"radar_{int(cav_share*100)}pct")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3 – Improvement summary (% gain of SAC over Baseline)
# ═══════════════════════════════════════════════════════════════════════
def plot_improvement(df: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart showing SAC % improvement over Baseline at 75% CAV."""
    _presentation_style()

    agg = df[df["cav_share"] == 0.75].groupby("mode").mean(numeric_only=True)
    if "sac" not in agg.index or "baseline" not in agg.index:
        print("  [skip] Need both 'sac' and 'baseline' modes for improvement chart")
        return

    labels, pcts, colors = [], [], []
    for metric, (label, higher_better) in METRICS.items():
        base_val = agg.loc["baseline", metric]
        sac_val = agg.loc["sac", metric]
        if base_val == 0:
            continue
        pct = (sac_val - base_val) / abs(base_val) * 100
        if not higher_better:
            pct = -pct  # flip so positive = improvement
        labels.append(label.split("(")[0].strip())
        pcts.append(pct)
        colors.append("#2ECC71" if pct >= 0 else "#E74C3C")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, pcts, height=0.55, color=colors, edgecolor="white", linewidth=1, zorder=3)

    for bar, pct in zip(bars, pcts):
        xpos = bar.get_width()
        align = "left" if xpos >= 0 else "right"
        offset = 0.3 if xpos >= 0 else -0.3
        ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                f"{pct:+.1f} %", va="center", ha=align, fontsize=13, fontweight="bold",
                color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.set_xlabel("Improvement over Baseline (%)")
    ax.set_title("SAC Improvement at 75 % CAV Penetration",
                 fontsize=18, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.invert_yaxis()
    _save(fig, out, "improvement_sac_vs_baseline")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4 – Speed vs Stability scatter (cleaner version)
# ═══════════════════════════════════════════════════════════════════════
def plot_speed_stability(df: pd.DataFrame, out: Path) -> None:
    """Clean scatter: Mean Speed vs Speed Variance with arrows showing CAV effect."""
    _presentation_style()

    agg = df.groupby(["mode", "cav_share"]).mean(numeric_only=True).reset_index()
    modes = [m for m in ["baseline", "adaptive", "sac", "ppo", "rl"] if m in agg["mode"].unique()]
    shares = sorted(agg["cav_share"].unique())

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    for mode in modes:
        sub = agg[agg["mode"] == mode].sort_values("cav_share")
        xs = sub["mean_speed"].values
        ys = sub["speed_var_global"].values
        color = PALETTE[mode]

        # draw arrow from low CAV → high CAV
        if len(xs) >= 2:
            ax.annotate(
                "", xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16),
                zorder=3,
            )
        # plot endpoints
        for j, s in enumerate(shares):
            row = sub[sub["cav_share"] == s]
            marker_size = 110 if j == len(shares) - 1 else 70
            ax.scatter(row["mean_speed"], row["speed_var_global"],
                       s=marker_size, color=color, edgecolors="white", linewidths=1.2,
                       zorder=4, label=MODE_NICE.get(mode, mode) if j == 0 else None)
            if j == len(shares) - 1:
                ax.annotate(
                    CAV_NICE.get(s, f"{s:.0%}"),
                    (row["mean_speed"].values[0] + 0.03, row["speed_var_global"].values[0]),
                    fontsize=10, color="#666",
                )

    # ideal region shading
    ax.annotate("← lower variance = smoother", xy=(0.5, 0.01), xycoords="axes fraction",
                fontsize=10, color="#999", ha="center")
    ax.annotate("higher speed = faster →", xy=(0.99, 0.5), xycoords="axes fraction",
                fontsize=10, color="#999", ha="right", rotation=90, va="center")

    ax.set_xlabel("Mean Speed (m/s)")
    ax.set_ylabel("Speed Variance (m²/s²)")
    ax.set_title("Speed vs Flow Stability", fontsize=18, fontweight="bold")
    ax.legend(frameon=False, fontsize=13)
    ax.grid(color="#E8E8E8", linewidth=0.6)
    _save(fig, out, "speed_vs_stability")


# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-ready plots")
    parser.add_argument("--summary_csv", type=Path,
                        default=Path("output/plot_smoke_input/summary_runs.csv"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("output/presentation_plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    print(f"Loaded {len(df)} rows  •  modes: {sorted(df['mode'].unique())}")
    print(f"Saving to {args.output_dir}/\n")

    plot_grouped_bars(df, args.output_dir)
    # pick the highest CAV share available for the radar
    best_share = df["cav_share"].max()
    plot_radar(df, args.output_dir, cav_share=best_share)
    plot_improvement(df, args.output_dir)
    plot_speed_stability(df, args.output_dir)

    print(f"\nDone – all plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
