"""Rank evaluated GPU PPO checkpoints using saved summary CSV files.

Usage:
    python rank_gpu_checkpoints.py --eval_root output/hpc_gpu_train/evaluation
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Dict, Iterable, List

from src.utils.primary_objective import PRIMARY_OBJECTIVE_METRIC
from src.utils.string_stability_metrics import STRING_STABILITY_METRIC


def load_summary(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_metric(row: Dict[str, str], key: str, fallback: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        value = row[fallback]
    return float(value)


def sort_key_or_floor(value: float, floor: float = -1.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(floor)


def prefer_lower(value: float, fallback: float = float("inf")) -> float:
    return -sort_key_or_floor(value, floor=fallback)


def build_checkpoint_row(ckpt_name: str, rows: List[Dict[str, str]]) -> Dict[str, float | str] | None:
    by_mode = {mode: [] for mode in ("baseline", "adaptive", "rl")}
    for row in rows:
        by_mode[row["mode"]].append(row)

    if not by_mode["rl"] or not by_mode["adaptive"]:
        return None

    objective_metric = by_mode["rl"][0].get(
        "primary_objective_metric", PRIMARY_OBJECTIVE_METRIC
    )
    rl_objective = sum(
        summary_metric(r, "primary_objective_mean", "mean_speed_last100_mean")
        for r in by_mode["rl"]
    ) / len(by_mode["rl"])
    ad_objective = sum(
        summary_metric(
            r,
            "primary_objective_mean",
            "mean_speed_last100_mean",
        )
        for r in by_mode["adaptive"]
    ) / len(by_mode["adaptive"])
    rl_reward = sum(float(r["mean_reward_mean"]) for r in by_mode["rl"]) / len(by_mode["rl"])
    rl_min_gap = min(float(r["min_gap_episode_mean"]) for r in by_mode["rl"])
    rl_var = sum(float(r["speed_var_last100_mean"]) for r in by_mode["rl"]) / len(by_mode["rl"])
    rl_safety_rate = min(
        float(r.get("safety_constraint_satisfied_rate", 1.0))
        for r in by_mode["rl"]
    )
    rl_string_stability = sum(
        summary_metric(
            r,
            "string_stability_value_mean",
            "string_stability_value_mean",
        )
        for r in by_mode["rl"]
        if r.get("string_stability_value_mean") not in (None, "", "nan")
    )
    rl_string_stability_count = sum(
        1
        for r in by_mode["rl"]
        if r.get("string_stability_value_mean") not in (None, "", "nan")
    )
    rl_string_stability = (
        rl_string_stability / rl_string_stability_count
        if rl_string_stability_count > 0
        else float("nan")
    )
    rl_string_stability_valid_rate = min(
        float(r.get("string_stability_metric_valid_rate", 1.0))
        for r in by_mode["rl"]
        if r.get("string_stability_metric_valid_rate") not in (None, "", "nan")
    ) if any(
        r.get("string_stability_metric_valid_rate") not in (None, "", "nan")
        for r in by_mode["rl"]
    ) else float("nan")
    rl_string_stability_stable_rate = min(
        float(r.get("string_stability_is_stable_rate", 1.0))
        for r in by_mode["rl"]
        if r.get("string_stability_is_stable_rate") not in (None, "", "nan")
    ) if any(
        r.get("string_stability_is_stable_rate") not in (None, "", "nan")
        for r in by_mode["rl"]
    ) else float("nan")

    return {
        "checkpoint": ckpt_name,
        "primary_objective_metric": objective_metric,
        "string_stability_metric": STRING_STABILITY_METRIC,
        "rl_primary_objective_avg": rl_objective,
        "adaptive_primary_objective_avg": ad_objective,
        "primary_objective_delta_vs_adaptive": rl_objective - ad_objective,
        "rl_speed_avg": rl_objective,
        "adaptive_speed_avg": ad_objective,
        "speed_delta_vs_adaptive": rl_objective - ad_objective,
        "rl_reward_avg": rl_reward,
        "rl_speed_var_avg": rl_var,
        "rl_min_gap_min": rl_min_gap,
        "rl_safety_pass_rate_min": rl_safety_rate,
        "rl_string_stability_avg": rl_string_stability,
        "rl_string_stability_valid_rate_min": rl_string_stability_valid_rate,
        "rl_string_stability_stable_rate_min": rl_string_stability_stable_rate,
    }


def ranking_key(
    row: Dict[str, float | str],
    *,
    policy: str,
) -> tuple[float, ...]:
    if policy == "safety_first":
        return (
            float(row["rl_safety_pass_rate_min"]),
            float(row["rl_min_gap_min"]),
            sort_key_or_floor(float(row["rl_string_stability_stable_rate_min"])),
            sort_key_or_floor(float(row["rl_string_stability_valid_rate_min"])),
            prefer_lower(float(row["rl_string_stability_avg"])),
            float(row["primary_objective_delta_vs_adaptive"]),
            float(row["rl_primary_objective_avg"]),
        )
    if policy == "objective_first":
        return (
            float(row["primary_objective_delta_vs_adaptive"]),
            float(row["rl_primary_objective_avg"]),
            float(row["rl_safety_pass_rate_min"]),
            sort_key_or_floor(float(row["rl_string_stability_stable_rate_min"])),
            prefer_lower(float(row["rl_string_stability_avg"])),
            float(row["rl_min_gap_min"]),
        )
    raise ValueError(f"Unknown ranking policy: {policy}")


def rank_rows(
    rows_out: Iterable[Dict[str, float | str]],
    *,
    policy: str,
) -> List[Dict[str, float | str]]:
    ranked = list(rows_out)
    ranked.sort(key=lambda row: ranking_key(row, policy=policy), reverse=True)
    return ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank evaluated GPU PPO checkpoints.")
    parser.add_argument("--eval_root", type=str, default="output/hpc_gpu_train/evaluation")
    parser.add_argument(
        "--ranking_policy",
        type=str,
        choices=("safety_first", "objective_first"),
        default="safety_first",
        help=(
            "Checkpoint selection rule. 'safety_first' prefers checkpoints with "
            "better worst-case safety, robustness, and only then higher objective."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows_out = []
    pattern = os.path.join(args.eval_root, "*")
    for eval_dir in sorted(glob.glob(pattern)):
        overall_path = os.path.join(eval_dir, "gpu_eval_overall_summary.csv")
        factor_path = os.path.join(eval_dir, "gpu_eval_summary.csv")
        path = overall_path if os.path.exists(overall_path) else factor_path
        if not os.path.exists(path):
            continue
        ckpt_name = os.path.basename(eval_dir)
        rows = load_summary(path)
        row = build_checkpoint_row(ckpt_name, rows)
        if row is not None:
            rows_out.append(row)

    rows_out = rank_rows(rows_out, policy=args.ranking_policy)

    if not rows_out:
        raise SystemExit("No evaluation summaries found.")

    print(
        "checkpoint,primary_objective_metric,rl_primary_objective_avg,"
        "adaptive_primary_objective_avg,primary_objective_delta_vs_adaptive,"
        "rl_speed_avg,adaptive_speed_avg,speed_delta_vs_adaptive,rl_reward_avg,"
        "rl_speed_var_avg,rl_min_gap_min,rl_safety_pass_rate_min,"
        "string_stability_metric,rl_string_stability_avg,"
        "rl_string_stability_valid_rate_min,rl_string_stability_stable_rate_min"
    )
    for row in rows_out:
        print(
            f"{row['checkpoint']},{row['primary_objective_metric']},"
            f"{row['rl_primary_objective_avg']:.4f},"
            f"{row['adaptive_primary_objective_avg']:.4f},"
            f"{row['primary_objective_delta_vs_adaptive']:.4f},"
            f"{row['rl_speed_avg']:.4f},{row['adaptive_speed_avg']:.4f},"
            f"{row['speed_delta_vs_adaptive']:.4f},{row['rl_reward_avg']:.4f},"
            f"{row['rl_speed_var_avg']:.4f},{row['rl_min_gap_min']:.4f},"
            f"{row['rl_safety_pass_rate_min']:.4f},{row['string_stability_metric']},"
            f"{row['rl_string_stability_avg']:.4f},"
            f"{row['rl_string_stability_valid_rate_min']:.4f},"
            f"{row['rl_string_stability_stable_rate_min']:.4f}"
        )

    best = rows_out[0]
    print()
    print(f"RANKING_POLICY={args.ranking_policy}")
    print(f"BEST_CHECKPOINT={best['checkpoint']}")
    if float(best["rl_safety_pass_rate_min"]) < 1.0:
        print(
            "BEST_CHECKPOINT_WARNING=no checkpoint satisfies the safety "
            "constraint across every evaluated human-ratio slice; selected the "
            "safest available tradeoff first"
        )


if __name__ == "__main__":
    main()
