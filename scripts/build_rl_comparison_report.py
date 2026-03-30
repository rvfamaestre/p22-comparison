from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import sys
from pathlib import Path as _Path

# Ensure the project root is importable
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.string_stability_metrics import compute_string_stability_from_traces
from src.utils.primary_objective import (
    annotate_with_primary_objective,
    SAFE_HEADWAY_M,
)


METRIC_COLUMNS = [
    "mean_speed",
    "mean_speed_last100",
    "speed_variance",
    "speed_var_last100",
    "rms_acc",
    "rms_jerk",
    "min_gap",
    "collision_count",
    "collision_clamp_count",
    "string_stability_value",
    "safety_min_gap_ok",
    "safety_constraint_satisfied",
]

METRIC_LABELS = {
    "mean_speed": "Mean Speed",
    "mean_speed_last100": "Mean Speed (Last 100)",
    "speed_variance": "Speed Variance",
    "speed_var_last100": "Speed Var (Last 100)",
    "rms_acc": "RMS Acc",
    "rms_jerk": "RMS Jerk",
    "min_gap": "Min Gap",
    "collision_count": "Collision Count",
    "collision_clamp_count": "Collision Clamp Count",
    "string_stability_value": "Amplification Ratio",
    "safety_min_gap_ok": "Min Gap Safe",
    "safety_constraint_satisfied": "Safety Satisfied",
}

LOWER_IS_BETTER = {
    "mean_speed": False,
    "mean_speed_last100": False,
    "speed_variance": True,
    "speed_var_last100": True,
    "rms_acc": True,
    "rms_jerk": True,
    "min_gap": False,
    "collision_count": True,
    "collision_clamp_count": True,
    "string_stability_value": True,
    "safety_min_gap_ok": False,
    "safety_constraint_satisfied": False,
}

TRAINING_METRICS = [
    "training_objective",
    "speed_var_global",
    "collision_clamp_count",
    "string_stability_value",
    "min_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RL comparison report artifacts.")
    parser.add_argument(
        "--base-output",
        type=Path,
        required=True,
        help="Base output directory containing ppo/ and sac/ subdirectories.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_rl", "ppo", "sac"],
        help="Methods to include in the comparison table.",
    )
    parser.add_argument(
        "--training-methods",
        nargs="+",
        default=["ppo", "sac"],
        help="Methods to include in the training summary and convergence plots.",
    )
    parser.add_argument(
        "--human-ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Human ratios to include.",
    )
    return parser.parse_args()


def tag_from_ratio(human_ratio: float) -> str:
    return f"h{int(round(human_ratio * 100))}"


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_eval_metrics(micro_csv_path: Path, metadata_path: Path, config_path: Path | None = None) -> dict[str, float]:
    df = pd.read_csv(micro_csv_path)
    metadata = read_json(metadata_path)
    dt = float(metadata["dt"])

    v = df["v"].to_numpy(dtype=float)
    a = df["a"].to_numpy(dtype=float)

    jerk_sq_means = []
    for _, group in df.groupby("id"):
        acc_hist = group.sort_values("t")["a"].to_numpy(dtype=float)
        if len(acc_hist) >= 2:
            jerk = np.diff(acc_hist) / dt
            jerk_sq_means.append(np.mean(jerk ** 2))

    # Mean speed over last 100 time steps
    times = np.sort(df["t"].unique())
    last100_times = times[-min(100, len(times)):]
    mask_last100 = df["t"].isin(last100_times)

    metrics = {
        "mean_speed": float(np.mean(v)),
        "mean_speed_last100": float(df.loc[mask_last100, "v"].mean()),
        "speed_variance": float(np.var(v)),
        "speed_var_last100": float(np.var(df.loc[mask_last100, "v"].to_numpy())),
        "rms_acc": float(np.sqrt(np.mean(a ** 2))),
        "rms_jerk": float(np.sqrt(np.mean(jerk_sq_means))) if jerk_sq_means else float("nan"),
        "min_gap": float(df["gap_s"].min()) if "gap_s" in df.columns else float("nan"),
        "collision_count": float(metadata.get("collision_count", 0.0)),
        "collision_clamp_count": float(metadata.get("collision_clamp_count", 0.0)),
        "string_stability_value": float("nan"),
        "string_stability_metric_valid": False,
        "string_stability_is_stable": None,
    }

    # String stability (if config available with perturbation info)
    cfg = {}
    if config_path is not None and config_path.is_file():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    perturbation_enabled = cfg.get(
        "perturbation_enabled",
        metadata.get("perturbation_enabled", False),
    )
    if perturbation_enabled and "gap_s" in df.columns:
        # Build minimal traces DataFrame with expected column names
        traces = df.rename(columns={"t": "time", "id": "vehicle_id"})
        ss_result = compute_string_stability_from_traces(
            traces,
            perturb_vehicle_id=int(
                cfg.get(
                    "perturbation_vehicle",
                    metadata.get("perturbation_vehicle", 0),
                )
            ),
            perturbation_time=float(
                cfg.get(
                    "perturbation_time",
                    metadata.get("perturbation_time", 3.0),
                )
            ),
            valid_base=(int(metadata.get("collision_clamp_count", 0)) == 0),
            applicable=True,
        )
        metrics.update(ss_result)

    # Safety and primary objective annotation
    metrics = annotate_with_primary_objective(
        metrics,
        min_gap_key="min_gap",
        min_gap_threshold=SAFE_HEADWAY_M,
        collision_count_key="collision_count",
        collision_clamp_count_key="collision_clamp_count",
        string_stability_key=(
            "string_stability_is_stable"
            if bool(metrics.get("string_stability_metric_valid", False))
            else None
        ),
    )

    return metrics


def collect_eval_rows(base_output: Path, methods: list[str], human_ratios: list[float]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for human_ratio in human_ratios:
            tag = tag_from_ratio(human_ratio)
            run_dir = base_output / method / tag / "simulation"
            micro_csv = run_dir / "micro.csv"
            metadata_json = run_dir / "metadata.json"

            config_effective = run_dir / "config_effective.yml"

            if not micro_csv.is_file():
                raise FileNotFoundError(f"Missing evaluation micro.csv: {micro_csv}")
            if not metadata_json.is_file():
                raise FileNotFoundError(f"Missing evaluation metadata.json: {metadata_json}")

            metrics = compute_eval_metrics(micro_csv, metadata_json, config_effective)
            row = {
                "method": method,
                "human_ratio": human_ratio,
                "tag": tag,
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["method", "human_ratio"]).reset_index(drop=True)
    return df


def parse_duration_seconds(duration_path: Path) -> float | None:
    if not duration_path.is_file():
        return None

    real_value = None
    with open(duration_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("real "):
                try:
                    real_value = float(line.split()[1])
                except (IndexError, ValueError):
                    real_value = None
    return real_value


def collect_training_rows(base_output: Path, methods: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    rows = []
    histories: dict[str, dict[str, list[float]]] = {}

    for method in methods:
        metrics_path = base_output / method / "training" / "trained_models" / "training_metrics.json"
        duration_path = base_output / method / "training_duration.txt"

        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing training metrics: {metrics_path}")

        history = read_json(metrics_path)
        histories[method] = history

        primary_metric = "training_objective" if history.get("training_objective", []) else "speed_var_global"
        primary_curve = history.get(primary_metric, [])
        if not primary_curve:
            raise ValueError(f"{primary_metric} missing or empty in {metrics_path}")

        best_idx = int(np.argmin(primary_curve))
        best_episode = best_idx + 1
        last_window = primary_curve[-10:]

        rows.append(
            {
                "method": method,
                "primary_metric": primary_metric,
                "episodes_completed": len(primary_curve),
                "best_episode": best_episode,
                "best_primary_value": float(primary_curve[best_idx]),
                "final_primary_value": float(primary_curve[-1]),
                "last10_mean_primary_value": float(np.mean(last_window)),
                "last10_std_primary_value": float(np.std(last_window)),
                "runtime_seconds": parse_duration_seconds(duration_path),
            }
        )

    training_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return training_df, histories


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_shared_strings(rows: list[list[object]]) -> tuple[list[str], dict[str, int]]:
    strings: list[str] = []
    string_to_index: dict[str, int] = {}

    for row in rows:
        for value in row:
            if isinstance(value, str) and value not in string_to_index:
                string_to_index[value] = len(strings)
                strings.append(value)

    return strings, string_to_index


def xlsx_column_name(index: int) -> str:
    name = ""
    while index >= 0:
        name = chr(index % 26 + ord("A")) + name
        index = index // 26 - 1
    return name


def write_excel_table(df: pd.DataFrame, path: Path, sheet_name: str = "comparison") -> None:
    rows = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    shared_strings, shared_string_lookup = build_shared_strings(rows)

    def cell_xml(row_idx: int, col_idx: int, value: object) -> str:
        cell_ref = f"{xlsx_column_name(col_idx)}{row_idx + 1}"
        if isinstance(value, str):
            return f'<c r="{cell_ref}" t="s"><v>{shared_string_lookup[value]}</v></c>'
        if value == "":
            return f'<c r="{cell_ref}"/>'
        if isinstance(value, (np.integer, int)):
            return f'<c r="{cell_ref}"><v>{int(value)}</v></c>'
        if isinstance(value, (np.floating, float)):
            if math.isnan(float(value)):
                return f'<c r="{cell_ref}"/>'
            return f'<c r="{cell_ref}"><v>{float(value)}</v></c>'
        return f'<c r="{cell_ref}" t="s"><v>{shared_string_lookup[str(value)]}</v></c>'

    sheet_rows = []
    for row_idx, row in enumerate(rows):
        cells = "".join(cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row))
        sheet_rows.append(f'<row r="{row_idx + 1}">{cells}</row>')

    table_ref = f"A1:{xlsx_column_name(df.shape[1] - 1)}{df.shape[0] + 1}"
    table_columns = "".join(
        f'<tableColumn id="{idx + 1}" name="{escape(col)}"/>'
        for idx, col in enumerate(df.columns)
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<dimension ref="{table_ref}"/>'
        '<sheetData>'
        + "".join(sheet_rows)
        + "</sheetData>"
        '<autoFilter ref="' + table_ref + '"/>'
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
        '<tableParts count="1"><tablePart r:id="rId1"/></tableParts>'
        "</worksheet>"
    )

    table_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'id="1" name="ComparisonTable" displayName="ComparisonTable" '
        f'ref="{table_ref}" totalsRowShown="0">'
        f'<autoFilter ref="{table_ref}"/>'
        f'<tableColumns count="{df.shape[1]}">{table_columns}</tableColumns>'
        '<tableStyleInfo name="TableStyleMedium2" showFirstColumn="0" showLastColumn="0" '
        'showRowStripes="1" showColumnStripes="0"/>'
        "</table>"
    )

    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
        + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared_strings)
        + "</sst>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
        "</Relationships>"
    )

    worksheet_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" '
        'Target="../tables/table1.xml"/>'
        "</Relationships>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '<Override PartName="/xl/tables/table1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"/>'
        "</Types>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
        zf.writestr("xl/worksheets/_rels/sheet1.xml.rels", worksheet_rels_xml)
        zf.writestr("xl/tables/table1.xml", table_xml)
        zf.writestr("xl/sharedStrings.xml", shared_strings_xml)
        zf.writestr("xl/styles.xml", styles_xml)


def plot_eval_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    methods = list(df["method"].unique())
    tags = list(df["tag"].unique())
    x = np.arange(len(tags))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(8, 4.8))
    for idx, method in enumerate(methods):
        subset = df[df["method"] == method].sort_values("human_ratio")
        plt.bar(x + (idx - (len(methods) - 1) / 2) * width, subset[metric], width=width, label=method.upper())

    plt.xticks(x, tags)
    plt.xlabel("Human Ratio Tag")
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"{METRIC_LABELS[metric]} Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_overview(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "mean_speed",
        "mean_speed_last100",
        "speed_variance",
        "rms_acc",
        "min_gap",
        "collision_count",
        "string_stability_value",
        "rms_jerk",
    ]
    # Filter to metrics actually present in the dataframe
    metrics = [m for m in metrics if m in df.columns]
    ncols = min(len(metrics), 4)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    methods = list(df["method"].unique())
    tags = list(df["tag"].unique())
    x = np.arange(len(tags))
    width = 0.8 / max(len(methods), 1)

    for ax, metric in zip(axes, metrics):
        for idx, method in enumerate(methods):
            subset = df[df["method"] == method].sort_values("human_ratio")
            ax.bar(
                x + (idx - (len(methods) - 1) / 2) * width,
                subset[metric],
                width=width,
                label=method.upper() if metric == metrics[0] else None,
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x, tags)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_metric(histories: dict[str, dict[str, list[float]]], metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    for method, history in sorted(histories.items()):
        values = history.get(metric, [])
        if not values:
            continue
        episodes = np.arange(1, len(values) + 1)
        plt.plot(episodes, values, label=method.upper())

    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(f"Training Convergence: {metric}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_training_overview(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    metrics = TRAINING_METRICS + ["speed_std_time_mean"]
    ncols = min(len(metrics), 3)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for method, history in sorted(histories.items()):
            values = history.get(metric, [])
            if not values:
                continue
            episodes = np.arange(1, len(values) + 1)
            ax.plot(episodes, values, label=method.upper())
        ax.set_title(metric)
        ax.grid(alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def best_method_for_metric(eval_df: pd.DataFrame, metric: str, human_ratio: float) -> str:
    if metric not in eval_df.columns:
        return "n/a"
    subset = eval_df[eval_df["human_ratio"] == human_ratio].copy()
    if subset.empty:
        return "n/a"
    subset = subset[subset[metric].notna()]
    if subset.empty:
        return "n/a"
    ascending = LOWER_IS_BETTER[metric]
    best_row = subset.sort_values(metric, ascending=ascending).iloc[0]
    return str(best_row["method"])


def write_markdown_report(
    base_output: Path,
    eval_df: pd.DataFrame,
    training_df: pd.DataFrame,
    report_path: Path,
    methods: list[str],
) -> None:
    lines: list[str] = []
    title = " vs ".join(method.replace("_", " ").upper() for method in methods)
    lines.append(f"# {title} Experiment Report")
    lines.append("")
    lines.append(f"Base output: `{base_output}`")
    lines.append("")
    if not training_df.empty:
        lines.append("## Training Summary")
        lines.append("")
        lines.append("| Method | Primary Metric | Episodes | Best Episode | Best Value | Final Value | Last10 Mean | Last10 Std | Runtime (s) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in training_df.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        str(row["primary_metric"]),
                        str(int(row["episodes_completed"])),
                        str(int(row["best_episode"])),
                        format_float(float(row["best_primary_value"])),
                        format_float(float(row["final_primary_value"])),
                        format_float(float(row["last10_mean_primary_value"])),
                        format_float(float(row["last10_std_primary_value"])),
                        format_float(float(row["runtime_seconds"])) if pd.notnull(row["runtime_seconds"]) else "nan",
                    ]
                )
                + " |"
            )

        lines.append("")
    lines.append("## Evaluation Summary")
    lines.append("")
    lines.append("| Method | Tag | HR | Mean Spd | Mean Spd L100 | Spd Var | Spd Var L100 | RMS Acc | RMS Jerk | Min Gap | Collisions | Clamps | Amp. Ratio | Safety |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in eval_df.iterrows():
        amp_ratio = format_float(float(row["string_stability_value"])) if "string_stability_value" in row and pd.notnull(row.get("string_stability_value")) else "n/a"
        safety = str(row.get("safety_constraint_satisfied", "n/a"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["tag"]),
                    f"{float(row['human_ratio']):.2f}",
                    format_float(float(row["mean_speed"])),
                    format_float(float(row.get("mean_speed_last100", float("nan")))),
                    format_float(float(row["speed_variance"])),
                    format_float(float(row.get("speed_var_last100", float("nan")))),
                    format_float(float(row["rms_acc"])),
                    format_float(float(row["rms_jerk"])),
                    format_float(float(row["min_gap"])),
                    str(int(row["collision_count"])),
                    str(int(row["collision_clamp_count"])),
                    amp_ratio,
                    safety,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Best Method By Metric And Human Ratio")
    lines.append("")
    for human_ratio in sorted(eval_df["human_ratio"].unique()):
        tag = tag_from_ratio(float(human_ratio))
        lines.append(f"### {tag}")
        lines.append("")
        for metric in METRIC_COLUMNS:
            if metric not in eval_df.columns:
                continue
            lines.append(f"- {METRIC_LABELS[metric]}: `{best_method_for_metric(eval_df, metric, float(human_ratio))}`")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_output = args.base_output
    comparison_dir = base_output / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    eval_df = collect_eval_rows(base_output, args.methods, args.human_ratios)
    training_df, histories = collect_training_rows(base_output, args.training_methods)

    comparison_csv = comparison_dir / "comparison_metrics.csv"
    comparison_xlsx = comparison_dir / "comparison_metrics.xlsx"
    training_csv = comparison_dir / "training_summary.csv"
    report_md = comparison_dir / "experiment_report.md"

    write_csv(eval_df, comparison_csv)
    write_excel_table(eval_df, comparison_xlsx)
    write_csv(training_df, training_csv)
    write_markdown_report(base_output, eval_df, training_df, report_md, args.methods)

    for metric in METRIC_COLUMNS:
        if metric not in eval_df.columns:
            continue
        if eval_df[metric].notna().sum() == 0:
            continue
        plot_eval_metric(eval_df, metric, comparison_dir / f"{metric}_comparison.png")

    plot_overview(eval_df, comparison_dir / "overview_comparison.png")

    if histories:
        for metric in TRAINING_METRICS:
            plot_training_metric(histories, metric, comparison_dir / f"{metric}_training_comparison.png")

        plot_training_overview(histories, comparison_dir / "training_overview_comparison.png")

    print(f"Saved evaluation CSV: {comparison_csv}")
    print(f"Saved evaluation XLSX: {comparison_xlsx}")
    print(f"Saved training CSV: {training_csv}")
    print(f"Saved markdown report: {report_md}")


if __name__ == "__main__":
    main()
