from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.analysis_plots import generate_comparison_report


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = ROOT / "Results_compare_all" / "summary_runs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Results_compare_all" / "summary_plots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate comparison plots from a summary_runs.csv file. "
            "The plotting implementation lives in src.visualization.analysis_plots."
        )
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Input summary_runs.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG and SVG plots will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_paths = generate_comparison_report(args.summary_csv, args.output_dir)
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
