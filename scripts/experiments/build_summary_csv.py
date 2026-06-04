#!/usr/bin/env python3
"""Aggregate summary_row.json files from multiple runs into a CSV table."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.experiment_utils import write_summary_table  # noqa: E402


LOGGER = logging.getLogger("build_summary_csv")
DEFAULT_SUMMARY_ROOTS: Tuple[Path, ...] = (
    REPO_ROOT / "runs" / "experiments",
    REPO_ROOT / "runs" / "chunking_sweeps",
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "latest" / "paper_summary.csv"


def parse_key_values(pairs: List[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def discover_summary_run_dirs(summary_file: str, roots=DEFAULT_SUMMARY_ROOTS) -> List[Path]:
    run_dirs: List[Path] = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        run_dirs.extend(sorted(path.parent for path in root.rglob(summary_file) if path.is_file()))
    return run_dirs


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-run summary_row.json files and emit an aggregated CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, help="Run directories that contain summary_row.json files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV path.")
    parser.add_argument(
        "--summary-file",
        default="summary_row.json",
        help="Filename to load within each run directory.",
    )
    parser.add_argument(
        "--extra-column",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional columns added to each row prior to writing the table.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows: List[Dict[str, object]] = []
    extra_columns = parse_key_values(args.extra_column)
    run_dirs = args.run_dirs or discover_summary_run_dirs(args.summary_file, DEFAULT_SUMMARY_ROOTS)

    if not run_dirs:
        LOGGER.warning("No run directories found; skipping CSV write.")
        return 0

    for run_dir in run_dirs:
        run_dir = run_dir.expanduser().resolve()
        summary_path = run_dir / args.summary_file
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row.update(extra_columns)
        rows.append(row)

    if not rows:
        LOGGER.warning("No rows collected; skipping CSV write.")
        return 0

    output_path = args.output.expanduser().resolve()
    write_summary_table(rows, output_path)
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
