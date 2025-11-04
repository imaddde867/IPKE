#!/usr/bin/env python3
"""
Run the baseline extractor multiple times on the tier A test set and evaluate.

Example:
    python scripts/run_baseline_loops.py --runs 3 --out logs/baseline_runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from src.pipelines.baseline import (
    TIER_A_TEST_DOCS,
    accumulate_metrics,
    evaluate_predictions,
    extract_documents,
    summarise_metrics,
)
from src.processors.streamlined_processor import StreamlinedDocumentProcessor


async def run_extraction(run_dir: Path) -> None:
    """Extract baseline predictions for all documents into ``run_dir``."""
    processor = StreamlinedDocumentProcessor()
    await extract_documents(
        processor,
        doc_sources=TIER_A_TEST_DOCS,
        run_dir=run_dir,
        status_callback=lambda doc_id, payload: print(
            f"[extract] {doc_id}: {len(payload['steps'])} steps, "
            f"{len(payload['constraints'])} constraints, {len(payload['entities'])} entities"
        ),
    )


def run_evaluation(pred_dir: Path, out_path: Path) -> Dict[str, Dict[str, float | None]]:
    """Run the evaluation CLI against ``pred_dir`` and return the parsed JSON results."""
    print(f"[evaluate] -> {out_path}")
    return evaluate_predictions(pred_dir=pred_dir, out_path=out_path)


async def orchestrate(runs: int, out_root: Path) -> None:
    """Execute multiple extraction/evaluation runs and summarise the results."""
    accumulator: Dict[str, Dict[str, List[float]]] = {}
    for run_idx in range(1, runs + 1):
        run_dir = out_root / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Run {run_idx}/{runs} ===")
        await run_extraction(run_dir)
        report_path = run_dir / "evaluation_report.json"
        metrics = run_evaluation(run_dir, report_path)
        accumulate_metrics(accumulator, metrics)

    average = summarise_metrics(accumulator)
    summary_path = out_root / "aggregate_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(average, indent=2), encoding="utf-8")
    print(f"\nAggregate metrics saved to {summary_path}")
    print(json.dumps(average, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline extractor/evaluator loops.")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs to execute.")
    parser.add_argument("--out", type=Path, default=Path("logs/baseline_runs"), help="Output directory root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(orchestrate(args.runs, args.out))


if __name__ == "__main__":
    main()
