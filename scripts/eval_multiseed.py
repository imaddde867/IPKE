#!/usr/bin/env python3
"""Multi-seed evaluation harness for IPKE.

Runs the extraction pipeline over each gold document for N seeds, then scores
with the Tier-A metrics. Records every run in a timestamped CSV and prints a
summary table with mean +/- 95% CI per (model, chunker, prompter).

Usage:
    uv run python scripts/eval_multiseed.py \\
        --gold-dir datasets/paper/gold \\
        --text-dir datasets/paper/text \\
        --seeds 5 \\
        --out-dir results/

    # Combine with archive thesis gold:
    uv run python scripts/eval_multiseed.py \\
        --gold-dir datasets/archive/gold_human \\
        --text-dir datasets/archive/test_data/text \\
        --seeds 3

Options:
    --gold-dir      Directory of gold JSON files (required).
    --text-dir      Directory of source text files — filename stem must match
                    gold filename stem (required).
    --seeds         Number of random seeds (default: 5). Seeds used: 0..N-1.
    --seed-start    Offset the seed sequence (default: 0).
    --out-dir       Directory for results CSV (default: results/).
    --chunker       Override chunking method (env: CHUNKING_METHOD).
    --prompter      Override prompting strategy (env: PROMPTING_STRATEGY).
    --threshold     Embedding similarity threshold for evaluation (default: 0.75).
    --dry-run       Print the run plan without executing any extraction.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRIC_COLUMNS = [
    "StepF1",
    "AdjacencyF1",
    "Kendall",
    "ConstraintCoverage",
    "ConstraintAttachmentF1",
    "Phi",
]

RUN_COLUMNS = [
    "doc_id",
    "seed",
    "model_id",
    "chunker",
    "prompter",
] + METRIC_COLUMNS


def _resolve(value: str | Path, base: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


async def _extract_one(text_path: Path, doc_id: str, seed: int, processor: Any) -> dict[str, Any]:
    """Run the pipeline for a single (doc, seed) pair.

    ``processor`` is a pre-built ``StreamlinedDocumentProcessor``. The seed is
    passed as an env var so the underlying LLM inference reads it at call time;
    the processor instance is reused across seeds to avoid repeated model loads.
    """
    from src.pipelines.baseline import extraction_payload

    os.environ["LLM_RANDOM_SEED"] = str(seed)
    result = await processor.process_document(file_path=str(text_path), document_id=doc_id)
    return extraction_payload(doc_id, result)


def _score(gold: dict[str, Any], pred: dict[str, Any], threshold: float, preprocessor, embedder) -> dict[str, Any]:
    from src.evaluation.metrics import evaluate_tier_a_document
    return evaluate_tier_a_document(gold, pred, preprocessor, embedder, threshold)


def _ci95(values: list[float]) -> tuple[float, float]:
    from scipy.stats import t as t_dist
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    margin = t_dist.ppf(0.975, df=n - 1) * se
    return (mean - margin, mean + margin)


def _summary_table(rows: list[dict[str, Any]]) -> str:
    from collections import defaultdict

    groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: {m: [] for m in METRIC_COLUMNS})
    for row in rows:
        key = (row["model_id"], row["chunker"], row["prompter"])
        for m in METRIC_COLUMNS:
            val = row.get(m)
            if val is not None and val != "":
                try:
                    groups[key][m].append(float(val))
                except (ValueError, TypeError):
                    pass

    lines: list[str] = []
    header = f"{'model':<30}  {'chunker':<18}  {'prompter':<5}  {'n':>5}"
    for m in METRIC_COLUMNS:
        header += f"  {m:>22}"
    lines.append(header)
    lines.append("-" * len(header))

    for (model, chunker, prompter), metrics in sorted(groups.items()):
        n_vals = max(len(v) for v in metrics.values()) if metrics else 0
        row_str = f"{model:<30}  {chunker:<18}  {prompter:<5}  {n_vals:>5}"
        for m in METRIC_COLUMNS:
            vals = metrics[m]
            if not vals:
                row_str += f"  {'N/A':>22}"
                continue
            mean = float(np.mean(vals))
            lo, hi = _ci95(vals)
            n = len(vals)
            cell = f"{mean:.3f} [{lo:.3f},{hi:.3f}] n={n}"
            row_str += f"  {cell:>22}"
        lines.append(row_str)

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", required=True, type=Path, help="Gold JSON directory.")
    parser.add_argument("--text-dir", required=True, type=Path, help="Source text directory.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5).")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value (default: 0).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results", help="Output directory.")
    parser.add_argument("--chunker", default=None, help="Override CHUNKING_METHOD env var.")
    parser.add_argument("--prompter", default=None, help="Override PROMPTING_STRATEGY env var.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Eval embedding threshold (default: 0.75).")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, no extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gold_dir = args.gold_dir.resolve()
    text_dir = args.text_dir.resolve()

    if not gold_dir.exists():
        print(f"ERROR: gold-dir not found: {gold_dir}", file=sys.stderr)
        return 1
    if not text_dir.exists():
        print(f"ERROR: text-dir not found: {text_dir}", file=sys.stderr)
        return 1

    gold_files = sorted(gold_dir.glob("*.json"))
    if not gold_files:
        print(f"ERROR: no JSON files in {gold_dir}", file=sys.stderr)
        return 1

    # Match text files by stem
    pairs: list[tuple[Path, Path]] = []
    for gf in gold_files:
        tf = text_dir / f"{gf.stem}.txt"
        if not tf.exists():
            print(f"WARNING: text file not found for {gf.stem}, skipping.", file=sys.stderr)
        else:
            pairs.append((gf, tf))

    if not pairs:
        print("ERROR: no (gold, text) pairs found.", file=sys.stderr)
        return 1

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    if args.chunker:
        os.environ["CHUNKING_METHOD"] = args.chunker
    if args.prompter:
        os.environ["PROMPTING_STRATEGY"] = args.prompter

    # Read effective config for metadata (after env overrides)
    from src.core.unified_config import reload_config

    os.environ.setdefault("PROMPTING_STRATEGY", "P3")
    os.environ.setdefault("CHUNKING_METHOD", "dsc")
    config = reload_config()
    model_id = config.llm_model_id or "unknown"
    chunker = (args.chunker or config.chunking_method or "dsc").lower()
    prompter = (args.prompter or config.prompting_strategy or "P3").upper()

    plan = [
        {"doc_id": gf.stem, "seed": seed, "model_id": model_id, "chunker": chunker, "prompter": prompter}
        for gf, _ in pairs
        for seed in seeds
    ]
    n_docs = len(pairs)
    n_runs = len(plan)

    print(f"Plan: {n_docs} docs x {len(seeds)} seeds = {n_runs} runs")
    print(f"Model: {model_id}  Chunker: {chunker}  Prompter: {prompter}")
    print(f"Seeds: {seeds}")
    for gf, tf in pairs:
        print(f"  {gf.stem}")

    if args.dry_run:
        print("\n[dry-run] No extraction performed.")
        return 0

    from src.evaluation.metrics import prepare_evaluator
    from src.processors.streamlined_processor import StreamlinedDocumentProcessor

    preprocessor, embedder = prepare_evaluator()

    # Reuse the config already loaded above; build processor once so the LLM
    # backend is loaded a single time for all (doc, seed) iterations.
    processor = StreamlinedDocumentProcessor(config=config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"multiseed_{timestamp}.csv"

    rows: list[dict[str, Any]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS)
        writer.writeheader()

        for gf, tf in pairs:
            doc_id = gf.stem
            gold = json.loads(gf.read_text(encoding="utf-8"))
            for seed in seeds:
                print(f"  {doc_id}  seed={seed} ...", end=" ", flush=True)
                try:
                    pred = asyncio.run(_extract_one(tf, doc_id, seed, processor))
                    metrics = _score(gold, pred, args.threshold, preprocessor, embedder)
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    metrics = {m: None for m in METRIC_COLUMNS}

                row: dict[str, Any] = {
                    "doc_id": doc_id,
                    "seed": seed,
                    "model_id": model_id,
                    "chunker": chunker,
                    "prompter": prompter,
                }
                for m in METRIC_COLUMNS:
                    row[m] = metrics.get(m)
                rows.append(row)
                writer.writerow(row)
                fh.flush()

                phi_str = f"Phi={row['Phi']:.3f}" if row["Phi"] is not None else "Phi=N/A"
                print(phi_str)

    print(f"\nResults: {csv_path}  ({len(rows)} rows)")
    print()
    print(_summary_table(rows))
    print(f"\nNote: CI via scipy.stats.t.ppf (95%, df=n-1). n = docs x seeds per group.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
