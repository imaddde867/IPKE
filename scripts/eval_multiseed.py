#!/usr/bin/env python3
"""Multi-seed evaluation harness for IPKE.

Runs the extraction pipeline over each gold document for N seeds, scores with
Tier-A metrics, and writes two CSVs:

  results_detail_<ts>.csv  -- one row per (doc, seed)
  results_summary_<ts>.csv -- one row per (model, chunker, prompter) with
                              mean, 95% CI, Phi sensitivity, and optionally a
                              paired-bootstrap p-value vs a prior run CSV.

Usage:
    uv run python scripts/eval_multiseed.py \\
        --gold-dir datasets/paper/gold \\
        --text-dir datasets/paper/text \\
        --seeds 5 \\
        --out-dir results/

    # With bootstrap comparison against a prior run:
    uv run python scripts/eval_multiseed.py ... \\
        --compare-against results/results_detail_20260601T120000.csv

    # With custom Phi weight schemes (format: cov:step:tau):
    uv run python scripts/eval_multiseed.py ... \\
        --phi-weights 0.5:0.3:0.2 \\
        --phi-weights 0.4:0.4:0.2 \\
        --phi-weights 0.6:0.2:0.2

Options:
    --gold-dir         Directory of gold JSON files (required).
    --text-dir         Directory of source text files — stem must match gold stem.
    --seeds            Number of random seeds (default: 5). Seeds used: seed-start..(start+N-1).
    --seed-start       First seed value (default: 0).
    --out-dir          Directory for result CSVs (default: results/).
    --chunker          Override CHUNKING_METHOD env var.
    --prompter         Override PROMPTING_STRATEGY env var.
    --threshold        Embedding similarity threshold for evaluation (default: 0.75).
    --compare-against  Path to a prior detail CSV for paired bootstrap p-value.
    --phi-weights      Cov:StepF1:Tau weight triple (repeatable; default: 0.5:0.3:0.2).
    --bootstrap-n      Bootstrap resampling iterations (default: 10000).
    --dry-run          Print plan only, no extraction.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
from collections import defaultdict
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

DETAIL_COLUMNS = [
    "doc_id",
    "seed",
    "model_id",
    "chunker",
    "prompter",
] + METRIC_COLUMNS

DEFAULT_PHI_WEIGHTS = [(0.5, 0.3, 0.2)]


# ---------------------------------------------------------------------------
# Pure helpers (testable without model)
# ---------------------------------------------------------------------------

def phi(coverage: float | None, step_f1: float | None, kendall: float | None,
        w_cov: float = 0.5, w_step: float = 0.3, w_tau: float = 0.2) -> float:
    c = coverage or 0.0
    s = step_f1 or 0.0
    k = kendall or 0.0
    return w_cov * c + w_step * s + w_tau * k


def _ci95(values: list[float]) -> tuple[float, float]:
    from scipy.stats import t as t_dist
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    margin = t_dist.ppf(0.975, df=n - 1) * se
    return (mean - margin, mean + margin)


def paired_bootstrap_pvalue(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """Two-sided paired bootstrap test: H0 = mean(A - B) == 0.

    Follows Dror et al. (2018) ACL "The Hitchhiker's Guide to Testing
    Statistical Significance in NLP", Algorithm 1 (paired bootstrap).

    Both lists must have the same length (one score per document). Returns
    p-value under H0. Lower is stronger evidence that A != B.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Score lists must be same length: {len(a)} != {len(b)}")
    if len(a) == 0:
        return float("nan")
    n = len(a)
    observed_diff = float(np.mean(a - b))
    rng = np.random.default_rng(rng_seed)
    # Paired bootstrap: resample document indices with replacement
    count_extreme = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_diff = float(np.mean(a[idx] - b[idx]))
        # Shift so H0 holds in this bootstrap world, then check if we exceed observed
        if abs(boot_diff - observed_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_resamples


def _load_detail_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summary_rows(
    detail_rows: list[dict[str, Any]],
    phi_weight_schemes: list[tuple[float, float, float]],
) -> list[dict[str, Any]]:
    """Aggregate detail rows into one summary row per (model, chunker, prompter)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (row.get("model_id", ""), row.get("chunker", ""), row.get("prompter", ""))
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for (model, chunker, prompter), rows in sorted(groups.items()):
        n = len(rows)
        base: dict[str, Any] = {
            "model_id": model,
            "chunker": chunker,
            "prompter": prompter,
            "n_runs": n,
        }
        for m in METRIC_COLUMNS:
            vals = [float(r[m]) for r in rows if r.get(m) not in (None, "", "None")]
            if not vals:
                base[f"{m}_mean"] = None
                base[f"{m}_ci_lo"] = None
                base[f"{m}_ci_hi"] = None
                base[f"{m}_n"] = 0
                continue
            mean = float(np.mean(vals))
            lo, hi = _ci95(vals)
            base[f"{m}_mean"] = round(mean, 4)
            base[f"{m}_ci_lo"] = round(lo, 4) if not math.isnan(lo) else None
            base[f"{m}_ci_hi"] = round(hi, 4) if not math.isnan(hi) else None
            base[f"{m}_n"] = len(vals)

        # Phi sensitivity across alternative weight schemes
        for w_cov, w_step, w_tau in phi_weight_schemes:
            label = f"Phi_{w_cov}_{w_step}_{w_tau}"
            phi_vals = []
            for r in rows:
                try:
                    phi_vals.append(phi(
                        float(r["ConstraintCoverage"]) if r.get("ConstraintCoverage") not in (None, "", "None") else None,
                        float(r["StepF1"]) if r.get("StepF1") not in (None, "", "None") else None,
                        float(r["Kendall"]) if r.get("Kendall") not in (None, "", "None") else None,
                        w_cov, w_step, w_tau,
                    ))
                except (ValueError, TypeError):
                    pass
            if phi_vals:
                mean = float(np.mean(phi_vals))
                lo, hi = _ci95(phi_vals)
                base[f"{label}_mean"] = round(mean, 4)
                base[f"{label}_ci_lo"] = round(lo, 4) if not math.isnan(lo) else None
                base[f"{label}_ci_hi"] = round(hi, 4) if not math.isnan(hi) else None
            else:
                base[f"{label}_mean"] = None
                base[f"{label}_ci_lo"] = None
                base[f"{label}_ci_hi"] = None

        summary.append(base)
    return summary


def _print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    for row in summary_rows:
        print(f"\n  {row['model_id']} / {row['chunker']} / {row['prompter']}  (n={row['n_runs']})")
        for m in METRIC_COLUMNS:
            mean = row.get(f"{m}_mean")
            lo = row.get(f"{m}_ci_lo")
            hi = row.get(f"{m}_ci_hi")
            n = row.get(f"{m}_n", 0)
            if mean is None:
                print(f"    {m:<25} N/A")
            else:
                ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None else "[N/A]"
                print(f"    {m:<25} {mean:.3f} {ci} n={n}")
        # Print phi sensitivity rows
        for key in sorted(k for k in row if k.startswith("Phi_") and k.endswith("_mean")):
            scheme = key[:-5]  # strip _mean
            mean = row.get(key)
            lo = row.get(f"{scheme}_ci_lo")
            hi = row.get(f"{scheme}_ci_hi")
            if mean is not None:
                ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None else "[N/A]"
                print(f"    {scheme:<25} {mean:.3f} {ci}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

async def _extract_one(text_path: Path, doc_id: str, seed: int) -> dict[str, Any]:
    """Run the pipeline for a single (doc, seed) pair.

    A new processor is built per seed so the seed value reaches the backend
    RNG at construction time. Both llama-cpp and transformers backends capture
    RNG state in __init__, not at generate() time; a shared processor would
    produce identical outputs for every nominal seed value.
    """
    from src.core.unified_config import reload_config
    from src.pipelines.baseline import extraction_payload
    from src.processors.streamlined_processor import StreamlinedDocumentProcessor

    os.environ["LLM_RANDOM_SEED"] = str(seed)
    config = reload_config()
    processor = StreamlinedDocumentProcessor(config=config)
    result = await processor.process_document(file_path=str(text_path), document_id=doc_id)
    return extraction_payload(doc_id, result)


def _score(
    gold: dict[str, Any],
    pred: dict[str, Any],
    threshold: float,
    preprocessor: Any,
    embedder: Any,
) -> dict[str, Any]:
    from src.evaluation.metrics import evaluate_tier_a_document
    return evaluate_tier_a_document(gold, pred, preprocessor, embedder, threshold)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_phi_weights(raw: str) -> tuple[float, float, float]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected cov:step:tau, got '{raw}'")
    w = tuple(float(p) for p in parts)
    if abs(sum(w) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"Weights must sum to 1.0, got {sum(w):.4f}")
    return w  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--text-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--chunker", default=None)
    parser.add_argument("--prompter", default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--compare-against", type=Path, default=None,
                        help="Prior detail CSV for paired bootstrap p-value.")
    parser.add_argument("--phi-weights", type=_parse_phi_weights, action="append", dest="phi_weights",
                        metavar="COV:STEP:TAU",
                        help="Phi weight scheme (repeatable). Default: 0.5:0.3:0.2.")
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help=(
            "Allow gold files with quality.review_status != 'reviewed'. "
            "Use only for development; unreviewed files are AI drafts and "
            "results cannot be used as paper evidence."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gold_dir = args.gold_dir.resolve()
    text_dir = args.text_dir.resolve()

    for d in (gold_dir, text_dir):
        if not d.exists():
            print(f"ERROR: directory not found: {d}", file=sys.stderr)
            return 1

    gold_files = sorted(gold_dir.glob("*.json"))
    if not gold_files:
        print(f"ERROR: no JSON files in {gold_dir}", file=sys.stderr)
        return 1

    pairs: list[tuple[Path, Path]] = []
    for gf in gold_files:
        tf = text_dir / f"{gf.stem}.txt"
        if not tf.exists():
            print(f"WARNING: text file missing for {gf.stem}, skipping.", file=sys.stderr)
        else:
            pairs.append((gf, tf))

    if not pairs:
        print("ERROR: no (gold, text) pairs found.", file=sys.stderr)
        return 1

    # Reject unreviewed gold unless explicitly overridden.
    # Unreviewed files are AI-assisted drafts; results over them cannot be
    # used as paper evidence.
    if not args.dry_run and not args.allow_unreviewed:
        unreviewed = []
        for gf, _ in pairs:
            try:
                data = json.loads(gf.read_text(encoding="utf-8"))
                status = data.get("quality", {}).get("review_status", "")
                if status != "reviewed":
                    unreviewed.append(gf.name)
            except Exception:  # noqa: BLE001
                pass
        if unreviewed:
            print(
                "ERROR: the following gold files are not reviewed "
                "(quality.review_status != 'reviewed'):",
                file=sys.stderr,
            )
            for name in unreviewed:
                print(f"  {name}", file=sys.stderr)
            print(
                "Complete a human correction pass and set quality.review_status='reviewed', "
                "or pass --allow-unreviewed for development use only.",
                file=sys.stderr,
            )
            return 1

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    phi_schemes: list[tuple[float, float, float]] = args.phi_weights or DEFAULT_PHI_WEIGHTS

    if args.chunker:
        os.environ["CHUNKING_METHOD"] = args.chunker
    if args.prompter:
        os.environ["PROMPTING_STRATEGY"] = args.prompter

    from src.core.unified_config import reload_config
    os.environ.setdefault("PROMPTING_STRATEGY", "P3")
    os.environ.setdefault("CHUNKING_METHOD", "dsc")
    config = reload_config()
    model_id = config.llm_model_id or "unknown"
    chunker = (args.chunker or config.chunking_method or "dsc").lower()
    prompter = (args.prompter or config.prompting_strategy or "P3").upper()

    n_docs = len(pairs)
    n_runs = n_docs * len(seeds)
    print(f"Plan: {n_docs} docs x {len(seeds)} seeds = {n_runs} runs")
    print(f"Model: {model_id}  Chunker: {chunker}  Prompter: {prompter}")
    print(f"Seeds: {seeds}")
    print(f"Phi schemes: {phi_schemes}")
    for gf, _ in pairs:
        print(f"  {gf.stem}")

    if args.dry_run:
        print("\n[dry-run] No extraction performed.")
        return 0

    from src.evaluation.metrics import prepare_evaluator
    preprocessor, embedder = prepare_evaluator()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"results_detail_{timestamp}.csv"
    summary_path = out_dir / f"results_summary_{timestamp}.csv"

    detail_rows: list[dict[str, Any]] = []

    with detail_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()

        for gf, tf in pairs:
            doc_id = gf.stem
            gold = json.loads(gf.read_text(encoding="utf-8"))
            for seed in seeds:
                print(f"  {doc_id}  seed={seed} ...", end=" ", flush=True)
                try:
                    pred = asyncio.run(_extract_one(tf, doc_id, seed))
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
                detail_rows.append(row)
                writer.writerow(row)
                fh.flush()

                phi_str = f"Phi={row['Phi']:.3f}" if row["Phi"] is not None else "Phi=N/A"
                print(phi_str)

    print(f"\nDetail CSV: {detail_path}")

    # Paired bootstrap if comparison CSV provided
    bootstrap_results: dict[str, float] = {}
    if args.compare_against:
        prior_rows = _load_detail_csv(args.compare_against)
        print(f"\nComparing against: {args.compare_against}")
        # Align by doc_id, use mean per doc over seeds
        def _doc_means(rows: list[dict]) -> dict[str, float]:
            by_doc: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                val = r.get("Phi")
                if val not in (None, "", "None"):
                    try:
                        by_doc[r["doc_id"]].append(float(val))
                    except (ValueError, TypeError):
                        pass
            return {doc: float(np.mean(vals)) for doc, vals in by_doc.items()}

        current_means = _doc_means(detail_rows)
        prior_means = _doc_means(prior_rows)
        shared_docs = sorted(set(current_means) & set(prior_means))
        if len(shared_docs) >= 2:
            curr_scores = [current_means[d] for d in shared_docs]
            prior_scores = [prior_means[d] for d in shared_docs]
            pval = paired_bootstrap_pvalue(curr_scores, prior_scores, args.bootstrap_n)
            bootstrap_results["Phi_pvalue"] = pval
            print(f"  Paired bootstrap Phi p-value: {pval:.4f}  (n_docs={len(shared_docs)}, n_boot={args.bootstrap_n})")
            print(f"  (Dror et al. 2018 procedure; H0: mean Phi difference == 0)")
        else:
            print(f"  WARNING: only {len(shared_docs)} shared docs — bootstrap skipped (need ≥ 2).")

    # Build summary CSV
    summary_rows = _summary_rows(detail_rows, phi_schemes)
    for row in summary_rows:
        row.update(bootstrap_results)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV: {summary_path}")

    print()
    _print_summary_table(summary_rows)
    print(f"\nNote: CI via scipy.stats.t.ppf(0.975, df=n-1). n = docs x seeds.")
    print(f"      Phi sensitivity across {len(phi_schemes)} weight scheme(s): {phi_schemes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
