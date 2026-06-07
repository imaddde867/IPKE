#!/usr/bin/env python3
"""
Multi-seed, multi-model evaluation sweep for IPKE paper (P0.2 + P0.4).

Covers three model families in the 8B class (Mistral-7B as thesis anchor,
Qwen3-8B as SOTA 2025, Llama-3.1-8B as Meta family), each evaluated under
DSC+P3 (primary) and ablation configs, across N random seeds.

95% CI computed via paired bootstrap. No Docker required.

Model download commands (run on core/5090 first):
  # Mistral-7B-Instruct-v0.2 (thesis anchor)
  huggingface-cli download bartowski/Mistral-7B-Instruct-v0.2-GGUF \\
      --include "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf" --local-dir models/llm/

  # Qwen3-8B (SOTA 2025 — best 8B as of May 2026)
  huggingface-cli download Qwen/Qwen3-8B-GGUF \\
      --include "Qwen3-8B-Q4_K_M.gguf" --local-dir models/llm/

  # Llama-3.1-8B-Instruct (Meta family, imatrix quant)
  huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
      --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --local-dir models/llm/

Usage (dry-run, prints plan only):
    python scripts/experiments/multi_seed_sweep.py --dry-run

Usage (single model, resume-safe):
    python scripts/experiments/multi_seed_sweep.py \\
        --configs mistral7b_dsc_p3 --seeds 42 123 456 --skip-existing

Usage (full sweep — run overnight on 3090):
    GPU_BACKEND=cuda python scripts/experiments/multi_seed_sweep.py \\
        --seeds 42 123 456

Outputs (all under --output-root/multi_seed_<timestamp>/):
    run_config.json          model, seeds, git sha, hardware (ECIR reproducibility)
    summary.csv              paper table: config x metric (mean, std, ci_lo, ci_hi)
    per_doc_summary.csv      same broken out per document
    aggregated_metrics.json  raw per-seed metrics for reevaluate_metrics.py
    <config_id>/
        seed_<seed>/              flat + tierb prediction JSONs
        metrics_seed_<seed>.json  Tier-A scores for that seed

Note on Qwen3 thinking mode (pending):
    Qwen3 supports a thinking mode (chain-of-thought before answering).
    The SweepConfig.think_mode field records the intent for each config,
    but the control tokens (/no_think, /think) are not yet wired into the
    prompting path — all configs currently run without think-mode control.
    See the TODO comment in _extract_one for implementation notes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logging_config import get_logger
from src.core.unified_config import reload_config
from src.pipelines.baseline import (
    TIER_A_TEST_DOCS,
    DEFAULT_GOLD_DIR,
    extract_documents,
    evaluate_predictions,
)
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from scripts.experiments.experiment_utils import current_git_sha

LOGGER = get_logger(__name__)

# Metrics reported in paper tables (Phi is primary).
PAPER_METRICS = ["Phi", "StepF1", "AdjacencyF1", "Kendall", "ConstraintCoverage"]

N_BOOTSTRAP = 10_000
BOOTSTRAP_RNG_SEED = 0  # fixed so CI values are reproducible even without a model

DEFAULT_SEEDS = [42, 123, 456]

# Default model paths (relative to REPO_ROOT). Override via --model-dir or per-config.
_MODEL_DIR = "models/llm"
_MISTRAL_GGUF = f"{_MODEL_DIR}/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"
_QWEN3_GGUF   = f"{_MODEL_DIR}/Qwen3-8B-Q4_K_M.gguf"
_LLAMA31_GGUF = f"{_MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """One experimental configuration evaluated across all seeds."""

    name: str
    chunking_method: str       # "dual_semantic" | "fixed" | "breakpoint_semantic"
    prompting_strategy: str    # "P3" | "P0" | "P1" | "P2"
    model_path: str = ""       # path to GGUF (relative to REPO_ROOT or absolute); empty = use env
    llm_backend: str = "llama_cpp"   # "llama_cpp" | "transformers"
    think_mode: Optional[bool] = None  # None = don't touch; False = /no_think; True = /think

    @property
    def config_id(self) -> str:
        return self.name

    @property
    def model_label(self) -> str:
        """Short human-readable model name derived from the path, for logging."""
        if not self.model_path:
            return "default"
        return Path(self.model_path).stem


# ---------------------------------------------------------------------------
# Default config matrix — three model families x ablations
# ---------------------------------------------------------------------------
#
# Paper Table 1:
#   Rows    : DSC+P3 | Fixed+P3 | DSC+P0 | Fixed+P0
#   Columns : Mistral-7B (anchor) | Qwen3-8B no_think | Qwen3-8B think | Llama-3.1-8B
#
# Think-mode configs are a novel contribution: no prior work evaluates industrial
# SOP extraction with LLM chain-of-thought reasoning in a two-stage pipeline.
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: List[SweepConfig] = [
    # ---- Mistral-7B-Instruct-v0.2  (thesis anchor — must reproduce Phi=0.699) ----
    SweepConfig("mistral7b_dsc_p3",   "dual_semantic", "P3",  model_path=_MISTRAL_GGUF),
    SweepConfig("mistral7b_fixed_p3", "fixed",         "P3",  model_path=_MISTRAL_GGUF),
    SweepConfig("mistral7b_dsc_p0",   "dual_semantic", "P0",  model_path=_MISTRAL_GGUF),
    SweepConfig("mistral7b_fixed_p0", "fixed",         "P0",  model_path=_MISTRAL_GGUF),

    # ---- Qwen3-8B  (SOTA 2025, non-thinking — fair comparison to Mistral) ----
    SweepConfig("qwen3_8b_dsc_p3",    "dual_semantic", "P3",  model_path=_QWEN3_GGUF, think_mode=False),
    SweepConfig("qwen3_8b_dsc_p0",    "dual_semantic", "P0",  model_path=_QWEN3_GGUF, think_mode=False),
    SweepConfig("qwen3_8b_fixed_p0",  "fixed",         "P0",  model_path=_QWEN3_GGUF, think_mode=False),

    # ---- Qwen3-8B thinking mode  (pending: requires LLM_PROMPT_PREFIX consumer in prompting path) ----
    # SweepConfig("qwen3_8b_think_dsc_p3", "dual_semantic", "P3", model_path=_QWEN3_GGUF, think_mode=True),

    # ---- Llama-3.1-8B-Instruct  (Meta family, broadens comparative coverage) ----
    SweepConfig("llama31_8b_dsc_p3",   "dual_semantic", "P3",  model_path=_LLAMA31_GGUF),
    SweepConfig("llama31_8b_fixed_p0", "fixed",         "P0",  model_path=_LLAMA31_GGUF),
]

_CONFIG_MAP: Dict[str, SweepConfig] = {c.name: c for c in DEFAULT_CONFIGS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed IPKE sweep — paper P0.2 (95%% CI via bootstrap).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="LLM random seeds to evaluate.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[c.name for c in DEFAULT_CONFIGS],
        choices=list(_CONFIG_MAP),
        metavar="CONFIG",
        help=f"Configs to run. Choices: {list(_CONFIG_MAP)}",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=None,
        metavar="DOC_ID",
        help="Document IDs to include (default: all three Tier-A docs).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Root directory for all run outputs.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=REPO_ROOT / DEFAULT_GOLD_DIR,
        help="Gold Tier-A annotation directory.",
    )
    # skip-existing is on by default; --no-skip-existing forces re-extraction.
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip extraction if all prediction JSONs for a (config, seed) already exist.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-extraction even when predictions are already on disk.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap samples for 95%% CI computation.",
    )
    parser.add_argument(
        "--gpu-backend",
        default=None,
        metavar="BACKEND",
        help="Override GPU_BACKEND env var (cuda | metal | cpu).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Override the models/ directory used to resolve relative model paths. "
            "Useful when models are stored outside the repo (e.g. ~/models/llm/)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and exit without extracting or evaluating.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Continue and write CSVs even when some (config, seed) evaluations fail. "
            "Partial runs are marked with n/a metrics. "
            "Without this flag, any evaluation failure exits non-zero."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hardware info (ECIR reproducibility requirement)
# ---------------------------------------------------------------------------


def collect_hardware_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip().split("\n")[0]
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------
# Run config (written immediately; survives interrupted runs)
# ---------------------------------------------------------------------------


def write_run_config(
    run_dir: Path,
    seeds: List[int],
    configs: List[SweepConfig],
    doc_ids: List[str],
    args: argparse.Namespace,
) -> None:
    # Read current LLM settings from env before any seed injection.
    cfg = reload_config()
    # Build per-config model provenance (ECIR Section 4 requirement).
    config_provenance = []
    for c in configs:
        abs_path = _resolve_model_path(c.model_path, model_dir=args.model_dir)
        config_provenance.append({
            "name": c.name,
            "chunking_method": c.chunking_method,
            "prompting_strategy": c.prompting_strategy,
            "model_path": abs_path or "(env default)",
            "model_file": Path(abs_path).name if abs_path else "(env default)",
            "llm_backend": c.llm_backend,
        })
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "configs": config_provenance,
        "documents": doc_ids,
        "n_bootstrap": args.n_bootstrap,
        "skip_existing": args.skip_existing,
        # Global defaults (may be overridden per config)
        "default_llm_model_path": cfg.llm_model_path,
        "default_llm_temperature": cfg.llm_temperature,
        "default_llm_n_ctx": cfg.llm_n_ctx,
        "hardware": collect_hardware_info(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Run config written: %s", run_dir / "run_config.json")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str, model_dir: Path | None = None) -> str:
    """Return an absolute model path.

    With model_dir: joins model_dir with the filename only (strips any repo-relative
    prefix such as models/llm/) so --model-dir ~/models/llm works as documented.
    Without model_dir: anchors the full relative path at REPO_ROOT.
    Absolute paths pass through unchanged.
    """
    if not model_path:
        return ""
    p = Path(model_path)
    if p.is_absolute():
        return str(p)
    if model_dir is not None:
        return str(model_dir / p.name)
    return str(REPO_ROOT / p)


async def _extract_one(
    sweep_cfg: SweepConfig,
    seed: int,
    doc_sources: Dict[str, Path],
    pred_dir: Path,
    skip_existing: bool,
    gpu_backend: Optional[str],
    model_dir: Path | None = None,
) -> None:
    """Extract PKG predictions for one (config, seed) combination."""
    if skip_existing and all(
        (pred_dir / f"{doc_id}.json").exists() for doc_id in doc_sources
    ):
        LOGGER.info(
            "Skipping extraction (all predictions exist): config=%s seed=%d",
            sweep_cfg.config_id,
            seed,
        )
        return

    LOGGER.info(
        "Extracting: config=%s  model=%s  seed=%d",
        sweep_cfg.config_id,
        sweep_cfg.model_label,
        seed,
    )

    # --- Core env vars (chunking, prompting, seed) ---
    os.environ["CHUNKING_METHOD"] = sweep_cfg.chunking_method
    os.environ["PROMPTING_STRATEGY"] = sweep_cfg.prompting_strategy
    os.environ["LLM_RANDOM_SEED"] = str(seed)
    os.environ["LLM_BACKEND"] = sweep_cfg.llm_backend

    # --- Model selection ---
    abs_model_path = _resolve_model_path(sweep_cfg.model_path, model_dir=model_dir)
    if abs_model_path:
        if not Path(abs_model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {abs_model_path}\n"
                f"  Download it with the huggingface-cli command in the script header."
            )
        os.environ["LLM_MODEL_PATH"] = abs_model_path

    # --- GPU backend ---
    if gpu_backend:
        os.environ["GPU_BACKEND"] = gpu_backend
        if gpu_backend.lower() in {"cuda", "metal"}:
            os.environ.setdefault("ENABLE_GPU", "1")

    # think_mode is not yet wired into the prompting path (LLM_PROMPT_PREFIX has no consumer
    # in src/ai/prompting/). All configs currently run without think-mode control tokens.
    # When prompt-prefix support is added, restore the LLM_PROMPT_PREFIX logic here.

    config = reload_config()
    processor = StreamlinedDocumentProcessor(config=config)

    await extract_documents(
        processor=processor,
        doc_sources=doc_sources,
        run_dir=pred_dir,
    )
    LOGGER.info("Extraction complete: config=%s seed=%d -> %s", sweep_cfg.config_id, seed, pred_dir)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_one(
    sweep_cfg: SweepConfig,
    seed: int,
    pred_dir: Path,
    gold_dir: Path,
    out_path: Path,
) -> Optional[Dict[str, Any]]:
    """Evaluate one (config, seed) pair; returns the metrics dict or None on failure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Evaluating: config=%s  seed=%d", sweep_cfg.config_id, seed)
    try:
        metrics = evaluate_predictions(
            pred_dir=pred_dir,
            out_path=out_path,
            gold_dir=gold_dir,
            tier="A",
        )
        LOGGER.info("Evaluation done: config=%s seed=%d", sweep_cfg.config_id, seed)
        return metrics
    except Exception as exc:
        LOGGER.error(
            "Evaluation failed for config=%s seed=%d: %s",
            sweep_cfg.config_id,
            seed,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int,
    ci: float = 0.95,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> Tuple[float, float]:
    """Return (lower, upper) confidence interval bounds via percentile bootstrap."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = np.random.default_rng(rng_seed)
    bootstrap_samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    hi = float(np.percentile(bootstrap_means, (1.0 - alpha / 2) * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    all_metrics: Dict[str, Dict[int, Optional[Dict[str, Any]]]],
    seeds: List[int],
    configs: List[SweepConfig],
    n_bootstrap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Aggregate per-seed metrics across seeds per config.

    Returns:
        summary_rows: one row per config, macro-averaged over docs.
        per_doc_rows: one row per (config, doc_id).
    """
    summary_rows: List[Dict[str, Any]] = []
    per_doc_rows: List[Dict[str, Any]] = []

    for cfg in configs:
        cid = cfg.config_id
        macro_by_metric: Dict[str, List[float]] = {m: [] for m in PAPER_METRICS}
        per_doc_collector: Dict[str, Dict[str, List[float]]] = {}

        for seed in seeds:
            seed_result = all_metrics.get(cid, {}).get(seed)
            if seed_result is None:
                LOGGER.warning("No metrics recorded for config=%s seed=%d", cid, seed)
                continue

            macro = seed_result.get("macro_avg") or {}
            for metric in PAPER_METRICS:
                val = macro.get(metric)
                if val is not None:
                    macro_by_metric[metric].append(float(val))

            for doc_id, doc_metrics in seed_result.items():
                if doc_id in ("macro_avg", "micro_avg") or not isinstance(doc_metrics, dict):
                    continue
                bucket = per_doc_collector.setdefault(doc_id, {m: [] for m in PAPER_METRICS})
                for metric in PAPER_METRICS:
                    val = doc_metrics.get(metric)
                    if val is not None:
                        bucket[metric].append(float(val))

        # Build macro summary row.
        n_completed = max((len(v) for v in macro_by_metric.values()), default=0)
        row: Dict[str, Any] = {"config": cid, "n_seeds": n_completed}
        for metric in PAPER_METRICS:
            vals = macro_by_metric[metric]
            if not vals:
                for suffix in ("mean", "std", "ci_lo", "ci_hi"):
                    row[f"{metric}_{suffix}"] = None
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(vals, n_bootstrap)
            row[f"{metric}_mean"] = round(mean_val + 1e-12, 4)
            row[f"{metric}_std"] = round(std_val + 1e-12, 4)
            row[f"{metric}_ci_lo"] = round(ci_lo + 1e-12, 4)
            row[f"{metric}_ci_hi"] = round(ci_hi + 1e-12, 4)
        summary_rows.append(row)

        # Build per-doc rows.
        for doc_id, metric_values in per_doc_collector.items():
            doc_n = max((len(v) for v in metric_values.values()), default=0)
            doc_row: Dict[str, Any] = {
                "config": cid,
                "doc_id": doc_id,
                "n_seeds": doc_n,
            }
            for metric in PAPER_METRICS:
                vals = metric_values.get(metric, [])
                if not vals:
                    doc_row[f"{metric}_mean"] = None
                    doc_row[f"{metric}_std"] = None
                    continue
                doc_row[f"{metric}_mean"] = round(float(np.mean(vals)) + 1e-12, 4)
                doc_row[f"{metric}_std"] = (
                    round(float(np.std(vals, ddof=1)) + 1e-12, 4) if len(vals) > 1 else 0.0
                )
            per_doc_rows.append(doc_row)

    return summary_rows, per_doc_rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write; skipping %s", path)
        return
    headers = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows -> %s", len(rows), path)


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------


def print_run_plan(
    configs: List[SweepConfig],
    seeds: List[int],
    doc_ids: List[str],
    run_dir: Path,
    model_dir: Path | None = None,
) -> None:
    total = len(configs) * len(seeds) * len(doc_ids)
    print("\n=== Multi-Seed Multi-Model Sweep Plan ===")
    print(f"  Run dir  : {run_dir}")
    print(f"  Seeds    : {seeds}")
    print(f"  Docs     : {doc_ids}")
    print(f"  Total    : {len(configs)} configs x {len(seeds)} seeds x {len(doc_ids)} docs = {total} LLM extractions")
    print()
    print(f"  {'Config':<30} {'Model':<42} {'Chunker':<16} {'Strategy':<8} {'Think'}")
    print(f"  {'-'*30} {'-'*42} {'-'*16} {'-'*8} {'-'*7}")
    for c in configs:
        abs_path = _resolve_model_path(c.model_path, model_dir=model_dir)
        model_label = Path(abs_path).name if abs_path else "(env default)"
        think_str = {True: "on", False: "off", None: "n/a"}[c.think_mode]
        exists_str = "" if not abs_path else (" [OK]" if Path(abs_path).exists() else " [MISSING]")
        print(f"  {c.name:<30} {model_label:<42}{exists_str:<10} {c.chunking_method:<16} {c.prompting_strategy:<8} {think_str}")
    print()
    missing = [
        c for c in configs
        if c.model_path and not Path(_resolve_model_path(c.model_path, model_dir=model_dir)).exists()
    ]
    if missing:
        print("  WARNING: missing model files (see download commands in script header):")
        for c in missing:
            print(f"    {_resolve_model_path(c.model_path, model_dir=model_dir)}")
    print("=========================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    args = parse_args()

    # Resolve selected configs.
    selected_configs = [_CONFIG_MAP[n] for n in args.configs]

    # Resolve document sources (absolute paths).
    doc_sources: Dict[str, Path] = {}
    for doc_id, rel_path in TIER_A_TEST_DOCS.items():
        if args.documents and doc_id not in args.documents:
            continue
        abs_path = (REPO_ROOT / rel_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Document not found: {abs_path}")
        doc_sources[doc_id] = abs_path

    if not doc_sources:
        raise ValueError("No documents selected. Check --documents or ensure Tier-A docs exist.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"multi_seed_{timestamp}"

    if args.dry_run:
        print_run_plan(selected_configs, args.seeds, list(doc_sources.keys()), run_dir, model_dir=args.model_dir)
        return

    # Write run config immediately so a crashed run can be diagnosed.
    write_run_config(run_dir, args.seeds, selected_configs, list(doc_sources.keys()), args)

    # ------------------------------------------------------------------
    # Phase 1: Extractions  (config x seed)
    # ------------------------------------------------------------------
    all_metrics: Dict[str, Dict[int, Optional[Dict[str, Any]]]] = {
        c.config_id: {} for c in selected_configs
    }

    for sweep_cfg in selected_configs:
        for seed in args.seeds:
            pred_dir = run_dir / sweep_cfg.config_id / f"seed_{seed}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            asyncio.run(
                _extract_one(
                    sweep_cfg=sweep_cfg,
                    seed=seed,
                    doc_sources=doc_sources,
                    pred_dir=pred_dir,
                    skip_existing=args.skip_existing,
                    gpu_backend=args.gpu_backend,
                    model_dir=args.model_dir,
                )
            )

    # ------------------------------------------------------------------
    # Phase 2: Evaluation  (config x seed)
    # ------------------------------------------------------------------
    for sweep_cfg in selected_configs:
        for seed in args.seeds:
            pred_dir = run_dir / sweep_cfg.config_id / f"seed_{seed}"
            out_path = run_dir / sweep_cfg.config_id / f"metrics_seed_{seed}.json"
            metrics = _evaluate_one(
                sweep_cfg=sweep_cfg,
                seed=seed,
                pred_dir=pred_dir,
                gold_dir=args.gold_dir,
                out_path=out_path,
            )
            all_metrics[sweep_cfg.config_id][seed] = metrics

    # Fail loudly if any evaluation failed, unless partial results are explicitly allowed.
    failed_pairs = [
        (cfg_id, seed)
        for cfg_id, seed_map in all_metrics.items()
        for seed, val in seed_map.items()
        if val is None
    ]
    if failed_pairs and not args.allow_partial:
        LOGGER.error(
            "Evaluation failed for %d (config, seed) pair(s). "
            "Re-run with --allow-partial to write incomplete tables.",
            len(failed_pairs),
        )
        for cfg_id, seed in failed_pairs:
            LOGGER.error("  config=%s  seed=%d", cfg_id, seed)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: Aggregate + write tables
    # ------------------------------------------------------------------
    summary_rows, per_doc_rows = aggregate_results(
        all_metrics=all_metrics,
        seeds=args.seeds,
        configs=selected_configs,
        n_bootstrap=args.n_bootstrap,
    )

    write_csv(summary_rows, run_dir / "summary.csv")
    write_csv(per_doc_rows, run_dir / "per_doc_summary.csv")

    # Persist raw metrics for reevaluate_metrics.py re-scoring.
    (run_dir / "aggregated_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, default=str), encoding="utf-8"
    )

    LOGGER.info("Done. Run dir: %s", run_dir)

    # Print summary table to stdout for quick inspection.
    if summary_rows:
        print(f"\n{'Config':<22} {'Phi':>8} {'[95% CI]':>20}  {'StepF1':>8}  {'Coverage':>10}")
        print("-" * 75)
        for row in summary_rows:
            phi_mean = row.get("Phi_mean")
            phi_lo = row.get("Phi_ci_lo")
            phi_hi = row.get("Phi_ci_hi")
            sf1 = row.get("StepF1_mean")
            cov = row.get("ConstraintCoverage_mean")
            phi_str = f"{phi_mean:.4f}" if phi_mean is not None else "  n/a "
            ci_str = (
                f"[{phi_lo:.4f}, {phi_hi:.4f}]" if phi_lo is not None else "         n/a        "
            )
            sf1_str = f"{sf1:.4f}" if sf1 is not None else "  n/a "
            cov_str = f"{cov:.4f}" if cov is not None else "   n/a  "
            print(f"  {row['config']:<20} {phi_str:>8} {ci_str:>20}  {sf1_str:>8}  {cov_str:>10}")
        print()


if __name__ == "__main__":
    main()
