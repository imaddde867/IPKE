#!/usr/bin/env python3
"""Chunking + prompting experiment runner for IPKE.

Example:
    python scripts/run_experiments.py --config configs/chunking_grid_core.yaml \
        --out-root logs/chunking_grid_core --evaluate true
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.unified_config import UnifiedConfig
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from scripts.experiment_lib import (
    EvaluationContext,
    apply_llm_settings,
    apply_prompt_settings,
    bool_arg,
    ensure_out_root,
    evaluate_document,
    load_spec,
    normalize_documents,
    prepare_evaluation_context,
    process_document,
    serialize_config_snapshot,
    write_summary_csv,
)

LOGGER = logging.getLogger("run_experiments")


def apply_chunk_parameters(config: UnifiedConfig, experiment: Dict[str, Any]) -> None:
    method = experiment.get("method")
    if not method:
        raise ValueError(f"Chunking experiment missing 'method': {experiment}")
    config.chunking_method = method.lower()
    params = experiment.get("params", {})
    if "chunk_max_chars" in params:
        config.chunk_max_chars = int(params["chunk_max_chars"])
    if config.chunking_method == "fixed":
        if "chunk_size_chars" in params:
            size = int(params["chunk_size_chars"])
            config.chunk_size = size
            config.chunk_max_chars = size
        if "chunk_overlap_chars" in params:
            config.chunk_overlap_chars = max(0, int(params["chunk_overlap_chars"]))
        if "chunk_overlap_dedup_ratio" in params:
            config.chunk_overlap_dedup_ratio = float(params["chunk_overlap_dedup_ratio"])
    else:
        if "embedding_model_path" in params:
            config.embedding_model_path = params["embedding_model_path"]
        if "sem_lambda" in params or "sem_min_sentences_per_chunk" in params:
            if "sem_lambda" in params:
                config.sem_lambda = float(params["sem_lambda"])
            if "sem_window_w" in params:
                config.sem_window_w = int(params["sem_window_w"])
            if "sem_min_sentences_per_chunk" in params:
                config.sem_min_sentences_per_chunk = int(params["sem_min_sentences_per_chunk"])
            if "sem_max_sentences_per_chunk" in params:
                config.sem_max_sentences_per_chunk = int(params["sem_max_sentences_per_chunk"])
            if "sem_preferred_sentences_per_chunk" in params:
                config.sem_preferred_sentences_per_chunk = int(params["sem_preferred_sentences_per_chunk"])
    if config.chunking_method in {"dual_semantic", "dsc"}:
        parent_block = int(params.get("dsc_parent_block_sentences", config.dsc_parent_min_sentences))
        config.dsc_parent_min_sentences = max(1, parent_block)
        config.dsc_parent_max_sentences = max(config.dsc_parent_min_sentences * 2, parent_block)
        if "dsc_child_sem_lambda" in params:
            config.sem_lambda = float(params["dsc_child_sem_lambda"])
        if "dsc_child_window_w" in params:
            config.sem_window_w = int(params["dsc_child_window_w"])
        if "dsc_dynamic_threshold_k" in params:
            config.dsc_threshold_k = float(params["dsc_dynamic_threshold_k"])
        if "dsc_use_headings" in params:
            config.dsc_use_headings = bool(params["dsc_use_headings"])
    if "enable_chunk_dedup" in params:
        config.enable_chunk_dedup = bool(params["enable_chunk_dedup"])
    if "chunk_dedup_threshold" in params:
        config.chunk_dedup_threshold = float(params["chunk_dedup_threshold"])
    if "chunk_dedup_overlap_ratio" in params:
        config.chunk_dedup_overlap_ratio = float(params["chunk_dedup_overlap_ratio"])
    if "chunk_dedup_min_unique_chars" in params:
        config.chunk_dedup_min_unique_chars = int(params["chunk_dedup_min_unique_chars"])
    if "chunk_dedup_embedding_model" in params:
        config.chunk_dedup_embedding_model = params["chunk_dedup_embedding_model"]


async def run_chunking_experiment(
    experiment: Dict[str, Any],
    base_config: UnifiedConfig,
    docs: List[Dict[str, Any]],
    out_root: Path,
    evaluate_flag: bool,
    eval_ctx: Optional[EvaluationContext],
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(base_config)
    apply_chunk_parameters(config, experiment)
    exp_name = experiment.get("name") or experiment.get("method")
    experiment_dir = out_root / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config_snapshot.json").write_text(
        json.dumps(config.__dict__, indent=2, default=str), encoding="utf-8"
    )
    processor = StreamlinedDocumentProcessor(config)
    summary_rows: List[Dict[str, Any]] = []

    for doc in docs:
        doc_out_dir = experiment_dir / doc["id"]
        result, prediction_path = await process_document(processor, doc, doc_out_dir)
        metrics = None
        if evaluate_flag:
            metrics = evaluate_document(doc, prediction_path, eval_ctx)
        summary_row = {
            "experiment": exp_name,
            "chunk_method": config.chunking_method,
            "document_id": doc["id"],
            "document_type": doc.get("type", "unknown"),
        }
        if metrics:
            summary_row.update(metrics)
        summary_rows.append(summary_row)

    processor.knowledge_engine.clear_cache()
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute multi-chunk experiments (e.g., python scripts/run_experiments.py "
            "--config configs/chunking_grid_core.yaml --out-root logs/chunking_grid_core --evaluate true)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment configuration (YAML/JSON), e.g., configs/chunking_grid_core.yaml.",
    )
    parser.add_argument("--out-root", default=None, help="Optional override for the output directory.")
    parser.add_argument("--evaluate", default="true", help="Whether to run evaluation (true/false).")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    spec_path = Path(args.config)
    if not spec_path.exists():
        raise FileNotFoundError(f"Config not found: {spec_path}")
    spec = load_spec(spec_path)
    out_root = ensure_out_root(spec, args.out_root, ROOT)

    docs = normalize_documents(spec, ROOT)
    chunking_experiments = spec.get("chunking_experiments")
    if not chunking_experiments:
        raise ValueError("Config must include a 'chunking_experiments' list.")

    base_config = UnifiedConfig.from_environment()
    apply_llm_settings(base_config, spec.get("llm", {}), spec_path.parent)
    apply_prompt_settings(base_config, spec.get("prompting", {}))
    serialize_config_snapshot(base_config, out_root / "base_config_snapshot.json")

    evaluation_flag = bool_arg(args.evaluate, default=True)
    eval_ctx = prepare_evaluation_context(spec.get("evaluation", {})) if evaluation_flag else None

    all_rows: List[Dict[str, Any]] = []
    for experiment in chunking_experiments:
        rows = await run_chunking_experiment(
            experiment=experiment,
            base_config=base_config,
            docs=docs,
            out_root=out_root,
            evaluate_flag=evaluation_flag,
            eval_ctx=eval_ctx,
        )
        all_rows.extend(rows)

    if evaluation_flag:
        fieldnames = [
            "experiment",
            "chunk_method",
            "document_id",
            "document_type",
            "StepF1",
            "AdjacencyF1",
            "Kendall",
            "ConstraintCoverage",
            "ConstraintAttachmentF1",
            "A_score",
            "GraphF1",
            "NEXT_EdgeF1",
            "Logic_EdgeF1",
            "ConstraintAttachmentF1_TierB",
            "B_score",
        ]
        write_summary_csv(out_root, all_rows, fieldnames)
    LOGGER.info("Artifacts available under %s", out_root)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
