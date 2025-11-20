#!/usr/bin/env python3
"""Prompting experiment runner for IPKE.

Example:
    python scripts/run_prompting_experiments.py --config configs/prompting_grid.yaml \
        --out-root logs/prompting_grid --evaluate true
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.unified_config import UnifiedConfig
from src.processors.streamlined_processor import ProcessingResult, StreamlinedDocumentProcessor
from src.ai.types import ExtractionResult, ExtractedEntity
from tools import evaluate as evaluator

LOGGER = logging.getLogger("run_prompting_experiments")


@dataclass
class EvaluationContext:
    preprocessor: evaluator.TextPreprocessor
    embedder: evaluator.EmbeddingCache
    threshold: float
    tiers: Tuple[str, ...]


def load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("PyYAML is required for YAML configs.") from exc
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def normalize_documents(spec: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    docs = spec.get("docs") or spec.get("documents")
    if not docs:
        raise ValueError("Config must include a non-empty 'docs' list.")
    normalized: List[Dict[str, Any]] = []
    for entry in docs:
        if not isinstance(entry, dict):
            raise ValueError(f"Document entries must be objects with id/path: {entry}")
        if "id" not in entry or "path" not in entry:
            raise ValueError(f"Document entries require 'id' and 'path': {entry}")
        normalized.append(
            {
                "id": entry["id"],
                "path": str(resolve_path(entry["path"], base_dir)),
                "gold": str(resolve_path(entry["gold"], base_dir)) if entry.get("gold") else None,
                "gold_tier_a": str(resolve_path(entry["gold_tier_a"], base_dir)) if entry.get("gold_tier_a") else None,
                "gold_tier_b": str(resolve_path(entry["gold_tier_b"], base_dir)) if entry.get("gold_tier_b") else None,
                "type": entry.get("type", "unknown"),
            }
        )
    return normalized


def entity_to_dict(entity: ExtractedEntity) -> Dict[str, Any]:
    return {
        "content": entity.content,
        "entity_type": entity.entity_type,
        "category": entity.category,
        "confidence": entity.confidence,
        "context": entity.context,
        "metadata": entity.metadata,
    }


def extraction_to_dict(result: ExtractionResult) -> Dict[str, Any]:
    return {
        "steps": result.steps,
        "constraints": result.constraints,
        "entities": [entity_to_dict(ent) for ent in result.entities],
        "confidence_score": result.confidence_score,
        "processing_time": result.processing_time,
        "strategy_used": result.strategy_used,
        "quality_metrics": result.quality_metrics,
        "metadata": result.metadata,
    }


def ensure_out_root(spec: Dict[str, Any], cli_out_root: Optional[str]) -> Path:
    if cli_out_root:
        out_root = Path(cli_out_root)
    elif spec.get("out_root"):
        out_root = Path(spec["out_root"])
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / "logs" / "experiments" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def bool_arg(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_llm_settings(config: UnifiedConfig, llm_spec: Dict[str, Any], base_dir: Path) -> None:
    if not llm_spec:
        return
    if llm_spec.get("backend"):
        config.llm_backend = llm_spec["backend"]
    if llm_spec.get("model_path"):
        config.llm_model_path = str(resolve_path(llm_spec["model_path"], base_dir))
    if llm_spec.get("n_ctx") is not None:
        config.llm_n_ctx = int(llm_spec["n_ctx"])
    if llm_spec.get("temperature") is not None:
        config.llm_temperature = float(llm_spec["temperature"])
    if llm_spec.get("top_p") is not None:
        config.llm_top_p = float(llm_spec["top_p"])
    if llm_spec.get("max_tokens") is not None:
        config.llm_max_tokens = int(llm_spec["max_tokens"])
    if llm_spec.get("repeat_penalty") is not None:
        config.llm_repeat_penalty = float(llm_spec["repeat_penalty"])
    if llm_spec.get("max_chunks") is not None:
        config.llm_max_chunks = int(llm_spec["max_chunks"])
    if llm_spec.get("num_workers") is not None:
        config.llm_num_workers = int(llm_spec["num_workers"])
    if llm_spec.get("device_strategy"):
        config.llm_device_strategy = llm_spec["device_strategy"]
    if llm_spec.get("random_seed") is not None:
        config.llm_random_seed = int(llm_spec["random_seed"])
    if llm_spec.get("confidence_threshold") is not None:
        config.confidence_threshold = float(llm_spec["confidence_threshold"])
    if llm_spec.get("quality_threshold") is not None:
        config.quality_threshold = float(llm_spec["quality_threshold"])

def apply_chunk_settings(config: UnifiedConfig, chunk_spec: Dict[str, Any]) -> None:
    if not chunk_spec:
        return
    method = chunk_spec.get("method")
    if not method:
        return  # Or raise error if chunking must be defined
    config.chunking_method = method.lower()
    params = chunk_spec.get("params", {})
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

def apply_prompt_settings(config: UnifiedConfig, prompting_spec: Dict[str, Any]) -> None:
    if not prompting_spec:
        return
    if prompting_spec.get("strategy"):
        config.prompting_strategy = prompting_spec["strategy"].upper()

def write_metrics_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

def sanitize_row(row: Dict[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    return {key: row.get(key) for key in columns}

def write_summary_csv(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "experiment",
        "prompting_strategy",
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
    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(sanitize_row(row, fieldnames))
    LOGGER.info("Summary CSV written to %s", summary_path)

def save_prediction(doc_out_dir: Path, result: ProcessingResult) -> Path:
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": result.document_id,
        "document_type": result.document_type,
        "metadata": result.metadata,
        **extraction_to_dict(result.extraction_result),
    }
    prediction_path = doc_out_dir / "predictions.json"
    prediction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return prediction_path

def prepare_evaluation_context(eval_spec: Dict[str, Any]) -> Optional[EvaluationContext]:
    if not eval_spec:
        return None
    tier = eval_spec.get("tier", "both").upper()
    if tier == "A":
        tiers = ("A",)
    elif tier == "B":
        tiers = ("B",)
    else:
        tiers = ("A", "B")
    spacy_model = eval_spec.get("spacy_model", "en_core_web_sm")
    embedding_model = eval_spec.get("embedding_model", "all-mpnet-base-v2")
    device = eval_spec.get("device")
    preprocessor, embedder = evaluator.prepare_evaluator(spacy_model, embedding_model, device=device)
    threshold = float(eval_spec.get("threshold", 0.75))
    return EvaluationContext(preprocessor=preprocessor, embedder=embedder, threshold=threshold, tiers=tiers)


async def process_document(
    processor: StreamlinedDocumentProcessor,
    doc: Dict[str, Any],
    doc_out_dir: Path,
) -> Tuple[ProcessingResult, Path]:
    LOGGER.info("Processing %s (%s)", doc["id"], doc["path"])
    result = await processor.process_document(doc["path"], document_id=doc["id"])
    prediction_path = save_prediction(doc_out_dir, result)
    stats_path = doc_out_dir / "extraction_stats.json"
    stats_payload = {
        "document_id": result.document_id,
        "document_type": result.document_type,
        "processing_time": result.processing_time,
        "file_size": result.file_size,
        "entity_count": len(result.extraction_result.entities),
        "step_count": len(result.extraction_result.steps),
        "constraint_count": len(result.extraction_result.constraints),
        "confidence_score": result.extraction_result.confidence_score,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    return result, prediction_path

def evaluate_document(
    doc: Dict[str, Any],
    prediction_path: Path,
    eval_ctx: Optional[EvaluationContext],
) -> Optional[Dict[str, Any]]:
    if not eval_ctx:
        return None
    metrics: Dict[str, Any] = {}
    tiers = set(eval_ctx.tiers)
    if "A" in tiers:
        gold_a = doc.get("gold_tier_a") or doc.get("gold")
        if not gold_a:
            LOGGER.warning("Missing Tier A gold for %s; skipping Tier A evaluation.", doc["id"])
        else:
            metrics.update(
                evaluator.run_evaluation(
                    gold_a,
                    str(prediction_path),
                    tiers=("A",),
                    threshold=eval_ctx.threshold,
                    preprocessor=eval_ctx.preprocessor,
                    embedder=eval_ctx.embedder,
                )
            )
    if "B" in tiers:
        gold_b = doc.get("gold_tier_b") or doc.get("gold")
        if not gold_b:
            LOGGER.warning("Missing Tier B gold for %s; skipping Tier B evaluation.", doc["id"])
        else:
            tier_b_metrics = evaluator.run_evaluation(
                gold_b,
                str(prediction_path),
                tiers=("B",),
                threshold=eval_ctx.threshold,
                preprocessor=eval_ctx.preprocessor,
                embedder=eval_ctx.embedder,
            )
            for key, value in tier_b_metrics.items():
                metrics[key if key not in metrics else f"{key}_TierB"] = value
    if not metrics:
        return None
    metrics_path = prediction_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


async def run_prompting_experiment(
    experiment: Dict[str, Any],
    base_config: UnifiedConfig,
    docs: List[Dict[str, Any]],
    out_root: Path,
    evaluate_flag: bool,
    eval_ctx: Optional[EvaluationContext],
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(base_config)
    apply_prompt_settings(config, experiment)
    exp_name = experiment.get("name") or experiment.get("strategy")
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
            "prompting_strategy": config.prompting_strategy,
            "document_id": doc["id"],
            "document_type": doc.get("type", "unknown"),
        }
        if metrics:
            summary_row.update(metrics)
        summary_rows.append(summary_row)

    processor.knowledge_engine.clear_cache()
    return summary_rows


def serialize_config_snapshot(config: UnifiedConfig, path: Path) -> None:
    snapshot = config.__dict__.copy()
    snapshot["environment"] = config.environment.value
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute prompting experiments (e.g., python scripts/run_prompting_experiments.py "
            "--config configs/prompting_grid.yaml --out-root logs/prompting_grid --evaluate true)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment configuration (YAML/JSON), e.g., configs/prompting_grid.yaml.",
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
    out_root = ensure_out_root(spec, args.out_root)

    docs = normalize_documents(spec, ROOT)
    prompting_experiments = spec.get("prompting_experiments")
    if not prompting_experiments:
        raise ValueError("Config must include a 'prompting_experiments' list.")

    base_config = UnifiedConfig.from_environment()
    apply_llm_settings(base_config, spec.get("llm", {}), spec_path.parent)
    apply_chunk_settings(base_config, spec.get("chunking", {}))
    serialize_config_snapshot(base_config, out_root / "base_config_snapshot.json")

    evaluation_flag = bool_arg(args.evaluate, default=True)
    eval_ctx = prepare_evaluation_context(spec.get("evaluation", {})) if evaluation_flag else None

    all_rows: List[Dict[str, Any]] = []
    for experiment in prompting_experiments:
        rows = await run_prompting_experiment(
            experiment=experiment,
            base_config=base_config,
            docs=docs,
            out_root=out_root,
            evaluate_flag=evaluation_flag,
            eval_ctx=eval_ctx,
        )
        all_rows.extend(rows)

    if evaluation_flag:
        write_summary_csv(out_root, all_rows)
    LOGGER.info("Artifacts available under %s", out_root)

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()