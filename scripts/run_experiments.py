#!/usr/bin/env python3
"""Configurable experiment runner for IPKE chunks + LLM extraction."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.unified_config import UnifiedConfig
from src.processors.streamlined_processor import ProcessingResult, StreamlinedDocumentProcessor
from src.ai.types import ExtractionResult, ExtractedEntity
from tools import evaluate as evaluator

LOGGER = logging.getLogger("run_experiments")


def load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("PyYAML is required to parse YAML experiment configs.") from exc
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def ensure_run_dir(spec: Dict[str, Any], cli_dir: Optional[str]) -> Path:
    if cli_dir:
        run_dir = Path(cli_dir)
    else:
        candidate = spec.get("run_dir") or spec.get("output_dir")
        if candidate:
            run_dir = Path(candidate)
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_dir = ROOT / "logs" / "experiments" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def _apply_chunking_overrides(config: UnifiedConfig, chunk_spec: Dict[str, Any]) -> None:
    if not chunk_spec:
        return
    method = chunk_spec.get("method")
    if method:
        config.chunking_method = method.lower()
    if "max_chars" in chunk_spec:
        config.chunk_max_chars = int(chunk_spec["max_chars"])
    if "sem_lambda" in chunk_spec:
        config.sem_lambda = float(chunk_spec["sem_lambda"])
    if "sem_window_w" in chunk_spec:
        config.sem_window_w = int(chunk_spec["sem_window_w"])
    if "embedding_model_path" in chunk_spec:
        config.embedding_model_path = chunk_spec["embedding_model_path"]


def _apply_prompt_overrides(config: UnifiedConfig, prompting: Dict[str, Any]) -> None:
    if not prompting:
        return
    strategy = prompting.get("strategy")
    if strategy:
        config.prompting_strategy = strategy.upper()


def _apply_llm_overrides(config: UnifiedConfig, llm_spec: Dict[str, Any]) -> None:
    if not llm_spec:
        return
    backend = llm_spec.get("backend")
    if backend:
        config.llm_backend = backend
    if "num_workers" in llm_spec:
        config.llm_num_workers = int(llm_spec["num_workers"])
    if "device_strategy" in llm_spec:
        config.llm_device_strategy = llm_spec["device_strategy"]
    if "max_tokens" in llm_spec:
        config.llm_max_tokens = int(llm_spec["max_tokens"])
    if "temperature" in llm_spec:
        config.llm_temperature = float(llm_spec["temperature"])
    if "model_path" in llm_spec:
        config.llm_model_path = llm_spec["model_path"]


def apply_overrides(config: UnifiedConfig, spec: Dict[str, Any]) -> None:
    _apply_chunking_overrides(config, spec.get("chunking", {}))
    _apply_prompt_overrides(config, spec.get("prompting", {}))
    _apply_llm_overrides(config, spec.get("llm", {}))
    overrides = spec.get("overrides", {})
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)


def normalize_documents(spec: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    docs = spec.get("documents")
    if not docs:
        raise ValueError("Experiment config must provide a non-empty 'documents' list.")
    normalized: List[Dict[str, Any]] = []
    for entry in docs:
        def resolve_path(raw: str) -> str:
            path = Path(raw)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            return str(path)

        if isinstance(entry, str):
            normalized.append({"path": resolve_path(entry), "id": None})
        elif isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError(f"Document entry is missing 'path': {entry}")
            normalized.append(
                {
                    "path": resolve_path(entry["path"]),
                    "id": entry.get("id"),
                    "type": entry.get("type"),
                }
            )
        else:
            raise ValueError(f"Unsupported document entry: {entry}")
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


def serialize_config(config: UnifiedConfig) -> Dict[str, Any]:
    snapshot = asdict(config)
    snapshot["environment"] = config.environment.value
    return snapshot


def write_metrics(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    metrics_path = run_dir / "logs" / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = run_dir / "logs" / "metrics.csv"
    fieldnames = [
        "document_id",
        "document_type",
        "processing_time",
        "file_size",
        "entity_count",
        "step_count",
        "constraint_count",
        "confidence_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_prediction(pred_dir: Path, result: ProcessingResult) -> Path:
    payload = {
        "document_id": result.document_id,
        "document_type": result.document_type,
        "metadata": result.metadata,
        **extraction_to_dict(result.extraction_result),
    }
    path = pred_dir / f"{result.document_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


async def run_documents(processor: StreamlinedDocumentProcessor, docs: List[Dict[str, Any]], run_dir: Path):
    pred_dir = run_dir / "predictions"
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        LOGGER.info("Processing %s", doc["path"])
        result = await processor.process_document(doc["path"], document_id=doc.get("id"))
        export_prediction(pred_dir, result)
        rows.append(
            {
                "document_id": result.document_id,
                "document_type": result.document_type,
                "processing_time": round(result.processing_time, 4),
                "file_size": result.file_size,
                "entity_count": len(result.extraction_result.entities),
                "step_count": len(result.extraction_result.steps),
                "constraint_count": len(result.extraction_result.constraints),
                "confidence_score": round(result.extraction_result.confidence_score, 4),
            }
        )
    return rows


def maybe_run_evaluation(run_dir: Path, eval_spec: Dict[str, Any]) -> Optional[Path]:
    if not eval_spec:
        return None
    gold_dir = eval_spec.get("gold_dir")
    if not gold_dir:
        LOGGER.warning("evaluation.gold_dir missing; skipping evaluation.")
        return None
    args = [
        "--gold_dir",
        str(gold_dir),
        "--pred_dir",
        str(run_dir / "predictions"),
        "--tier",
        eval_spec.get("tier", "both"),
    ]
    if "threshold" in eval_spec:
        args.extend(["--threshold", str(eval_spec["threshold"])])
    if "spacy_model" in eval_spec:
        args.extend(["--spacy_model", eval_spec["spacy_model"]])
    if "embedding_model" in eval_spec:
        args.extend(["--embedding_model", eval_spec["embedding_model"]])
    if "device" in eval_spec:
        args.extend(["--device", eval_spec["device"]])
    report_path = Path(eval_spec.get("out_file", run_dir / "logs" / "evaluation_report.json"))
    args.extend(["--out_file", str(report_path)])
    LOGGER.info("Running evaluation via tools/evaluate.py")
    rc = evaluator.main(args)
    if rc != 0:
        LOGGER.warning("Evaluation exited with status %s", rc)
        return None
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute extraction experiments from a config file.")
    parser.add_argument("--config", required=True, help="Path to experiment config (JSON or YAML).")
    parser.add_argument("--run-dir", default=None, help="Override run directory.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    spec_path = Path(args.config)
    if not spec_path.exists():
        raise FileNotFoundError(f"Config not found: {spec_path}")
    spec = load_spec(spec_path)
    run_dir = ensure_run_dir(spec, args.run_dir)

    config = UnifiedConfig.from_environment()
    apply_overrides(config, spec)
    (run_dir / "logs" / "config_snapshot.json").write_text(
        json.dumps(serialize_config(config), indent=2), encoding="utf-8"
    )

    documents = normalize_documents(spec, spec_path.parent)
    processor = StreamlinedDocumentProcessor(config)
    metrics_rows = await run_documents(processor, documents, run_dir)
    write_metrics(run_dir, metrics_rows)

    evaluation_spec = dict(spec.get("evaluation", {}))
    if "gold_dir" in evaluation_spec:
        gold_candidate = Path(evaluation_spec["gold_dir"])
        if not gold_candidate.is_absolute():
            evaluation_spec["gold_dir"] = str((spec_path.parent / gold_candidate).resolve())
    if "out_file" in evaluation_spec:
        out_candidate = Path(evaluation_spec["out_file"])
        if not out_candidate.is_absolute():
            evaluation_spec["out_file"] = str((run_dir / out_candidate).resolve())

    eval_path = maybe_run_evaluation(run_dir, evaluation_spec)
    if eval_path:
        LOGGER.info("Saved evaluation summary to %s", eval_path)
    LOGGER.info("Experiment artifacts stored in %s", run_dir)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
