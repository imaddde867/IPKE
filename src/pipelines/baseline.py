"""Helpers for running the Tier-A baseline extraction + evaluation loops."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from evaluate import main as evaluate_main

from src.processors.streamlined_processor import ProcessingResult, StreamlinedDocumentProcessor
from src.graph.adapter import flat_to_tierb

# Canonical Tier-A documents bundled with the repository (test split)
TIER_A_TEST_DOCS: Dict[str, Path] = {
    "3M_OEM_SOP": Path("datasets/archive/test_data/text/3m_marine_oem_sop.txt"),
    "DOA_Food_Man_Proc_Stor": Path("datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"),
    "op_firesafety_guideline": Path("datasets/archive/test_data/text/op_firesafety_guideline.txt"),
}

DEFAULT_GOLD_DIR = Path("datasets/archive/gold_human")
DEFAULT_EMBEDDING_MODEL = "models/embeddings/all-mpnet-base-v2"


def _normalise_path(path: Path | str, base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        return candidate
    return (base_dir / candidate).resolve()


def extraction_payload(doc_id: str, result: ProcessingResult) -> Dict[str, Any]:
    """Serialise a processing result into the JSON structure expected by the evaluator."""
    extraction = result.extraction_result
    return {
        "document_id": doc_id,
        "document_type": result.document_type,
        "steps": extraction.steps,
        "constraints": extraction.constraints,
        "entities": [asdict(entity) for entity in extraction.entities],
        "confidence_score": extraction.confidence_score,
        "processing_time": result.processing_time,
        "strategy_used": extraction.strategy_used,
        "metadata": result.metadata,
    }


async def extract_documents(
    processor: StreamlinedDocumentProcessor,
    doc_sources: Mapping[str, Path | str] = TIER_A_TEST_DOCS,
    run_dir: Path | str = Path("logs/baseline_runs/run_1"),
    *,
    base_dir: Optional[Path] = None,
    status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Path:
    """Extract structured predictions for the provided documents into ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Also emit Tier-B structured files alongside flat predictions
    tierb_dir = run_dir / "tierb"
    tierb_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, source_path in doc_sources.items():
        resolved = _normalise_path(source_path, base_dir)
        result = await processor.process_document(file_path=str(resolved), document_id=doc_id)
        payload = extraction_payload(doc_id, result)
        (run_dir / f"{doc_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Write Tier-B structured view for the evaluator (nodes + edges with lowercase types)
        tierb_payload = flat_to_tierb(payload)
        (tierb_dir / f"{doc_id}.json").write_text(json.dumps(tierb_payload, indent=2), encoding="utf-8")

        if status_callback:
            status_callback(doc_id, payload)

    return run_dir


def evaluate_predictions(
    pred_dir: Path | str,
    out_path: Path | str,
    *,
    gold_dir: Path | str = DEFAULT_GOLD_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    tier: str = "A",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Run the evaluator against ``pred_dir`` and return parsed metrics."""
    args: list[str] = [
        "--gold_dir",
        str(_normalise_path(gold_dir)),
        "--pred_dir",
        str(_normalise_path(pred_dir)),
        "--tier",
        tier,
        "--embedding_model",
        embedding_model,
        "--out_file",
        str(_normalise_path(out_path)),
    ]
    exit_code = evaluate_main(args)
    if exit_code != 0:
        raise RuntimeError(f"Evaluation failed for {pred_dir} (exit code {exit_code})")
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


def accumulate_metrics(
    accumulator: MutableMapping[str, MutableMapping[str, list[float]]],
    run_metrics: Mapping[str, Mapping[str, Optional[float]]],
) -> None:
    """Append run metrics into ``accumulator`` for later averaging."""
    for doc_id, metrics in run_metrics.items():
        doc_bucket = accumulator.setdefault(doc_id, defaultdict(list))  # type: ignore[arg-type]
        for metric_name, value in metrics.items():
            if value is None:
                continue
            doc_bucket[metric_name].append(float(value))


def _round3(value: float) -> float:
    return round(value + 1e-12, 3)


def summarise_metrics(
    accumulator: Mapping[str, Mapping[str, Iterable[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Average the collected metrics across runs."""
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for doc_id, metrics in accumulator.items():
        summary[doc_id] = {}
        for metric_name, values in metrics.items():
            values = list(values)
            summary[doc_id][metric_name] = _round3(sum(values) / len(values)) if values else None
    return summary
