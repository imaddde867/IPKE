#!/usr/bin/env python3
"""
Run a single-shot extraction with DSC chunking + P3 prompting and persist PKG-ready output.

Usage:
    python scripts/run_pkg_extraction.py \
        --input-path datasets/archive/test_data/text/3m_marine_oem_sop.txt \
        --doc-id 3M_OEM_SOP
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.unified_config import CHUNKING_METHOD_CHOICES, reload_config
from src.graph.adapter import flat_to_tierb
from src.pipelines.baseline import extraction_payload
from src.processors.streamlined_processor import StreamlinedDocumentProcessor


def _resolve_path(value: str) -> Path:
    """Return an absolute path anchored at the repo root when needed."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _apply_env_overrides(
    chunking_method: str,
    prompting_strategy: str,
    gpu_backend: str | None,
    llm_backend: str | None,
    hf_model_id: str | None,
    hf_quantization: str | None,
) -> None:
    """Set environment overrides prior to instantiating the config singleton."""
    os.environ["CHUNKING_METHOD"] = chunking_method.lower()
    os.environ["PROMPTING_STRATEGY"] = prompting_strategy.upper()
    if gpu_backend:
        os.environ["GPU_BACKEND"] = gpu_backend
    # Default to GPU acceleration when explicitly targeting Metal/CUDA
    if gpu_backend and gpu_backend.lower() in {"metal", "cuda"}:
        os.environ["ENABLE_GPU"] = "1"
    if llm_backend:
        os.environ["LLM_BACKEND"] = llm_backend
        if llm_backend.lower() == "transformers":
            # Enable PyTorch automatic CPU fallback for unsupported MPS ops.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if hf_model_id:
        os.environ["LLM_MODEL_ID"] = hf_model_id
    if hf_quantization:
        os.environ["LLM_QUANTIZATION"] = hf_quantization


async def run_extraction(args: argparse.Namespace) -> None:
    """Execute the extraction pipeline and persist both flat + PKG views."""
    _apply_env_overrides(
        args.chunking_method,
        args.prompting_strategy,
        args.gpu_backend,
        args.llm_backend,
        args.hf_model_id,
        args.hf_quantization,
    )
    config = reload_config()
    processor = StreamlinedDocumentProcessor(config=config)

    doc_path = _resolve_path(args.input_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = await processor.process_document(file_path=str(doc_path), document_id=args.doc_id)
    payload = extraction_payload(args.doc_id, result)

    flat_path = output_dir / f"{args.doc_id}_extracted.json"
    flat_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pkg_payload = flat_to_tierb(payload)
    pkg_path = output_dir / f"{args.doc_id}_pkg.json"
    pkg_path.write_text(json.dumps(pkg_payload, indent=2), encoding="utf-8")

    print(f"Extraction complete via DSC + {args.prompting_strategy.upper()} for {args.doc_id}.")
    print(f"Flat structured output : {flat_path}")
    print(f"PKG-ready graph output: {pkg_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSC chunked, P3 prompted extraction and emit PKG-ready artifacts."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/archive/test_data/text/3m_marine_oem_sop.txt",
        help="Document to process.",
    )
    parser.add_argument(
        "--doc-id",
        default="3M_OEM_SOP",
        help="Document identifier included inside the saved JSON payloads.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/pkg_runs/3m_marine_oem_sop",
        help="Directory for saving extraction artifacts.",
    )
    parser.add_argument(
        "--chunking-method",
        default="dsc",
        choices=sorted(CHUNKING_METHOD_CHOICES),
        help="Chunking method override (default: dsc).",
    )
    parser.add_argument(
        "--prompting-strategy",
        default="P3",
        choices=["P0", "P1", "P2", "P3"],
        help="Prompting strategy override (default: P3 two-stage schema).",
    )
    parser.add_argument(
        "--gpu-backend",
        default="metal",
        help="GPU backend hint ('metal' for Apple M-series, 'cpu' to disable).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["llama_cpp", "transformers"],
        default=None,
        help="Override the LLM backend (set to 'transformers' for HuggingFace/MPS runs).",
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Optional HuggingFace model identifier when using the transformers backend.",
    )
    parser.add_argument(
        "--hf-quantization",
        choices=["4bit", "8bit", "none"],
        default=None,
        help="Optional quantization hint for HuggingFace models (use 'none' for pure MPS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_extraction(args))


if __name__ == "__main__":
    main()
