"""Offline chunking experiment runner for thesis analysis."""

import asyncio
import json
import os
from pathlib import Path
from typing import Iterable, List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.ai.chunkers import TextChunk, get_chunker  # noqa: E402
from src.core.unified_config import get_config  # noqa: E402
from src.processors.streamlined_processor import StreamlinedDocumentProcessor  # noqa: E402
from src.exceptions import ProcessingError  # noqa: E402


DOCUMENTS: List[Path] = [
    Path("datasets/archive/test_data/text/3m_marine_oem_sop.txt"),
    Path("datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"),
    Path("datasets/archive/test_data/text/op_firesafety_guideline.txt"),
]
CHUNKING_METHODS: Iterable[str] = ("fixed", "breakpoint_semantic", "dsc")
RESULTS_DIR = Path("results")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_text(processor: StreamlinedDocumentProcessor, doc_path: Path) -> str:
    async def _extract() -> str:
        return await processor.extract_text_content(str(doc_path))

    return asyncio.run(_extract())


def save_result(doc_path: Path, method: str, chunks: List[TextChunk]) -> None:
    avg_chunk_size = (
        sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0
    )
    output_file = RESULTS_DIR / f"{doc_path.stem}_{method}.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "document": str(doc_path),
                "method": method,
                "num_chunks": len(chunks),
                "avg_chunk_size": avg_chunk_size,
                "chunks": [
                    {"text": chunk.text, "sentences": chunk.sentence_span} for chunk in chunks
                ],
            },
            handle,
            indent=2,
        )


def run_experiments() -> None:
    ensure_results_dir()
    cfg = get_config()
    processor = StreamlinedDocumentProcessor(config=cfg)

    for doc_path in DOCUMENTS:
        if not doc_path.exists():
            print(f"⚠️ Skipping {doc_path}: file not found")
            continue

        print(f"→ Extracting text from {doc_path}")
        try:
            text = extract_text(processor, doc_path)
        except ProcessingError as exc:
            print(f"✗ Failed to extract {doc_path}: {exc}")
            continue
        if not text.strip():
            print(f"⚠️ No text extracted from {doc_path}, skipping chunking")
            continue

        for method in CHUNKING_METHODS:
            cfg.chunking_method = method
            chunker = get_chunker(cfg)
            chunks = chunker.chunk(text, source_filename=str(doc_path))
            save_result(doc_path, method, chunks)
            print(f"✓ {doc_path} + {method}: {len(chunks)} chunks")

    processor.executor.shutdown(wait=True)


if __name__ == "__main__":
    run_experiments()
