#!/usr/bin/env python3
"""
Ad-hoc CLI to inspect chunk boundaries for different methods.

Example:
    python scripts/inspect_chunking.py --text-file sample.txt --method dsc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.ai.chunkers import get_chunker
from src.core.unified_config import UnifiedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect chunking output for a document.")
    parser.add_argument("--text-file", required=True, help="Path to a UTF-8 text file.")
    parser.add_argument(
        "--method",
        choices=["fixed", "breakpoint_semantic", "dsc"],
        help="Override CHUNKING_METHOD.",
    )
    parser.add_argument("--chunk-max-chars", type=int, help="Override CHUNK_MAX_CHARS.")
    parser.add_argument("--sem-lambda", type=float, help="Override SEM_LAMBDA.")
    parser.add_argument("--sem-window", type=int, help="Override SEM_WINDOW_W.")
    parser.add_argument("--parent-min", type=int, help="Override DSC_PARENT_MIN_SENTENCES.")
    parser.add_argument("--parent-max", type=int, help="Override DSC_PARENT_MAX_SENTENCES.")
    parser.add_argument("--threshold-k", type=float, help="Override DSC_THRESHOLD_K.")
    parser.add_argument("--debug", action="store_true", help="Print full chunk metadata.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UnifiedConfig:
    cfg = UnifiedConfig.from_environment()
    if args.method:
        cfg.chunking_method = args.method
    if args.chunk_max_chars:
        cfg.chunk_max_chars = args.chunk_max_chars
    if args.sem_lambda is not None:
        cfg.sem_lambda = args.sem_lambda
    if args.sem_window is not None:
        cfg.sem_window_w = args.sem_window
    if args.parent_min is not None:
        cfg.dsc_parent_min_sentences = args.parent_min
    if args.parent_max is not None:
        cfg.dsc_parent_max_sentences = args.parent_max
    if args.threshold_k is not None:
        cfg.dsc_threshold_k = args.threshold_k
    if args.debug:
        cfg.debug_chunking = True
    return cfg


def summarize_chunks(chunks) -> Dict[str, Any]:
    if not chunks:
        return {"count": 0, "avg_chars": 0, "avg_sentences": 0, "avg_cohesion": None}
    counts = [chunk.meta.get("sentence_count") or 0 for chunk in chunks]
    cohesions = [chunk.meta.get("cohesion") for chunk in chunks if chunk.meta.get("cohesion") is not None]
    avg_sentences = sum(counts) / len(chunks)
    avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks)
    avg_cohesion = sum(cohesions) / len(cohesions) if cohesions else None
    return {
        "count": len(chunks),
        "avg_chars": round(avg_chars, 1),
        "avg_sentences": round(avg_sentences, 2),
        "avg_cohesion": round(avg_cohesion, 3) if avg_cohesion is not None else None,
    }


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    text = Path(args.text_file).read_text(encoding="utf-8")
    chunker = get_chunker(cfg)
    chunks = chunker.chunk(text)
    summary = summarize_chunks(chunks)

    print(f"Method: {cfg.chunking_method} | Chunk cap: {cfg.chunk_max_chars} chars")
    print(f"Chunks: {summary['count']}  Avg chars: {summary['avg_chars']}  "
          f"Avg sentences: {summary['avg_sentences']}  Avg cohesion: {summary['avg_cohesion']}")
    print("-" * 80)
    for idx, chunk in enumerate(chunks):
        snippet = chunk.text[:80].replace("\n", " ")
        meta = chunk.meta
        print(
            f"[{idx:02}] chars={chunk.start_char}-{chunk.end_char} "
            f"sentences={chunk.sentence_span} "
            f"len={len(chunk.text)} "
            f"sent_count={meta.get('sentence_count')} "
            f"cohesion={meta.get('cohesion')}"
        )
        print(f"      {snippet}...")
        if args.debug:
            print(f"      meta={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
