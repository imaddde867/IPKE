"""Chunker factory and public exports."""

from __future__ import annotations

from typing import Any

_CHUNKING_ALIASES = {
    "dsc": "dual_semantic",
}


def _canonical_method(method: str) -> str:
    return _CHUNKING_ALIASES.get(method, method)

from .base import BaseChunker, Chunk
from .breakpoint import BreakpointSemanticChunker
from .dual_semantic import DualSemanticChunker
from .fixed import FixedChunker


def get_chunker(cfg: Any) -> BaseChunker:
    """Return the configured chunker implementation."""

    method = _canonical_method(getattr(cfg, "chunking_method", "fixed"))
    if method == "fixed":
        max_chars = getattr(cfg, "chunk_max_chars", 2000)
        overlap = getattr(cfg, "chunk_overlap_chars", 0)
        dedup_ratio = getattr(cfg, "chunk_overlap_dedup_ratio", 0.8)
        log_stats = bool(getattr(cfg, "debug_chunking", False))
        return FixedChunker(max_chars, overlap, dedup_ratio, log_stats)
    if method == "breakpoint_semantic":
        return BreakpointSemanticChunker(cfg)
    if method == "dual_semantic":
        return DualSemanticChunker(cfg)
    if method == "parent_only":
        return DualSemanticChunker(cfg, parent_only=True)
    raise NotImplementedError(f"Chunking method '{method}' is not supported.")


__all__ = [
    "BaseChunker",
    "BreakpointSemanticChunker",
    "Chunk",
    "DualSemanticChunker",
    "FixedChunker",
    "get_chunker",
]
