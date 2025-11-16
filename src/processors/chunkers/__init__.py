"""Chunker factory and public exports."""

from __future__ import annotations

from typing import Any

from .base import BaseChunker, Chunk
from .breakpoint import BreakpointSemanticChunker
from .dual_semantic import DualSemanticChunker
from .fixed import FixedChunker


def get_chunker(cfg: Any) -> BaseChunker:
    """Return the configured chunker implementation."""

    method = getattr(cfg, "chunking_method", "fixed")
    if method == "fixed":
        max_chars = getattr(cfg, "chunk_max_chars", 2000)
        overlap = getattr(cfg, "chunk_overlap_chars", 0)
        return FixedChunker(max_chars, overlap)
    if method == "breakpoint_semantic":
        return BreakpointSemanticChunker(cfg)
    if method in {"dual_semantic", "dsc"}:
        return DualSemanticChunker(cfg)
    raise NotImplementedError(f"Chunking method '{method}' is not supported.")


__all__ = [
    "BaseChunker",
    "BreakpointSemanticChunker",
    "Chunk",
    "DualSemanticChunker",
    "FixedChunker",
    "get_chunker",
]
