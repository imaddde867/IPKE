"""
Chunking utilities for Explainium/IPKE.

Currently only provides a fixed-length chunker that splits raw text on whitespace
boundaries. Future chunkers (breakpoint semantic, DSC) will rely on embeddings
or linguistic signals, but those are intentionally omitted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    """Represents a span of text extracted from the source document."""

    text: str
    start_char: int
    end_char: int
    sentence_span: Tuple[int, int] = (0, 0)
    meta: Dict[str, object] = field(default_factory=dict)


class BaseChunker(ABC):
    """Common interface for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split a document into chunks."""


class FixedChunker(BaseChunker):
    """Splits text into roughly equal sized pieces using whitespace boundaries."""

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max(200, max_chars)

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chars, length)
            if end < length:
                backtrack = text.rfind(" ", start, end)
                if backtrack == -1 or backtrack <= start:
                    backtrack = text.rfind("\n", start, end)
                if backtrack != -1 and backtrack > start:
                    end = backtrack
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_char=start,
                        end_char=start + len(chunk_text),
                        sentence_span=(0, 0),
                        meta={"strategy": "fixed"},
                    )
                )
            start = end
            while start < length and text[start].isspace():
                start += 1

        return chunks


def get_chunker(cfg) -> BaseChunker:
    """Factory for chunker instances."""

    method = getattr(cfg, "chunking_method", "fixed")
    if method == "fixed":
        max_chars = getattr(cfg, "chunk_max_chars", 2000)
        return FixedChunker(max_chars)
    raise NotImplementedError(f"Method {method} not implemented yet")
