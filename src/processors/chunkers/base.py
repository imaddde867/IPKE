"""Common chunk data structures and interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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


__all__ = ["Chunk", "BaseChunker"]
