"""Fixed-size chunker implementation."""

from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk


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


__all__ = ["FixedChunker"]
