"""Fixed-size chunker implementation."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .base import BaseChunker, Chunk


LOGGER = logging.getLogger(__name__)


class FixedChunker(BaseChunker):
    """Splits text into roughly equal sized pieces using whitespace boundaries."""

    def __init__(
        self,
        max_chars: int = 2000,
        overlap_chars: int = 0,
        dedup_overlap_ratio: float = 0.75,
        log_stats: bool = False,
    ):
        self.max_chars = max(200, max_chars)
        self.overlap = max(0, min(overlap_chars, self.max_chars - 1))
        self.step = self.max_chars - self.overlap
        self.dedup_overlap_ratio = max(0.0, min(dedup_overlap_ratio, 0.95)) if self.overlap else 1.0
        self.log_stats = log_stats

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []

        length = len(text)
        start = 0
        chunks: List[Chunk] = []
        total_overlap_chars = 0
        trimmed_events = 0
        skipped_duplicates = 0

        while start < length:
            window_end = min(start + self.max_chars, length)
            end = self._find_boundary(text, start, window_end)
            chunk_data = self._prepare_chunk(text, start, end)
            if chunk_data is None:
                start = self._next_start(start, text, length)
                continue
            chunk_text, start_char, end_char = chunk_data

            if chunks and self.overlap:
                overlap_chars = max(0, chunks[-1].end_char - start_char)
                total_overlap_chars += overlap_chars
                dedup_result = self._clip_duplicate_prefix(text, chunks[-1], start_char, end_char)
                if dedup_result is not None:
                    trimmed_text, trimmed_start, trimmed_end = dedup_result
                    if not trimmed_text:
                        skipped_duplicates += 1
                        start = self._next_start(start, text, length)
                        continue
                    chunk_text, start_char, end_char = trimmed_text, trimmed_start, trimmed_end
                    trimmed_events += 1

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_span=(0, 0),
                    meta={"strategy": "fixed"},
                )
            )
            start = self._next_start(start, text, length)

        if self.log_stats and chunks:
            covered_chars = sum(chunk.end_char - chunk.start_char for chunk in chunks)
            overlap_fraction = total_overlap_chars / max(1, covered_chars)
            LOGGER.info(
                "Fixed chunker: %d chunks | overlap=%d chars (%.2f frac) | trimmed=%d | skipped=%d",
                len(chunks),
                total_overlap_chars,
                overlap_fraction,
                trimmed_events,
                skipped_duplicates,
            )

        return chunks

    def _next_start(self, current_start: int, text: str, length: int) -> int:
        next_start = current_start + self.step
        if next_start >= length:
            return length
        while next_start < length and text[next_start].isspace():
            next_start += 1
        return next_start

    def _find_boundary(self, text: str, start: int, window_end: int) -> int:
        end = window_end
        if end < len(text):
            backtrack = text.rfind(" ", start, end)
            if backtrack == -1 or backtrack <= start:
                backtrack = text.rfind("\n", start, end)
            if backtrack != -1 and backtrack > start:
                end = backtrack
        return end

    def _prepare_chunk(self, text: str, start: int, end: int) -> Optional[Tuple[str, int, int]]:
        if start >= end:
            return None
        chunk_text, start_char, end_char = self._trim_span(text, start, end)
        if not chunk_text:
            return None
        return chunk_text, start_char, end_char

    def _clip_duplicate_prefix(
        self,
        text: str,
        prev_chunk: Chunk,
        start_char: int,
        end_char: int,
    ) -> Optional[Tuple[str, int, int]]:
        if self.dedup_overlap_ratio >= 1.0:
            return None
        overlap = max(0, prev_chunk.end_char - start_char)
        chunk_len = max(1, end_char - start_char)
        if overlap <= 0:
            return None
        overlap_ratio = overlap / chunk_len
        if overlap_ratio < self.dedup_overlap_ratio:
            return None
        clip_start = prev_chunk.end_char
        if clip_start >= end_char:
            return ("", end_char, end_char)
        trimmed_text, trimmed_start, trimmed_end = self._trim_span(text, clip_start, end_char)
        return trimmed_text, trimmed_start, trimmed_end

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Tuple[str, int, int]:
        slice_text = text[start:end]
        left = 0
        right = len(slice_text)
        while left < right and slice_text[left].isspace():
            left += 1
        while right > left and slice_text[right - 1].isspace():
            right -= 1
        trimmed = slice_text[left:right]
        new_start = start + left
        new_end = start + right
        return trimmed, new_start, new_end


__all__ = ["FixedChunker"]
