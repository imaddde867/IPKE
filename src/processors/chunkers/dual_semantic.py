"""Dual semantic (hierarchical) chunker implementation."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .base import BaseChunker, Chunk
from .breakpoint import BreakpointSemanticChunker


LOGGER = logging.getLogger(__name__)


class DualSemanticChunker(BaseChunker):
    """Hierarchical semantic chunker using parent sections + breakpoint refinement."""

    def __init__(self, cfg, *, parent_only: bool = False):
        self.cfg = cfg
        self.base = BreakpointSemanticChunker(cfg)
        self.parent_only = parent_only

    def chunk(self, text: str) -> List[Chunk]:
        sentences, spans = self.base._sentences(text)
        if not sentences:
            return []

        embeddings = self.base._embeddings(sentences)
        if embeddings.size == 0 or embeddings.shape[0] != len(sentences):
            return self.base._fallback_chunk(text, spans)

        if len(sentences) == 1:
            return self.base._fallback_chunk(text, spans)

        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        prefix_sims = np.zeros(len(sentences))
        if sims.size:
            prefix_sims[1:] = np.cumsum(sims)
        distances = 1.0 - sims
        parents = self._parent_boundaries(distances, sentences, spans, text)

        if not parents:
            parents = [(0, len(sentences))]

        self._log_parent_summary(parents, spans, len(text))

        if self.parent_only:
            return self._emit_parent_chunks(parents, sentences, spans, prefix_sims, len(text))

        chunks: List[Chunk] = []
        for start, end in parents:
            if start >= end:
                continue
            boundaries = self.base._compute_boundaries(end - start, prefix_sims, offset=start)
            if not boundaries:
                boundaries = [(start, end)]
            parent_cohesion = self.base._mean_similarity(prefix_sims, start, end)
            for child_start, child_end in boundaries:
                chunk = self.base._build_chunk(sentences, spans, child_start, child_end, prefix_sims)
                if chunk is None:
                    continue
                if isinstance(chunk.meta, dict):
                    chunk.meta["strategy"] = "dual_semantic"
                    chunk.meta["sentences"] = (child_start, child_end)
                    chunk.meta["parent"] = (start, end)
                    chunk.meta["parent_cohesion"] = parent_cohesion
                chunks.append(chunk)

        refined = self.base._enforce_char_cap(chunks, sentences, spans, prefix_sims)
        self.base._log_chunk_summary(refined, len(text))
        return refined

    def _emit_parent_chunks(
        self,
        parents: List[Tuple[int, int]],
        sentences: List[str],
        spans: List[Tuple[int, int]],
        prefix_sims: np.ndarray,
        text_len: int,
    ) -> List[Chunk]:
        parent_chunks: List[Chunk] = []
        for start, end in parents:
            chunk = self.base._build_chunk(sentences, spans, start, end, prefix_sims)
            if chunk is None:
                continue
            if isinstance(chunk.meta, dict):
                chunk.meta["strategy"] = "parent_only"
                chunk.meta["sentences"] = (start, end)
                chunk.meta["parent"] = (start, end)
                chunk.meta["parent_cohesion"] = self.base._mean_similarity(prefix_sims, start, end)
            parent_chunks.append(chunk)

        refined = self.base._enforce_char_cap(parent_chunks, sentences, spans, prefix_sims)
        self.base._log_chunk_summary(refined, text_len)
        return refined

    def _parent_boundaries(self, distances, sentences, spans, text):
        min_len = max(1, getattr(self.cfg, "dsc_parent_min_sentences", 10))
        max_len = max(min_len, getattr(self.cfg, "dsc_parent_max_sentences", 120))
        window = max(1, getattr(self.cfg, "dsc_delta_window", 25))
        threshold_k = max(0.0, getattr(self.cfg, "dsc_threshold_k", 1.0))
        use_headings = bool(getattr(self.cfg, "dsc_use_headings", True))

        n = len(sentences)
        boundaries: List[Tuple[int, int]] = []
        start = 0
        t = window

        def local_stats(center: int):
            left = max(0, center - window)
            right = min(len(distances), center + window)
            segment = distances[left:right]
            if segment.size == 0:
                return 0.0, 0.0
            mean = float(np.mean(segment))
            std = float(np.std(segment))
            return mean, std

        heading_pattern = None
        if use_headings:
            import re
            heading_pattern = re.compile(r"^\s*((\d+(\.\d+)*)|([IVXLC]+\.?)|([A-Z][\w\s]{0,60}:))")

        while start < n:
            end = min(n, start + max_len)
            boundary = end

            for idx in range(start + min_len, end):
                if idx - start < min_len:
                    continue
                mean, std = local_stats(idx - 1)
                threshold = mean + threshold_k * std
                distance_value = distances[idx - 1] if idx - 1 < len(distances) else 0.0
                heading_break = False
                if heading_pattern and idx < n:
                    heading_break = bool(heading_pattern.match(sentences[idx]))
                if distance_value > threshold or heading_break:
                    boundary = idx
                    break

            boundaries.append((start, boundary))
            start = boundary

        if boundaries and boundaries[-1][1] != n:
            boundaries[-1] = (boundaries[-1][0], n)

        return boundaries

    def _log_parent_summary(self, parents: List[Tuple[int, int]], spans: List[Tuple[int, int]], doc_chars: int) -> None:
        if not parents or not bool(getattr(self.cfg, "debug_chunking", False)):
            return
        parent_lengths = [end - start for start, end in parents]
        parent_chars = [spans[end - 1][1] - spans[start][0] for start, end in parents]
        avg_sent = float(np.mean(parent_lengths)) if parent_lengths else 0.0
        median_sent = float(np.median(parent_lengths)) if parent_lengths else 0.0
        median_chars = float(np.median(parent_chars)) if parent_chars else 0.0
        LOGGER.info(
            "Dual semantic parents: %d blocks | mean_sent=%.1f | median_sent=%.1f | median_chars=%.0f | doc_chars=%d",
            len(parents),
            avg_sent,
            median_sent,
            median_chars,
            doc_chars,
        )


__all__ = ["DualSemanticChunker"]
