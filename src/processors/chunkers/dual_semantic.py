"""Dual semantic (hierarchical) chunker implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaseChunker, Chunk
from .breakpoint import BreakpointSemanticChunker


class DualSemanticChunker(BaseChunker):
    """Hierarchical semantic chunker using parent sections + breakpoint refinement."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.base = BreakpointSemanticChunker(cfg)

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

        chunks: List[Chunk] = []
        for start, end in parents:
            if start >= end:
                continue
            parent_text = text[spans[start][0]:spans[end - 1][1]]
            sub_chunks = self.base.chunk(parent_text)
            offset = spans[start][0]
            for chunk in sub_chunks:
                chunk.start_char += offset
                chunk.end_char += offset
                global_span = (
                    start + chunk.sentence_span[0],
                    start + chunk.sentence_span[1]
                )
                chunk.sentence_span = global_span
                if isinstance(chunk.meta, dict):
                    chunk.meta["sentences"] = global_span
                    chunk.meta["parent"] = (start, end)
                    chunk.meta["parent_cohesion"] = self.base._mean_similarity(prefix_sims, start, end)
                chunks.append(chunk)

        return self.base._enforce_char_cap(chunks, sentences, spans, prefix_sims)

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


__all__ = ["DualSemanticChunker"]
