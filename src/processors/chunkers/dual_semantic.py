"""Dual semantic (hierarchical) chunker implementation."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .base import Chunk
from .breakpoint import BreakpointSemanticChunker


LOGGER = logging.getLogger(__name__)


class DualSemanticChunker(BreakpointSemanticChunker):
    """Hierarchical semantic chunker using parent sections + breakpoint refinement."""

    def __init__(self, cfg, *, parent_only: bool = False):
        super().__init__(cfg)
        self.parent_only = parent_only

    def chunk(self, text: str) -> List[Chunk]:
        sentences, spans = self._sentences(text)
        if not sentences:
            return []

        embeddings = self._embeddings(sentences)
        if embeddings.size == 0 or embeddings.shape[0] != len(sentences):
            return self._fallback_chunk(text, spans)

        if len(sentences) == 1:
            return self._fallback_chunk(text, spans)

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
            boundaries = self._compute_boundaries(end - start, prefix_sims, offset=start)
            if not boundaries:
                boundaries = [(start, end)]
            parent_cohesion = self._mean_similarity(prefix_sims, start, end)
            for child_start, child_end in boundaries:
                chunk = self._build_chunk(sentences, spans, child_start, child_end, prefix_sims)
                if chunk is None:
                    continue
                if isinstance(chunk.meta, dict):
                    chunk.meta["strategy"] = "dual_semantic"
                    chunk.meta["sentences"] = (child_start, child_end)
                    chunk.meta["parent"] = (start, end)
                    chunk.meta["parent_cohesion"] = parent_cohesion
                chunks.append(chunk)

        refined = self._enforce_char_cap(chunks, sentences, spans, prefix_sims)
        self._log_chunk_summary(refined, len(text))
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
            chunk = self._build_chunk(sentences, spans, start, end, prefix_sims)
            if chunk is None:
                continue
            if isinstance(chunk.meta, dict):
                chunk.meta["strategy"] = "parent_only"
                chunk.meta["sentences"] = (start, end)
                chunk.meta["parent"] = (start, end)
                chunk.meta["parent_cohesion"] = self._mean_similarity(prefix_sims, start, end)
            parent_chunks.append(chunk)

        refined = self._enforce_char_cap(parent_chunks, sentences, spans, prefix_sims)
        self._log_chunk_summary(refined, text_len)
        return refined

    def _parent_boundaries(self, distances, sentences, spans, text):
        lam = float(getattr(self.cfg, "dsc_lambda", 0.1))
        beta = float(getattr(self.cfg, "dsc_beta", 0.2))
        use_headings = bool(getattr(self.cfg, "dsc_use_headings", True))

        n = len(sentences)
        if n == 0:
            return []
        if n == 1:
            return [(0, 1)]

        # Recover sims from distances; build prefix array for O(1) cohesion queries.
        sims = 1.0 - np.asarray(distances, dtype=float)
        prefix = np.zeros(n)
        prefix[1:] = np.cumsum(sims)  # prefix[k] = sum(sims[0:k])

        # Detect heading positions (sentence j starts a heading → bonus β at split j).
        is_heading = np.zeros(n, dtype=bool)
        if use_headings and beta > 0.0:
            import re
            # Require a dot after Roman numerals to avoid matching ordinary words
            # starting with V/I/X/L/C (e.g. "Verify", "Inspect", "Check").
            heading_re = re.compile(r"^\s*((\d+(\.\d+)*\.?)|([IVXLC]+\.)|([A-Z][\w\s]{0,60}:))")
            for idx, sent in enumerate(sentences):
                is_heading[idx] = bool(heading_re.match(sent))

        # Global DP: dp[j] = best objective for sentences [0, j).
        # DP[j] = max_{i<j} { DP[i] + H(b_{i:j}) − λ + β·𝟙[j is heading] }
        # H(b_{i:j}) = mean within-block cosine similarity = (prefix[j-1] - prefix[i]) / (j-1-i)
        dp = np.full(n + 1, -np.inf)
        back = np.full(n + 1, -1, dtype=int)
        dp[0] = 0.0

        for j in range(1, n + 1):
            heading_bonus = beta if (j < n and is_heading[j]) else 0.0
            for i in range(j):
                cohesion = self._mean_similarity(prefix, i, j)
                score = dp[i] + cohesion - lam + heading_bonus
                if score > dp[j]:
                    dp[j] = score
                    back[j] = i

        # Backtrack to recover partition.
        boundaries: List[Tuple[int, int]] = []
        idx = n
        while idx > 0:
            i = back[idx]
            assert i != -1, f"DP invariant violated: back[{idx}] unset (dp[0]=0 should always propagate)"
            boundaries.append((i, idx))
            idx = i
        boundaries.reverse()
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
