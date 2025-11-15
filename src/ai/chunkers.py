"""
Chunking utilities for Explainium/IPKE.

Currently only provides a fixed-length chunker plus scaffolding for semantic
approaches. Heavy dependencies (spaCy, sentence-transformers) are loaded lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from abc import ABC, abstractmethod
import numpy as np

# Ensure tokenizers stay single-threaded before any optional imports occur.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


class BreakpointSemanticChunker(BaseChunker):
    """Semantic chunker scaffold using spaCy + sentence-transformer embeddings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._nlp = None
        self._model = None

    def _load_spacy(self):
        if self._nlp is not None:
            return self._nlp
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        self._nlp = nlp
        return self._nlp

    def _load_embedder(self):
        if self._model is not None:
            return self._model

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer

        device = "cuda" if getattr(self.cfg, "gpu_backend", "cpu") == "cuda" else "cpu"
        model_path = getattr(self.cfg, "embedding_model_path", "models/embeddings/all-mpnet-base-v2")
        self._model = SentenceTransformer(model_path, device=device)
        return self._model

    @property
    def nlp(self):
        return self._load_spacy()

    @property
    def embedder(self):
        return self._load_embedder()

    def _sentences(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if not text:
            return [], []
        doc = self.nlp(text)
        sentences = []
        spans: List[Tuple[int, int]] = []
        for sent in doc.sents:
            sentences.append(sent.text.strip())
            spans.append((sent.start_char, sent.end_char))
        return sentences, spans

    def chunk(self, text: str) -> List[Chunk]:
        sentences, spans = self._sentences(text)
        if not sentences:
            return []

        embeddings = self._embeddings(sentences)
        if embeddings.size == 0 or embeddings.shape[0] != len(sentences):
            return self._fallback_chunk(text, spans)

        n_sentences = len(sentences)
        if n_sentences == 1:
            return self._fallback_chunk(text, spans)

        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        prefix_sims = np.zeros(n_sentences)
        if sims.size:
            prefix_sims[1:] = np.cumsum(sims)

        boundaries = self._compute_boundaries(n_sentences, prefix_sims)
        if not boundaries:
            boundaries = [(0, n_sentences)]

        chunks = self._splice_chunks(sentences, spans, boundaries, prefix_sims)
        return self._enforce_char_cap(chunks, sentences, spans, prefix_sims)
    def _embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Embed sentences and return L2-normalized embeddings.
        Shape: (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])

        model = self._load_embedder()
        embeddings = model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def _fallback_chunk(self, text: str, spans: List[Tuple[int, int]]) -> List[Chunk]:
        clean = text.strip()
        if not clean:
            return []
        start_char = spans[0][0] if spans else 0
        end_char = spans[-1][1] if spans else len(text)
        return [
            Chunk(
                text=clean,
                start_char=start_char,
                end_char=end_char,
                sentence_span=(0, max(1, len(spans))),
                meta={"strategy": "fallback"}
            )
        ]

    def _compute_boundaries(self, n_sentences: int, prefix_sims: np.ndarray) -> List[Tuple[int, int]]:
        min_len = max(1, getattr(self.cfg, "sem_min_sentences_per_chunk", 2))
        max_len = max(min_len, getattr(self.cfg, "sem_max_sentences_per_chunk", 40))
        window = max(min_len, getattr(self.cfg, "sem_window_w", max_len))
        lam = getattr(self.cfg, "sem_lambda", 0.15)

        dp = [-float("inf")] * (n_sentences + 1)
        back = [-1] * (n_sentences + 1)
        dp[0] = 0.0

        for end in range(1, n_sentences + 1):
            best_score = -float("inf")
            best_start = -1
            max_window = min(end, min(max_len, window))
            min_window = min_len
            for length in range(min_window, max_window + 1):
                start = end - length
                if start < 0 or dp[start] == -float("inf"):
                    continue
                cohesion = self._mean_similarity(prefix_sims, start, end)
                score = dp[start] + cohesion - lam
                if score > best_score:
                    best_score = score
                    best_start = start
            if best_start != -1:
                dp[end] = best_score
                back[end] = best_start

        if dp[n_sentences] == -float("inf"):
            return []

        boundaries: List[Tuple[int, int]] = []
        idx = n_sentences
        while idx > 0 and back[idx] != -1:
            start = back[idx]
            boundaries.append((start, idx))
            idx = start

        if idx != 0:
            return []

        return list(reversed(boundaries))

    def _mean_similarity(self, prefix_sims: np.ndarray, start: int, end: int) -> float:
        steps = end - start - 1
        if steps <= 0:
            return 0.0
        total = prefix_sims[end - 1] - prefix_sims[start]
        return total / steps if steps > 0 else 0.0

    def _build_chunk(
        self,
        sentences: List[str],
        spans: List[Tuple[int, int]],
        start: int,
        end: int,
        prefix_sims: np.ndarray
    ) -> Optional[Chunk]:
        if start >= end or start >= len(sentences):
            return None
        end = min(end, len(sentences))
        chunk_text = " ".join(sentences[start:end]).strip()
        if not chunk_text:
            return None
        start_char = spans[start][0]
        end_char = spans[end - 1][1]
        cohesion = self._mean_similarity(prefix_sims, start, end)
        return Chunk(
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            sentence_span=(start, end),
            meta={
                "strategy": "breakpoint_semantic",
                "sentences": (start, end),
                "cohesion": cohesion
            }
        )

    def _splice_chunks(
        self,
        sentences: List[str],
        spans: List[Tuple[int, int]],
        boundaries: List[Tuple[int, int]],
        prefix_sims: np.ndarray
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        for start, end in boundaries:
            chunk = self._build_chunk(sentences, spans, start, end, prefix_sims)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _enforce_char_cap(
        self,
        chunks: List[Chunk],
        sentences: List[str],
        spans: List[Tuple[int, int]],
        prefix_sims: np.ndarray
    ) -> List[Chunk]:
        max_chars = max(1, getattr(self.cfg, "chunk_max_chars", 2000))
        refined: List[Chunk] = []
        for chunk in chunks:
            ranges = []
            if isinstance(chunk.meta, dict) and "sentences" in chunk.meta:
                ranges.append(chunk.meta["sentences"])
            else:
                refined.append(chunk)
                continue

            while ranges:
                start, end = ranges.pop(0)
                split_chunk = self._build_chunk(sentences, spans, start, end, prefix_sims)
                if split_chunk is None:
                    continue
                if len(split_chunk.text) <= max_chars or (end - start) <= 1:
                    refined.append(split_chunk)
                else:
                    mid = start + (end - start) // 2
                    if mid == start or mid == end:
                        refined.append(split_chunk)
                    else:
                        ranges.insert(0, (mid, end))
                        ranges.insert(0, (start, mid))
        return refined


def get_chunker(cfg) -> BaseChunker:
    """Factory for chunker instances."""

    method = getattr(cfg, "chunking_method", "fixed")
    if method == "fixed":
        max_chars = getattr(cfg, "chunk_max_chars", 2000)
        return FixedChunker(max_chars)
    if method == "breakpoint_semantic":
        return BreakpointSemanticChunker(cfg)
    raise NotImplementedError(f"Method {method} not implemented yet")
