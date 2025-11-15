"""
Chunking utilities for Explainium/IPKE.

Currently only provides a fixed-length chunker plus scaffolding for semantic
approaches. Heavy dependencies (spaCy, sentence-transformers) are loaded lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
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
        raise NotImplementedError("Semantic breakpoint chunking not implemented yet")
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


def get_chunker(cfg) -> BaseChunker:
    """Factory for chunker instances."""

    method = getattr(cfg, "chunking_method", "fixed")
    if method == "fixed":
        max_chars = getattr(cfg, "chunk_max_chars", 2000)
        return FixedChunker(max_chars)
    if method == "breakpoint_semantic":
        return BreakpointSemanticChunker(cfg)
    raise NotImplementedError(f"Method {method} not implemented yet")
