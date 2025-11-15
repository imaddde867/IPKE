import re
from typing import List

import numpy as np

from src.ai.chunkers import (
    BreakpointSemanticChunker,
    DualSemanticChunker,
    FixedChunker,
)
from src.ai.knowledge_engine import BaseExtractionStrategy, ChunkExtraction
from src.core.unified_config import UnifiedConfig


class DummySpan:
    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start_char = start
        self.end_char = end


class DummyDoc:
    def __init__(self, sents: List[DummySpan]):
        self.sents = sents


class DummyNLP:
    pattern = re.compile(r"[^.!?\n]+[.!?\n]*")

    def __call__(self, text: str) -> DummyDoc:
        spans: List[DummySpan] = []
        for match in self.pattern.finditer(text):
            span_text = match.group()
            if not span_text.strip():
                continue
            spans.append(DummySpan(span_text, match.start(), match.end()))
        return DummyDoc(spans)


class DummyEmbedder:
    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for sent in sentences:
            low = sent.lower()
            vec = np.array(
                [
                    1.0 if "alpha" in low else 0.0,
                    1.0 if "beta" in low else 0.0,
                    1.0 if "gamma" in low else 0.0,
                    len(sent) / 100.0,
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(vec) == 0:
                vec[-1] = 1.0
            vectors.append(vec)
        arr = np.vstack(vectors) if vectors else np.zeros((0, 4), dtype=np.float32)
        if normalize_embeddings and len(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
            arr = arr / norms
        return arr


class BrokenEmbedder:
    def encode(self, *_, **__):
        raise RuntimeError("embedding boom")


class DummyStrategy(BaseExtractionStrategy):
    async def _initialize(self):
        return

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        return ChunkExtraction()

    def get_strategy_name(self) -> str:
        return "dummy"


def make_config() -> UnifiedConfig:
    cfg = UnifiedConfig()
    cfg.chunking_method = "fixed"
    cfg.chunk_max_chars = 200
    cfg.sem_min_sentences_per_chunk = 2
    cfg.sem_max_sentences_per_chunk = 8
    cfg.sem_lambda = 0.15
    cfg.sem_window_w = 30
    cfg.dsc_parent_min_sentences = 4
    cfg.dsc_parent_max_sentences = 20
    cfg.dsc_delta_window = 5
    cfg.embedding_model_path = "dummy"
    cfg.debug_chunking = True
    return cfg


def _topic_document(section_counts):
    blocks = []
    for idx, count in enumerate(section_counts):
        label = ["Alpha", "Beta", "Gamma", "Delta"][idx % 4]
        for sent in range(count):
            blocks.append(f"{label} topic sentence {idx}-{sent}.")
    return " ".join(blocks)


def test_fixed_chunker_respects_char_cap():
    chunker = FixedChunker(max_chars=24)
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    chunks = chunker.chunk(text)
    assert chunks, "Expected at least one chunk"
    rebuilt = " ".join(chunk.text for chunk in chunks)
    for chunk in chunks:
        assert len(chunk.text) <= 24
    assert "alpha" in rebuilt.lower()


def test_breakpoint_semantic_aligns_on_toy_shifts():
    cfg = make_config()
    text = _topic_document([5, 5, 5])
    chunker = BreakpointSemanticChunker(cfg, nlp=DummyNLP(), embedder=DummyEmbedder())
    chunks = chunker.chunk(text)
    assert 3 <= len(chunks) <= 6
    assert all(chunks[i].sentence_span[0] < chunks[i].sentence_span[1] for i in range(3))
    assert any("Alpha" in chunk.text for chunk in chunks)
    assert any("Beta" in chunk.text for chunk in chunks)
    assert any("Gamma" in chunk.text for chunk in chunks)


def test_dsc_produces_coarser_then_finer():
    cfg = make_config()
    cfg.dsc_parent_min_sentences = 6
    cfg.dsc_parent_max_sentences = 18
    text = _topic_document([8, 8, 8])
    base = BreakpointSemanticChunker(cfg, nlp=DummyNLP(), embedder=DummyEmbedder())
    dual = DualSemanticChunker(cfg, nlp=DummyNLP(), embedder=DummyEmbedder())
    base_chunks = base.chunk(text)
    dual_chunks = dual.chunk(text)
    assert len({chunk.meta.get("parent_block") for chunk in dual_chunks}) == 3
    assert len(dual_chunks) <= len(base_chunks)
    assert any(chunk.meta.get("parent_sentence_span") for chunk in dual_chunks)


def test_semantic_fallbacks_when_short_or_unembeddable():
    cfg = make_config()
    short_text = "Alpha only sentence."
    chunker = BreakpointSemanticChunker(cfg, nlp=DummyNLP(), embedder=DummyEmbedder())
    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1

    bad_chunker = BreakpointSemanticChunker(cfg, nlp=DummyNLP(), embedder=BrokenEmbedder())
    strategy = DummyStrategy(cfg)
    strategy.chunker = bad_chunker
    fallback_chunks = strategy._chunk_content("Alpha Beta Gamma " * 5)
    assert fallback_chunks
    assert fallback_chunks[0].meta["chunking_method"] == "fixed"
