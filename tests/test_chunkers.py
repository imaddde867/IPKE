import numpy as np
import pytest

from src.ai.chunkers import BreakpointSemanticChunker, DualSemanticChunker


class DummyCfg:
    gpu_backend = "cpu"
    embedding_model_path = "sentence-transformers/all-MiniLM-L6-v2"
    chunking_method = "breakpoint_semantic"
    chunk_max_chars = 1000
    sem_min_sentences_per_chunk = 1
    sem_max_sentences_per_chunk = 3
    sem_window_w = 4
    sem_lambda = 0.05
    dsc_parent_min_sentences = 2
    dsc_parent_max_sentences = 6
    dsc_delta_window = 2
    dsc_threshold_k = 0.5
    dsc_use_headings = True


@pytest.mark.integration
def test_breakpoint_semantic_embeddings_normalized(monkeypatch):
    monkeypatch.setenv("EXPLAINIUM_ENV", "testing")
    chunker = BreakpointSemanticChunker(DummyCfg())

    class StubEmbedder:
        def encode(self, sentences, show_progress_bar, convert_to_numpy, normalize_embeddings):
            assert show_progress_bar is False
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            data = np.arange(len(sentences) * 4, dtype=float).reshape(len(sentences), 4) + 1.0
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            return data / norms

    monkeypatch.setattr(chunker, "_load_embedder", lambda: StubEmbedder())

    sentences = [
        "First sentence for embedding.",
        "Second sentence, still simple.",
        "Third one closes the test."
    ]

    embeddings = chunker._embeddings(sentences)
    assert embeddings.shape == (3, 4)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, np.ones(3))

    empty = chunker._embeddings([])
    assert isinstance(empty, np.ndarray)
    assert empty.size == 0


@pytest.mark.integration
def test_breakpoint_semantic_chunk_boundaries(monkeypatch):
    monkeypatch.setenv("EXPLAINIUM_ENV", "testing")
    chunker = BreakpointSemanticChunker(DummyCfg())

    sentences = [
        "Prep station with PPE ready.",
        "Verify ventilation is on.",
        "Switch robot to maintenance mode.",
        "Drain hydraulic pressure carefully."
    ]

    text = ""
    spans = []
    cursor = 0
    for idx, sent in enumerate(sentences):
        text += sent
        spans.append((cursor, cursor + len(sent)))
        cursor += len(sent)
        if idx < len(sentences) - 1:
            text += " "
            cursor += 1

    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [-1.0, 0.0],
        [-0.9, -0.1],
    ], dtype=float)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    monkeypatch.setattr(chunker, "_sentences", lambda _: (sentences, spans))
    monkeypatch.setattr(chunker, "_embeddings", lambda sent: embeddings[:len(sent)])

    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert chunks[0].meta["sentences"] == (0, 2)
    assert chunks[1].meta["sentences"] == (2, 4)
    assert "Prep station" in chunks[0].text
    assert "Drain hydraulic" in chunks[1].text


@pytest.mark.integration
def test_dual_semantic_parent_boundaries(monkeypatch):
    monkeypatch.setenv("EXPLAINIUM_ENV", "testing")

    class DSCfg(DummyCfg):
        chunking_method = "dsc"

    chunker = DualSemanticChunker(DSCfg())

    sentences = [
        "Section 1. PPE setup instructions.",
        "Ensure gloves and goggles are ready.",
        "Heading 2. Ventilation guidelines.",
        "Start exhaust fans and monitor sensors.",
        "Heading 3. Calibration tasks.",
        "Run diagnostic sequence for axis motors.",
        "Finalize with checklist sign-off."
    ]

    text = ""
    spans = []
    cursor = 0
    for idx, sent in enumerate(sentences):
        text += sent
        spans.append((cursor, cursor + len(sent)))
        cursor += len(sent)
        if idx < len(sentences) - 1:
            text += "\n"
            cursor += 1

    topic_vectors = np.array([
        [1.0, 0.0],
        [0.95, 0.05],
        [-0.8, -0.5],
        [-0.82, -0.48],
        [0.2, 0.98],
        [0.23, 0.97],
        [0.3, 0.95]
    ], dtype=float)
    embeddings = topic_vectors / np.linalg.norm(topic_vectors, axis=1, keepdims=True)

    monkeypatch.setattr(chunker.base, "_sentences", lambda _: (sentences, spans))
    monkeypatch.setattr(chunker.base, "_embeddings", lambda sents: embeddings[:len(sents)])

    chunks = chunker.chunk(text)
    spans = [chunk.sentence_span for chunk in chunks]
    assert spans.count((0, 2)) >= 1
    assert spans.count((2, 4)) >= 1
    assert spans.count((4, 7)) >= 1
