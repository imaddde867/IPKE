import numpy as np
import pytest

from src.ai.chunkers import BreakpointSemanticChunker


class DummyCfg:
    gpu_backend = "cpu"
    embedding_model_path = "sentence-transformers/all-MiniLM-L6-v2"


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
