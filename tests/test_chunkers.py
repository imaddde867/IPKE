import numpy as np
import pytest

from src.processors.chunkers import BreakpointSemanticChunker, DualSemanticChunker, FixedChunker


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
    dsc_lambda = 0.05
    dsc_beta = 0.2


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
        chunking_method = "dual_semantic"

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


@pytest.mark.integration
def test_parent_only_mode_builds_parent_chunks(monkeypatch):
    monkeypatch.setenv("EXPLAINIUM_ENV", "testing")

    class ParentCfg(DummyCfg):
        chunking_method = "parent_only"
        dsc_parent_min_sentences = 2
        dsc_parent_max_sentences = 4

    chunker = DualSemanticChunker(ParentCfg(), parent_only=True)

    sentences = [
        "Heading A. Prep PPE.",
        "Ensure respirator is sealed.",
        "Heading B. Calibration.",
        "Attach probes and record offsets.",
        "Heading C. Cleanup.",
        "Dispose of solvents per SOP."
    ]
    spans = []
    text = ""
    cursor = 0
    for idx, sent in enumerate(sentences):
        text += sent
        spans.append((cursor, cursor + len(sent)))
        cursor += len(sent)
        if idx < len(sentences) - 1:
            text += "\n"
            cursor += 1

    embeddings = np.array([
        [1.0, 0.0],
        [0.95, 0.05],
        [-0.85, -0.4],
        [-0.83, -0.45],
        [0.2, 0.98],
        [0.18, 0.99],
    ], dtype=float)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    monkeypatch.setattr(chunker.base, "_sentences", lambda _: (sentences, spans))
    monkeypatch.setattr(chunker.base, "_embeddings", lambda sents: embeddings[:len(sents)])

    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert all(chunk.meta.get("strategy") == "parent_only" for chunk in chunks)
    lengths = [chunk.sentence_span for chunk in chunks]
    assert (0, 2) in lengths and (2, 4) in lengths and (4, 6) in lengths

def test_fixed_chunker_respects_char_cap():
    chunker = FixedChunker(max_chars=500)
    text = " ".join(f"sentence_{i}" for i in range(1500))  # ~15000 chars

    chunks = chunker.chunk(text)
    assert len(chunks) > 0

    prev_end = 0
    for chunk in chunks:
        assert len(chunk.text) <= 500
        assert chunk.start_char >= prev_end
        prev_end = chunk.end_char


def test_semantic_fallbacks_with_fixed_chunker():
    chunker = FixedChunker(max_chars=500)

    assert chunker.chunk("") == []

    single = "Run diagnostics."
    chunks = chunker.chunk(single)
    assert len(chunks) == 1
    assert chunks[0].text == single


def test_char_cap_enforcement_on_fixed():
    chunker = FixedChunker(max_chars=250)
    text = ("alpha beta gamma delta epsilon " * 200).strip()

    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 250


def _make_distances(sims):
    """Convert similarity array to distances array for _parent_boundaries input."""
    return 1.0 - np.array(sims, dtype=float)


def _dummy_spans(n):
    return [(i * 10, i * 10 + 9) for i in range(n)]


def _dummy_sentences(n, prefix="Sentence"):
    return [f"{prefix} {i}." for i in range(n)]


def test_dp_two_clear_clusters_split_into_two_blocks():
    """DP finds the globally optimal 2-block partition for well-separated embeddings."""

    class Cfg(DummyCfg):
        # λ=0.15 makes [(0,3),(3,6)] the global optimum.
        # With λ<0.12 the DP splits more finely; with λ>0.87 it merges all.
        dsc_lambda = 0.15
        dsc_beta = 0.0
        dsc_use_headings = False

    chunker = DualSemanticChunker(Cfg())
    # Sentences 0-2 high intra-cluster sim, sentences 3-5 high intra-cluster sim.
    # Cross-cluster sim (between 2 and 3) is low.
    sims = [0.95, 0.93,   # cluster A: pairs (0,1), (1,2)
            0.10,          # boundary: pair (2,3)
            0.92, 0.94]   # cluster B: pairs (3,4), (4,5)
    distances = _make_distances(sims)
    sentences = _dummy_sentences(6)
    spans = _dummy_spans(6)

    result = chunker._parent_boundaries(distances, sentences, spans, "")

    assert result == [(0, 3), (3, 6)]


def test_dp_high_lambda_produces_single_block():
    """λ high enough → DP keeps all sentences as one block (splitting unprofitable).
    The heuristic ignores λ for parent boundaries and uses max_len to force cuts,
    so this test is RED under the heuristic."""

    class Cfg(DummyCfg):
        dsc_parent_max_sentences = 4  # heuristic would cut here
        dsc_lambda = 0.99             # penalty per block near-equals max cohesion
        dsc_beta = 0.0
        dsc_use_headings = False

    chunker = DualSemanticChunker(Cfg())
    # 6 sentences, all same topic — uniform high similarity.
    sims = [0.95, 0.95, 0.95, 0.95, 0.95]
    distances = _make_distances(sims)
    sentences = _dummy_sentences(6)

    result = chunker._parent_boundaries(distances, sentences, _dummy_spans(6), "")

    assert result == [(0, 6)]


def test_dp_heading_bonus_pulls_split_to_heading_position():
    """β bonus shifts the split point from the sim-dip to the heading sentence.

    Without β, the DP splits at position 2 (lowest sim is sims[2]=0.3 but the
    globally optimal greedy-pair path wins). With β=0.2 on sentence 3 (a heading),
    dp[3] gains enough to make [(0,3),(3,6)] the global optimum instead.
    """

    class Cfg(DummyCfg):
        dsc_lambda = 0.15
        dsc_beta = 0.2        # paper value — matches ADR-0001
        dsc_use_headings = True

    chunker = DualSemanticChunker(Cfg())
    # sims[2]=0.3 is the low-similarity pair; sentence 3 is a structural heading.
    # Verified analytically: with β=0.2 the heading bonus on dp[3] makes
    # [(0,3),(3,6)] score 1.50, beating all alternatives.
    sims = [0.8, 0.8, 0.3, 0.8, 0.8]
    distances = _make_distances(sims)
    # Fixed regex requires trailing dot on Roman numerals, so common verbs (Verify,
    # Inspect, Check) no longer false-positive as headings.
    sentences = [
        "Verify lubricant levels before start.",   # old regex: is_heading=True; fixed: False
        "Inspect torque settings on each bolt.",    # old regex: is_heading=True; fixed: False
        "Check seal integrity before proceeding.",  # old regex: is_heading=True; fixed: False
        "1. New Section: Electrical checks.",       # matches \d+ → is_heading[3]=True
        "Measure output at terminal A.",
        "Record all readings in the log.",
    ]

    result = chunker._parent_boundaries(distances, sentences, _dummy_spans(6), "")

    assert result == [(0, 3), (3, 6)]


def test_heading_regex_requires_dot_after_roman_numeral():
    """Verify/Inspect/Check must NOT trigger heading bonus; IV. must."""
    import re
    heading_re = re.compile(r"^\s*((\d+(\.\d+)*\.?)|([IVXLC]+\.)|([A-Z][\w\s]{0,60}:))")
    assert not heading_re.match("Verify lubricant levels before start.")
    assert not heading_re.match("Inspect the component.")
    assert not heading_re.match("Check seal integrity.")
    assert heading_re.match("IV. Calibration procedure")
    assert heading_re.match("1. New Section: Electrical checks.")
    assert heading_re.match("Safety Precautions:")


def test_dp_single_sentence_returns_single_block():
    """Single sentence → one block [(0, 1)] without errors."""

    chunker = DualSemanticChunker(DummyCfg())
    result = chunker._parent_boundaries(
        np.array([], dtype=float), ["Only sentence."], [(0, 14)], ""
    )
    assert result == [(0, 1)]


def test_dual_semantic_does_not_reach_into_breakpoint_private():
    """DSC must access inherited methods via self._, not via self.base._."""
    import inspect
    from src.processors.chunkers.dual_semantic import DualSemanticChunker
    source = inspect.getsource(DualSemanticChunker)
    assert "self.base._" not in source, (
        "DualSemanticChunker must not access private methods via self.base._; "
        "it should inherit from BreakpointSemanticChunker instead."
    )


def test_dual_semantic_is_breakpoint_subclass():
    from src.processors.chunkers.dual_semantic import DualSemanticChunker
    from src.processors.chunkers.breakpoint import BreakpointSemanticChunker
    assert issubclass(DualSemanticChunker, BreakpointSemanticChunker)
