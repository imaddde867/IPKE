import numpy as np

from src.ai.knowledge_engine import UnifiedKnowledgeEngine
from src.core.unified_config import UnifiedConfig


def test_step_dedup_merges_similar_steps(monkeypatch):
    """Similar steps collapse to one canonical step with alias mapping."""
    engine = UnifiedKnowledgeEngine(UnifiedConfig())
    monkeypatch.setattr(
        engine,
        "_embed_chunk_texts",
        lambda texts: np.array([[1.0, 0.0], [0.95, 0.1]])[: len(texts)],
    )
    steps = [
        {"id": "Sx", "text": "Inspect valve"},
        {"id": "Sy", "text": "Inspect valve "},
    ]

    normalized, id_map, alias_map = engine._normalize_steps(steps)

    assert len(normalized) == 1
    assert id_map["Sx"] == "S1"
    assert id_map["Sy"] == "S1"
    assert alias_map["Sy"] == "Sx"


def test_constraint_dedup_and_fuzzy_attachment(monkeypatch):
    """Constraints with near-duplicate text merge and keep valid step links, including fuzzy matches."""
    engine = UnifiedKnowledgeEngine(UnifiedConfig())
    monkeypatch.setattr(
        engine,
        "_embed_chunk_texts",
        lambda texts: np.array([[1.0, 0.0], [1.0, 0.0]])[: len(texts)],
    )
    step_id_map = {"Sx": "S1"}
    step_alias_map = {"Sx": "Sx", "Sl": "Sx"}  # "Sl" is a near-miss for "Sx"

    constraints = [
        {"id": "C_raw", "text": "If pressure high then stop", "attached_to": ["Sx"]},
        {"id": "C_dup", "text": "If pressure is high then stop", "attached_to": ["Sx", "Sl"]},
    ]

    normalized = engine._normalize_constraints(constraints, step_id_map, step_alias_map)

    assert len(normalized) == 1
    assert normalized[0]["steps"] == ["S1"]
    assert normalized[0]["text"].startswith("If pressure")
