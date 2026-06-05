import numpy as np

from src.ai.types import ExtractedEntity
from src.graph.canonicalizer import (
    assign_canonical_ids,
    cluster_entities,
    normalize_entity_name,
)


class DummyEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vector = np.array(self.mapping.get(text, self.mapping.get("__default__", [1.0, 0.0])), dtype=float)
            if normalize_embeddings:
                norm = np.linalg.norm(vector) or 1.0
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


def test_normalize_entity_name_unifies_equipment_codes():
    a = normalize_entity_name("valve V-23")
    b = normalize_entity_name("V-23")
    c = normalize_entity_name("Valve V23")
    assert a == b == c


def test_cluster_entities_merges_semantic_equivalents():
    entities = [
        ExtractedEntity(content="fire extinguisher", entity_type="tool", category="safety", confidence=0.9),
        ExtractedEntity(content="extinguisher", entity_type="tool", category="safety", confidence=0.7),
    ]
    encoder = DummyEncoder({
        "fire extinguisher": [1.0, 0.0],
        "extinguisher": [0.95, 0.05],
    })
    clusters = cluster_entities(entities, threshold=0.8, encoder=encoder)
    assert len(clusters) == 1
    assert set(clusters[0]) == {0, 1}


def test_assign_canonical_ids_preserves_distinct_numeric_codes():
    entities = [
        ExtractedEntity(content="fire extinguisher", entity_type="tool", category="safety", confidence=0.9),
        ExtractedEntity(content="extinguisher", entity_type="tool", category="safety", confidence=0.7),
        ExtractedEntity(content="pump P-101", entity_type="equipment", category="pump", confidence=0.8),
        ExtractedEntity(content="pump P-102", entity_type="equipment", category="pump", confidence=0.85),
    ]
    encoder = DummyEncoder({
        "fire extinguisher": [1.0, 0.0],
        "extinguisher": [0.95, 0.05],
    })
    merged = assign_canonical_ids(entities, threshold=0.8, encoder=encoder)
    assert len(merged) == 3
    names = sorted(entity.content for entity in merged)
    assert "fire extinguisher" in names
    assert "pump p-101" in [name.lower() for name in names]
    assert "pump p-102" in [name.lower() for name in names]
    # Canonical IDs must be unique
    canonical_ids = {entity.canonical_id for entity in merged}
    assert len(canonical_ids) == len(merged)
