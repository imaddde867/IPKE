"""Entity canonicalization utilities.

This module provides lightweight helpers to normalize entity names,
cluster semantically similar mentions, and assign canonical IDs so that
entities extracted from different chunks can be merged reliably.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import guard
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore

from src.ai.types import ExtractedEntity


_EMBEDDER = None


def normalize_entity_name(text: Optional[str]) -> str:
    """Return a canonical lowercase representation for fuzzy matching."""

    if not text:
        return ""
    normalized = text.strip().lower()
    normalized = normalized.replace("–", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9\-\/\s]", " ", normalized)
    normalized = re.sub(r"([a-z])-(\d)", r"\1\2", normalized)
    normalized = re.sub(r"(\d)-([a-z])", r"\1\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    tokens = normalized.split()
    code_tokens = []
    for token in tokens:
        compact = token.replace("-", "")
        if _has_digit(compact):
            code_tokens.append(compact)
    if code_tokens:
        return " ".join(code_tokens)
    return normalized.strip()


def cluster_entities(
    entities: Sequence[ExtractedEntity],
    threshold: float = 0.85,
    encoder: Optional[Any] = None,
) -> List[List[int]]:
    """Cluster entities by semantic similarity.

    Returns a list of clusters, where each cluster is a list of indices
    pointing into the ``entities`` sequence.
    """

    if not entities:
        return []

    normalized = [normalize_entity_name(entity.content) for entity in entities]
    counts = Counter(normalized)
    clusters: List[List[int]] = []
    assigned: set[int] = set()

    # Exact-match grouping first (covers codes like V-23 reliably)
    for norm_value, count in counts.items():
        if norm_value and count > 1:
            idxs = [idx for idx, norm in enumerate(normalized) if norm == norm_value]
            clusters.append(idxs)
            assigned.update(idxs)

    remaining = [idx for idx in range(len(entities)) if idx not in assigned]
    if not remaining:
        return clusters

    digit_limited = []
    semantic_candidates = []
    for idx in remaining:
        norm = normalized[idx]
        if _has_digit(norm):
            digit_limited.append([idx])
        else:
            semantic_candidates.append(idx)

    clusters.extend(digit_limited)

    if not semantic_candidates:
        # No semantic work needed: remaining entries already in clusters
        singles = [[idx] for idx in remaining if idx not in {i for group in digit_limited for i in group}]
        clusters.extend(singles)
        return clusters

    texts = [_text_for_embedding(entities[idx], normalized[idx]) for idx in semantic_candidates]
    embeddings = _embed_texts(texts, encoder=encoder)
    sem_clusters: List[dict] = []

    for vector, idx in zip(embeddings, semantic_candidates):
        placed = False
        for cluster in sem_clusters:
            similarity = float(np.dot(vector, cluster["centroid"]))
            if similarity >= threshold:
                cluster["indices"].append(idx)
                cluster["centroid"] = _normalize_vector(
                    (cluster["centroid"] * (len(cluster["indices"]) - 1) + vector) / len(cluster["indices"])
                )
                placed = True
                break
        if not placed:
            sem_clusters.append({"indices": [idx], "centroid": _normalize_vector(vector)})

    clusters.extend(cluster["indices"] for cluster in sem_clusters)

    # Include any singletons not touched above
    already_grouped = {idx for cluster in clusters for idx in cluster}
    for idx in range(len(entities)):
        if idx not in already_grouped:
            clusters.append([idx])

    return clusters


def assign_canonical_ids(
    entities: Sequence[ExtractedEntity],
    threshold: float = 0.85,
    encoder: Optional[Any] = None,
) -> List[ExtractedEntity]:
    """Merge entities and assign canonical IDs."""

    if not entities:
        return []

    clusters = cluster_entities(entities, threshold=threshold, encoder=encoder)
    merged: List[ExtractedEntity] = []

    for cluster_idx, cluster in enumerate(clusters, start=1):
        members = [entities[i] for i in cluster if i < len(entities)]
        if not members:
            continue
        aliases = [m.content.strip() for m in members if m.content]
        alias_list = list(dict.fromkeys(alias for alias in aliases if alias))
        canonical_name = _choose_canonical_alias(aliases)
        canonical_id = f"E{cluster_idx}"
        metadata = _merge_metadata(members)
        metadata["aliases"] = alias_list
        metadata["canonical_name"] = canonical_name
        metadata["source_count"] = len(members)

        merged.append(
            ExtractedEntity(
                content=canonical_name,
                entity_type=_most_common(m.entity_type for m in members) or members[0].entity_type,
                category=_most_common(m.category for m in members) or members[0].category,
                confidence=max(m.confidence for m in members),
                context=_select_context(members),
                metadata=metadata,
                canonical_id=canonical_id,
            )
        )

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def _text_for_embedding(entity: ExtractedEntity, normalized: str) -> str:
    return normalized or entity.content.lower()


def _embed_texts(texts: Sequence[str], encoder: Optional[Any] = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if encoder is not None:
        embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    else:
        embedder = _get_embedder()
        if embedder is None:
            embeddings = _lexical_fallback(texts)
        else:
            embeddings = embedder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    return arr


def _get_embedder():  # pragma: no cover - exercised via integration
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if SentenceTransformer is None:
        return None
    model_id = os.environ.get("ENTITY_CANONICALIZER_MODEL", "all-mpnet-base-v2")
    _EMBEDDER = SentenceTransformer(model_id)
    return _EMBEDDER


def _lexical_fallback(texts: Sequence[str]) -> np.ndarray:
    vocab_size = 64
    matrix = np.zeros((len(texts), vocab_size), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.split():
            idx = hash(token) % vocab_size
            matrix[row, idx] += 1.0
        norm = np.linalg.norm(matrix[row]) or 1.0
        matrix[row] = matrix[row] / norm
    return matrix


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector) or 1.0
    return vector / norm


def _choose_canonical_alias(aliases: Sequence[str]) -> str:
    if not aliases:
        return ""
    for alias in aliases:
        if _has_digit(alias):
            return alias
    return aliases[0]


def _merge_metadata(members: Sequence[ExtractedEntity]) -> dict:
    merged = {}
    for member in members:
        if member.metadata:
            merged.update(member.metadata)
    return merged


def _select_context(members: Sequence[ExtractedEntity]) -> str:
    for member in members:
        if member.context:
            return member.context
    return members[0].context if members else ""


def _most_common(values: Iterable[str]) -> Optional[str]:
    filtered = [value for value in values if value]
    if not filtered:
        return None
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


__all__ = [
    "normalize_entity_name",
    "cluster_entities",
    "assign_canonical_ids",
]
