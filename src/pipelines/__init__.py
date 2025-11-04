"""Utility pipelines aggregating document extraction workflows."""

from .baseline import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GOLD_DIR,
    TIER_A_TEST_DOCS,
    accumulate_metrics,
    evaluate_predictions,
    extraction_payload,
    extract_documents,
    summarise_metrics,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GOLD_DIR",
    "TIER_A_TEST_DOCS",
    "accumulate_metrics",
    "evaluate_predictions",
    "extraction_payload",
    "extract_documents",
    "summarise_metrics",
]
