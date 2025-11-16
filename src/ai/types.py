"""Typed data models shared across LLM components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExtractedEntity:
    content: str
    entity_type: str
    category: str
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkExtraction:
    entities: List[ExtractedEntity] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity]
    confidence_score: float
    processing_time: float
    strategy_used: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ChunkExtraction",
    "ExtractedEntity",
    "ExtractionResult",
]
