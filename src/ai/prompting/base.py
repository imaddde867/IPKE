"""Base interface and utilities for prompt strategies."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from src.ai.types import ChunkExtraction, ExtractedEntity
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLMBackend(Protocol):
    """Minimal protocol implemented by LLM backends."""

    def generate(self, prompt: str, *, stop: Optional[List[str]] = None) -> str:
        ...


class PromptStrategy(ABC):
    """Base interface for prompting regimes."""

    name: str = "P0"

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def run(self, backend: LLMBackend, chunk: str, document_type: str) -> ChunkExtraction:
        """Execute the prompt logic for a chunk and return structured output."""

    def _parse_json(self, response: str, chunk: str) -> ChunkExtraction:
        payload = _extract_json_payload(response)
        if not payload:
            return ChunkExtraction()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON payload from response: %.80s", response)
            return ChunkExtraction()
        steps = _ensure_dict_list(data.get("steps"))
        constraints = _ensure_dict_list(data.get("constraints"))
        entities = _build_entities(_ensure_dict_list(data.get("entities")), chunk)
        return ChunkExtraction(entities=entities, steps=steps, constraints=constraints)


def _extract_json_payload(text: str) -> Optional[str]:
    """Return the JSON portion from an LLM response."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    return text[start:end]


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return []


def _build_entities(items: List[Dict[str, Any]], chunk: str) -> List[ExtractedEntity]:
    entities: List[ExtractedEntity] = []
    for item in items:
        content = (item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        entities.append(
            ExtractedEntity(
                content=content,
                entity_type=item.get("type", "unknown"),
                category=item.get("category", "general"),
                confidence=_coerce_confidence(item.get("confidence")),
                context=(chunk[:200] + "...") if chunk else "",
                metadata={"llm_extracted": True},
            )
        )
    return entities


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


__all__ = [
    "LLMBackend",
    "PromptStrategy",
    "_extract_json_payload",
    "_coerce_confidence",
    "_ensure_dict_list",
    "_build_entities",
    "_escape_braces",
]
