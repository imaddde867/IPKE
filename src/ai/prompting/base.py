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
    stop_sequences: List[str] = ["</s>", "[/INST]"]

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def run(self, backend: LLMBackend, chunk: str, document_type: str) -> ChunkExtraction:
        """Execute the prompt logic for a chunk and return structured output."""

    def _format_prompt(self, template: str, document_type: str, chunk: str, **extra: Any) -> str:
        """Format a prompt template with escaped document_type and chunk."""
        values: Dict[str, Any] = {
            "document_type": _escape_braces(document_type),
            "chunk": _escape_braces(chunk),
        }
        values.update(extra)
        return template.format(**values)

    def _parse_json(self, response: str, chunk: str) -> ChunkExtraction:
        """Parse LLM response into ChunkExtraction.
        
        All strategies should output constraints with "attached_to" array field
        (not "steps") for consistency across P0, P1, P2, P3.
        """
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

    lower = text.lower()
    start_tag = lower.find("<json>")
    end_tag = lower.find("</json>")
    if start_tag != -1 and end_tag != -1 and end_tag > start_tag:
        payload = text[start_tag + len("<json>"):end_tag].strip()
        return payload or None

    opening = text.find("{")
    if opening == -1:
        opening = text.find("[")
    if opening == -1:
        return None

    stack = []
    in_string = False
    escape = False
    for idx in range(opening, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener, ch) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                return text[opening : idx + 1]
    return None


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
