"""Base interface and utilities for prompt strategies."""

from __future__ import annotations

import json
import re
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
        
        Robustly attempts to find JSON in the response.
        """
        payload = _extract_json_payload(response)
        if not payload:
            logger.warning("No JSON found in response. Response length: %d", len(response))
            return ChunkExtraction()
            
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # Try to repair common JSON errors (like trailing commas)
            try:
                cleaned = re.sub(r",\s*([\]}])", r"\1", payload)
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON payload from response: %.80s...", payload.replace("\n", " "))
                return ChunkExtraction()
        
        # flexible key matching
        steps_key = next((k for k in data.keys() if k.lower() in ["steps", "procedures", "actions"]), "steps")
        constraints_key = next((k for k in data.keys() if k.lower() in ["constraints", "conditions", "rules"]), "constraints")
        entities_key = next((k for k in data.keys() if k.lower() in ["entities", "items", "objects"]), "entities")

        steps = _ensure_dict_list(data.get(steps_key))
        constraints = _ensure_dict_list(data.get(constraints_key))
        entities = _build_entities(_ensure_dict_list(data.get(entities_key)), chunk)
        
        return ChunkExtraction(entities=entities, steps=steps, constraints=constraints)


def _extract_json_payload(text: str) -> Optional[str]:
    """Return the JSON portion from an LLM response.
    
    Priority:
    1. Content between <json> and </json> tags (explicit)
    2. Content between ```json and ``` (code blocks)
    3. The last valid {...} or [...] block in the text (heuristic for CoT)
    """
    if not text:
        return None

    # 1. Explicit tags (highest priority)
    pattern_tags = r"<json>(.*?)</json>"
    match = re.search(pattern_tags, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Markdown code blocks
    pattern_code = r"```(?:json)?(.*?)```"
    match = re.search(pattern_code, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 3. Brute force: Find the largest valid JSON object
    # This is useful when the model outputs "Reasoning... {JSON}" without tags
    candidates = []
    stack = []
    start = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
    
    # If we found candidates, try to parse them backwards (assuming the last one is the result)
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    # Fallback for simple errors (trailing characters)
    clean_text = text.strip()
    if clean_text.startswith("{") and clean_text.endswith("}"):
        return clean_text
        
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
