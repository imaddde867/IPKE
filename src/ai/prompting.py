"""Prompt strategy implementations (P0–P3)."""

from __future__ import annotations

import json
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from src.ai.types import ChunkExtraction, ExtractedEntity
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLMBackend(Protocol):
    """Minimal protocol implemented by LLM backends."""

    def generate(self, prompt: str, *, stop: Optional[List[str]] = None) -> str:
        ...


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


class SingleShotJSONStrategy(PromptStrategy):
    """Single call JSON prompting."""

    template: str = ""

    def run(self, backend: LLMBackend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self.template.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        return self._parse_json(response, chunk)


class ZeroShotJSONStrategy(SingleShotJSONStrategy):
    name = "P0"
    template = textwrap.dedent(
        """[INST] You produce lightweight procedural structure from {document_type} documents.

Read the provided text and return a SINGLE JSON object with concise fields:
{{
  "steps": [
    {{"id": "S1", "text": "Action statement written as an imperative.", "type": "procedure_step", "confidence": 0.9}},
    {{"id": "S2", "text": "Next ordered action.", "type": "procedure_step", "confidence": 0.88}}
  ],
  "constraints": [
    {{"id": "C1", "text": "Condition, warning, or requirement.", "steps": ["S1"], "confidence": 0.85}}
  ],
  "entities": [
    {{"content": "Supporting fact or measurement.", "type": "specification", "category": "distance_requirement", "confidence": 0.9}}
  ]
}}

Guidance:
- Keep IDs sequential (S1, S2, ..., C1, C2, ...).
- Steps must be actual actions or instructions in execution order.
- Constraints capture requirements, cautions, or prerequisites. Refer to step IDs when possible.
- Include granular facts in `entities` when helpful (measurements, tools, materials).
- Return only the JSON and nothing else.

Text:
\"\"\"{chunk}\"\"\"
[/INST]"""
    ).strip()


class FewShotPromptStrategy(SingleShotJSONStrategy):
    name = "P1"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.examples = self._build_examples()

    def _build_examples(self) -> str:
        return textwrap.dedent(
            """
Example 1 Input:
Inspect the hydraulic manifold. Close supply valve. Tag and lock out upstream pumps.

Example 1 Output:
{
  "steps": [
    {"id": "S1", "text": "Inspect the hydraulic manifold.", "confidence": 0.88},
    {"id": "S2", "text": "Close the supply valve.", "confidence": 0.86},
    {"id": "S3", "text": "Apply lockout tags to upstream pumps.", "confidence": 0.84}
  ],
  "constraints": [
    {"id": "C1", "text": "Follow site lockout procedures.", "steps": ["S3"], "confidence": 0.81}
  ],
  "entities": [
    {"content": "Hydraulic manifold", "type": "equipment", "confidence": 0.8}
  ]
}
"""
        ).strip()

    @property
    def template(self) -> str:
        return textwrap.dedent(
            """[INST]You convert industrial instructions into JSON steps, constraints, and supporting entities.
Study the few-shot example and mimic its structure.

{examples}

Now process the following {document_type} text and output ONLY JSON:
\"\"\"{chunk}\"\"\"
[/INST]"""
        ).strip().format(examples=self.examples, document_type="{document_type}", chunk="{chunk}")


class CoTPromptStrategy(SingleShotJSONStrategy):
    name = "P2"

    template = textwrap.dedent(
        """[INST]Reason step-by-step about the following {document_type} text.
After reasoning, emit the FINAL JSON between <json></json> tags.

Reasoning guidelines:
- Identify discrete procedure steps in order.
- Note constraints/guards that apply to those steps.
- Capture auxiliary facts as entities.

Text:
\"\"\"{chunk}\"\"\"

<json>
{{ "steps": [], "constraints": [], "entities": [] }}
</json>
[/INST]"""
    ).strip()

    def run(self, backend: LLMBackend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self.template.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        payload = response
        if "<json" in response.lower():
            start = response.lower().find("<json")
            closing = response.lower().rfind("</json>")
            if closing != -1:
                payload = response[start:closing]
        return self._parse_json(payload, chunk)


class TwoStageSchemaStrategy(PromptStrategy):
    """Two-stage prompting: steps first, then attach constraints/entities."""

    name = "P3"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.stage1_prompt = textwrap.dedent(
            """[INST]Extract ordered procedure steps from {document_type} content.
Respond with a JSON object containing only the "steps" array as shown:
{{"steps": [{{"id": "S1", "text": "Action", "confidence": 0.9}}]}}

Text:
\"\"\"{chunk}\"\"\"
[/INST]"""
        ).strip()
        self.stage2_prompt = textwrap.dedent(
            """[INST]You are provided with extracted steps:
{steps_json}

Now enrich them with constraints and supporting entities derived from:
\"\"\"{chunk}\"\"\"

Output a JSON object with fields steps (reuse IDs), constraints, and entities.
[/INST]"""
        ).strip()

    def run(self, backend: LLMBackend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt1 = self.stage1_prompt.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        step_response = backend.generate(prompt1, stop=["</s>", "[/INST]"])
        step_extraction = self._parse_json(step_response, chunk)
        steps_json = json.dumps({"steps": step_extraction.steps}, ensure_ascii=False)
        safe_steps_json = steps_json.replace("{", "{{").replace("}", "}}")
        prompt2 = self.stage2_prompt.format(steps_json=safe_steps_json, chunk=_escape_braces(chunk))
        combined_response = backend.generate(prompt2, stop=["</s>", "[/INST]"])
        final = self._parse_json(combined_response, chunk)
        if not final.steps:
            final.steps = step_extraction.steps
        return final


def build_prompt_strategy(config) -> PromptStrategy:
    strategy = getattr(config, "prompting_strategy", "P0").upper()
    if strategy == "P1":
        return FewShotPromptStrategy(config)
    if strategy == "P2":
        return CoTPromptStrategy(config)
    if strategy == "P3":
        return TwoStageSchemaStrategy(config)
    return ZeroShotJSONStrategy(config)


__all__ = [
    "PromptStrategy",
    "ZeroShotJSONStrategy",
    "FewShotPromptStrategy",
    "CoTPromptStrategy",
    "TwoStageSchemaStrategy",
    "build_prompt_strategy",
]
