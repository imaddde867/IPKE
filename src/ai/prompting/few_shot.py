"""Few-shot prompt strategy with examples demonstrating constraint types."""

import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


class FewShotPromptStrategy(PromptStrategy):
    """Few-shot in-context learning with annotated examples showing constraint types and parameters."""

    name = "P1"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.examples = self._build_examples()

    def _build_examples(self) -> str:
        return textwrap.dedent(
            """
Example 1 Input:
Mix 3M™ Marine Blister Repair Filler and apply. Fill 85% of the cavity. If needed, feather the repair with 3M™ Hookit™ 775L Cubitron™ II abrasive disc after 30 minutes.

Example 1 Output:
{{
  "steps": [
    {{"id": "S1", "text": "Mix 3M™ Marine Blister Repair Filler and apply", "order": 1, "confidence": 0.92}},
    {{"id": "S2", "text": "Fill 85% of the cavity", "order": 2, "confidence": 0.90}},
    {{"id": "S3", "text": "Feather the repair with 3M™ Hookit™ 775L Cubitron™ II abrasive disc", "order": 3, "confidence": 0.88}}
  ],
  "constraints": [
    {{"id": "C1", "expression": "If needed", "type": "guard", "attached_to": ["S3"], "confidence": 0.85}},
    {{"id": "C2", "expression": "Fill 85% of the cavity", "type": "parameter", "attached_to": ["S2"], "confidence": 0.89, "parameters": {{"fill_level": {{"value": 85, "unit": "%"}}}}}},
    {{"id": "C3", "expression": "After 30 minutes", "type": "temporal", "attached_to": ["S3"], "confidence": 0.87, "parameters": {{"wait_time": {{"value": 30, "unit": "minutes"}}}}}}
  ],
  "entities": [
    {{"content": "3M™ Marine Blister Repair Filler", "type": "material", "category": "material", "confidence": 0.95}},
    {{"content": "3M™ Hookit™ 775L Cubitron™ II abrasive disc", "type": "tool", "category": "tool", "confidence": 0.92}},
    {{"content": "grade 80+", "type": "specification", "category": "parameter", "confidence": 0.88}}
  ]
}}

Example 2 Input:
Inspect the engine compartment daily. Clean the engine compartment when operating in dusty or fire-hazard conditions.

Example 2 Output:
{{
  "steps": [
    {{"id": "S1", "text": "Inspect the engine compartment", "order": 1, "confidence": 0.91}},
    {{"id": "S2", "text": "Clean the engine compartment", "order": 2, "confidence": 0.89}}
  ],
  "constraints": [
    {{"id": "C1", "expression": "Frequency: daily", "type": "schedule", "attached_to": ["S1", "S2"], "confidence": 0.90, "parameters": {{"frequency": {{"value": 1, "unit": "day"}}}}}},
    {{"id": "C2", "expression": "When operating in dusty or fire-hazard conditions", "type": "environmental_guard", "attached_to": ["S1", "S2"], "confidence": 0.87, "parameters": {{"conditions": ["dusty", "fire_hazard"]}}}}
  ],
  "entities": [
    {{"content": "engine compartment", "type": "component", "category": "component", "confidence": 0.94}},
    {{"content": "dusty conditions", "type": "environmental", "category": "condition", "confidence": 0.88}},
    {{"content": "fire-hazard conditions", "type": "environmental", "category": "condition", "confidence": 0.86}}
  ]
}}
"""
        ).strip()

    @property
    def template(self) -> str:
        base = textwrap.dedent(
            """[INST]You are an expert at extracting procedural knowledge from industrial documents.
Your task is to identify steps, constraints, and entities. Pay special attention to distinguishing between:
  - Steps (S*): Concrete actions in execution order
  - Constraints (C*): Conditions, guards, temporal markers, parameters, and schedules that modify or guard steps
  - Entities (E*): Materials, tools, components, and specifications

Study these annotated examples showing how to capture constraint types and their parameters:

{examples}

Key guidance:
- STEPS are actions or instructions in execution order
- CONSTRAINTS capture requirements/conditions/guards/schedules. Types include:
    * guard: conditional prerequisites ("If X", "Only if Y")
    * parameter: quantitative limits or specifications ("85% fill", "grade 80+")
    * temporal: timing or duration constraints ("After 30 min", "Wait until dry")
    * schedule: frequency or recurrence ("Daily", "Weekly")
    * environmental_guard: contextual conditions ("When dusty", "In high temperature")
- ENTITIES are concrete materials, tools, components, conditions, and specifications
- Use "attached_to" to link constraints to step IDs
- When constraints are part of steps, extract them as separate constraint objects
- Always include "type" for constraints and "order" for steps
- Include "parameters" dict when a constraint has measurable/structured data

Now process the following {{document_type}} text and output ONLY JSON:
\"\"\"{{chunk}}\"\"\"
[/INST]"""
        ).strip()
        return base.format(examples=self.examples)

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self.template.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        return self._parse_json(response, chunk)


__all__ = ["FewShotPromptStrategy"]
