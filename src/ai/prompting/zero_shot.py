"""Zero-shot JSON prompting strategy."""

import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


class ZeroShotJSONStrategy(PromptStrategy):
    """Single call JSON prompting without examples."""

    name = "P0"

    @property
    def template(self) -> str:
        return textwrap.dedent(
            """[INST] You produce lightweight procedural structure from {document_type} documents.

Read the provided text and return a SINGLE JSON object with concise fields:
{{
  "steps": [
    {{"id": "S1", "text": "Action statement written as an imperative.", "type": "procedure_step", "confidence": 0.9}},
    {{"id": "S2", "text": "Next ordered action.", "type": "procedure_step", "confidence": 0.88}}
  ],
  "constraints": [
    {{"id": "C1", "text": "Condition, warning, or requirement.", "attached_to": ["S1"], "confidence": 0.85}}
  ],
  "entities": [
    {{"content": "Supporting fact or measurement.", "type": "specification", "category": "distance_requirement", "confidence": 0.9}}
  ]
}}

Guidance:
- Keep IDs sequential (S1, S2, ..., C1, C2, ...).
- Steps must be actual actions or instructions in execution order.
- Constraints capture requirements, cautions, or prerequisites. Use "attached_to" array to link constraints to step IDs.
- Include granular facts in `entities` when helpful (measurements, tools, materials).
- Return only the JSON and nothing else.

Text:
\"\"\"{chunk}\"\"\"
[/INST]"""
        ).strip()

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self._format_prompt(self.template, document_type, chunk)
        response = backend.generate(prompt, stop=self.stop_sequences)
        return self._parse_json(response, chunk)


__all__ = ["ZeroShotJSONStrategy"]
