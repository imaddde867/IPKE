"""Few-shot prompt strategy with examples."""

import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


class FewShotPromptStrategy(PromptStrategy):
    """Few-shot prompting with example demonstrations."""

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

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self.template.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        return self._parse_json(response, chunk)


__all__ = ["FewShotPromptStrategy"]
