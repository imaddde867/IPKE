"""Two-stage prompt strategy with sequential extraction."""

import json
import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


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

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
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


__all__ = ["TwoStageSchemaStrategy"]
