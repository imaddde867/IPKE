"""Chain-of-Thought prompt strategy."""

import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


class CoTPromptStrategy(PromptStrategy):
    """Chain-of-Thought prompting with reasoning steps."""

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

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self.template.format(document_type=_escape_braces(document_type), chunk=_escape_braces(chunk))
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        payload = response
        if "<json" in response.lower():
            start = response.lower().find("<json")
            closing = response.lower().rfind("</json>")
            if closing != -1:
                payload = response[start:closing]
        return self._parse_json(payload, chunk)


__all__ = ["CoTPromptStrategy"]
