"""Chain-of-Thought prompt strategy with explicit reasoning decomposition."""

import textwrap

from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction


class CoTPromptStrategy(PromptStrategy):
    """Chain-of-Thought reasoning with explicit decomposition steps.
    
    Implements Wei et al. (2022) CoT approach with four reasoning stages:
    1. CLASSIFY: Distinguish between Descriptive and Procedural content
    2. FILTER: Extract human actions and ignore non-human events
    3. ATTACH: Link preconditions and safety guards to steps
    4. EMIT: Generate structured JSON output
    """

    name = "P2"

    @property
    def template(self) -> str:
        return textwrap.dedent(
            """[INST]You are an expert technical analyst specializing in procedural knowledge extraction.
Your goal is to extract a structured procedural graph from industrial documents.

Perform the following reasoning steps in natural language BEFORE generating the final JSON:

STEP 1: CLASSIFY
Determine if the text is "Descriptive" (facts, definitions, biology) or "Procedural" (instructions, rules, maintenance).
- If Descriptive: Explicitly state "TEXT IS DESCRIPTIVE - NO STEPS TO EXTRACT"
- If Procedural: Identify the primary HUMAN ACTOR(s) (e.g., Operator, Technician, Engineer, Inspector)

STEP 2: FILTER & EXTRACT STEPS
List candidate procedural steps:
- Include: Actionable instructions with clear verbs and objects (e.g., "Inspect valve", "Apply pressure")
- Exclude: Non-human actions (e.g., "Beetle eats grain", "Water evaporates")
- Exclude: General observations (e.g., "Fire is dangerous")
- Exclude: Passive states (e.g., "The pump is running")
Keep ONLY human-initiated actions in execution order.

STEP 3: ATTACH CONSTRAINTS
For each identified step, scan the immediate context for:
- Preconditions: "before", "after", "ensure", "first", "prior to", "initially"
- Safety guards: "must", "must not", "never", "if", "only if", "provided that"
- Quantitative specs: temperatures, pressures, distances, durations, equipment names
- Environmental conditions: "when", "during", "in case of", "under conditions"
Explicitly link each constraint to its step ID.

STEP 4: EMIT JSON
Convert your analysis into strict JSON with "steps", "constraints", and "entities" arrays.
Use the schema:
{{
  "steps": [{{"id": "S1", "text": "Action", "order": 1, "confidence": 0.9}}],
  "constraints": [{{"id": "C1", "expression": "Guard", "type": "guard", "attached_to": ["S1"], "confidence": 0.85}}],
  "entities": [{{"content": "Item", "type": "material", "category": "material", "confidence": 0.9}}]
}}

---

Text to analyze (document_type: {{document_type}}):
\"\"\"{{chunk}}\"\"\"

---

Now provide your reasoning (1-2 sentences for each step), then output ONLY the JSON object between <json></json> tags:

<json>
{{ "steps": [], "constraints": [], "entities": [] }}
</json>
[/INST]"""
        ).strip()

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self._format_prompt(self.template, document_type, chunk)
        response = backend.generate(prompt, stop=self.stop_sequences)
        return self._parse_json(response, chunk)


__all__ = ["CoTPromptStrategy"]
