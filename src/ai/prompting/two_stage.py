"""Two-stage prompt strategy with sequential extraction.

P3 Strategy: Reduces cognitive load by splitting extraction into two stages:
- Stage 1: Extract steps (skeleton) - minimal cognitive load on step ordering
- Stage 2: Extract constraints & entities with strict validation that all constraints
  must reference existing step IDs from Stage 1
"""

import json
import textwrap
from typing import List, Dict, Any, Optional

from src.ai.prompting.base import PromptStrategy, _escape_braces, _extract_json_payload
from src.ai.types import ChunkExtraction
from src.logging_config import get_logger

logger = get_logger(__name__)


class TwoStageSchemaStrategy(PromptStrategy):
    """Two-stage prompting reduces cognitive load by splitting extraction.
    
    Stage 1: Extract only procedural steps in execution order (skeleton)
    Stage 2: Extract constraints and entities with strict validation:
             - All constraints MUST reference pre-existing step IDs from Stage 1
             - This prevents hallucination of constraints that don't attach to any step
    
    This approach is particularly effective for smaller models (e.g., Mistral-7B)
    that are vulnerable to instruction drift where they lose track of ID-linking rules.
    """

    name = "P3"

    def __init__(self, config) -> None:
        super().__init__(config)

    @property
    def stage1_template(self) -> str:
        return textwrap.dedent(
            """[INST]You are an expert at extracting procedural knowledge from technical documents.

TASK: Extract ONLY the ordered procedural steps. Ignore constraints, prerequisites, and entities for now.

Focus ONLY on identifying and ordering actionable human steps in execution sequence.

From the following {document_type} text, extract steps with:
- id: Unique step ID (S1, S2, S3, etc.)
- text: Concise description of the action
- order: Sequence number (1, 2, 3, etc.)
- confidence: Your confidence (0.0 to 1.0)

Respond ONLY with a JSON object:
{{"steps": [{{"id": "S1", "text": "Action", "order": 1, "confidence": 0.9}}]}}

Text:
"{chunk}"
[/INST]"""
        ).strip()

    @property
    def stage2_template(self) -> str:
        return textwrap.dedent(
            """[INST]You are an expert at extracting procedural knowledge from technical documents.

You have already extracted these steps:
{steps_json}

TASK: Now extract constraints and entities from the same text, with a CRITICAL RULE:
  >> EVERY extracted constraint MUST explicitly reference one of the step IDs above <<
  >> Do NOT create constraints that don't reference any step ID <<

From the text, extract:
- Constraints (C*): Guard conditions, preconditions, timing, frequency, parameters, environmental conditions
  * MUST have "attached_to" array with at least one valid step ID from above
- Entities: Materials, tools, components, specifications
  * MUST be relevant to the procedural context

Respond ONLY with a JSON object:
{{"steps": [], "constraints": [{{"id": "C1", "expression": "Condition", "type": "guard", "attached_to": ["S1"], "confidence": 0.85}}], "entities": [{{"content": "Item", "type": "material", "category": "material", "confidence": 0.9}}]}}

Original text:
"{chunk}"
[/INST]"""
        ).strip()

    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        stage1_prompt = self._format_prompt(self.stage1_template, document_type, chunk)
        step_response = backend.generate(stage1_prompt, stop=self.stop_sequences)
        step_extraction = self._parse_json(step_response, chunk)

        if not step_extraction.steps:
            logger.debug("Stage 1: No steps extracted from chunk")
            return step_extraction

        valid_step_ids = {step.get("id") for step in step_extraction.steps if step.get("id")}
        steps_json = json.dumps({"steps": step_extraction.steps}, ensure_ascii=False)
        safe_steps_json = _escape_braces(steps_json)

        stage2_prompt = self.stage2_template.format(
            steps_json=safe_steps_json,
            chunk=_escape_braces(chunk)
        )
        combined_response = backend.generate(stage2_prompt, stop=self.stop_sequences)
        final = self._parse_json(combined_response, chunk)

        final.constraints = self._validate_constraints(final.constraints, valid_step_ids)

        if not final.steps:
            final.steps = step_extraction.steps

        return final

    def _validate_constraints(
        self, constraints: List[Dict[str, Any]], valid_step_ids: set
    ) -> List[Dict[str, Any]]:
        """Validate that all constraints reference at least one valid step ID.
        
        This enforces the P3 validation rule: constraints must attach to pre-existing steps.
        """
        validated: List[Dict[str, Any]] = []

        for constraint in constraints:
            attached_to = constraint.get("attached_to", [])
            if not attached_to:
                logger.debug(
                    "Filtering constraint without attachment: %s",
                    constraint.get("expression", "unknown")
                )
                continue

            if not isinstance(attached_to, list):
                attached_to = [attached_to]

            valid_attachments = [step_id for step_id in attached_to if step_id in valid_step_ids]
            if not valid_attachments:
                logger.debug(
                    "Filtering constraint with invalid step IDs %s: %s",
                    attached_to,
                    constraint.get("expression", "unknown")
                )
                continue

            constraint["attached_to"] = valid_attachments
            validated.append(constraint)

        return validated


__all__ = ["TwoStageSchemaStrategy"]
