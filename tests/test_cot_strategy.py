"""Tests for the enhanced P2 Chain-of-Thought strategy with reasoning decomposition."""

import json
from unittest.mock import Mock

import pytest

from src.ai.prompting.chain_of_thought import CoTPromptStrategy
from src.core.unified_config import UnifiedConfig


class TestChainOfThoughtEnhanced:
    """Test the P2 CoT strategy with explicit reasoning decomposition."""

    @pytest.fixture
    def config(self):
        cfg = UnifiedConfig.from_environment()
        cfg.prompting_strategy = "P2"
        return cfg

    @pytest.fixture
    def strategy(self, config):
        return CoTPromptStrategy(config)

    def test_p2_strategy_name(self):
        """Verify P2 strategy has correct name."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        assert strategy.name == "P2"

    def test_p2_template_includes_reasoning_steps(self):
        """Verify template includes CLASSIFY, FILTER, ATTACH, EMIT steps."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        reasoning_stages = ["CLASSIFY", "FILTER", "ATTACH", "EMIT"]
        for stage in reasoning_stages:
            assert stage in template, f"Template must include '{stage}' reasoning stage"

    def test_p2_template_distinguishes_descriptive_procedural(self):
        """Verify template mentions classification between Descriptive and Procedural."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        assert "Descriptive" in template
        assert "Procedural" in template
        assert "NO STEPS TO EXTRACT" in template

    def test_p2_template_filters_human_actions(self):
        """Verify template emphasizes filtering non-human actions."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        assert "human" in template.lower()
        assert "non-human" in template.lower()

    def test_p2_template_mentions_preconditions(self):
        """Verify template guides on linking preconditions."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        precondition_keywords = ["before", "after", "ensure", "prior to"]
        for keyword in precondition_keywords:
            assert keyword in template.lower()

    def test_p2_template_mentions_safety_guards(self):
        """Verify template guides on safety guards."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        safety_keywords = ["must", "never", "if", "only if"]
        for keyword in safety_keywords:
            assert keyword in template.lower()

    def test_p2_extracts_steps_from_procedural_text(self, strategy):
        """Verify P2 can extract steps from procedural content."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Inspect the valve", "order": 1, "confidence": 0.92},
                {"id": "S2", "text": "Close the supply", "order": 2, "confidence": 0.90},
            ],
            "constraints": [],
            "entities": []
        })

        result = strategy.run(mock_backend, "Inspect valve. Close supply.", "manual")

        assert len(result.steps) == 2
        assert result.steps[0]["text"] == "Inspect the valve"
        assert result.steps[1]["text"] == "Close the supply"
        assert result.steps[0]["order"] == 1
        assert result.steps[1]["order"] == 2

    def test_p2_filters_non_human_actions(self, strategy):
        """Verify P2 reasoning guides filtering of non-human actions."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Apply pressure", "order": 1, "confidence": 0.91},
            ],
            "constraints": [],
            "entities": []
        })

        result = strategy.run(mock_backend, "Apply pressure. Water evaporates.", "manual")

        assert len(result.steps) == 1
        assert "Apply pressure" in result.steps[0]["text"]

    def test_p2_attaches_preconditions(self, strategy):
        """Verify P2 can attach preconditions to steps."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Inspect valve", "order": 1, "confidence": 0.91},
                {"id": "S2", "text": "Close valve", "order": 2, "confidence": 0.89},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "Ensure pressure is stable before proceeding",
                    "type": "precondition",
                    "attached_to": ["S1"],
                    "confidence": 0.88
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "Ensure pressure stable. Inspect valve. Close.", "manual")

        assert len(result.constraints) == 1
        assert result.constraints[0]["type"] == "precondition"
        assert "S1" in result.constraints[0]["attached_to"]

    def test_p2_attaches_safety_guards(self, strategy):
        """Verify P2 can attach safety guards to steps."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Apply treatment", "order": 1, "confidence": 0.90},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "Must wear protective equipment",
                    "type": "safety_guard",
                    "attached_to": ["S1"],
                    "confidence": 0.92
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "Must wear protection. Apply treatment.", "manual")

        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert constraint["type"] == "safety_guard"
        assert "S1" in constraint["attached_to"]
        assert "protective" in constraint.get("expression", "").lower()

    def test_p2_handles_multiple_constraints_per_step(self, strategy):
        """Verify P2 can attach multiple constraints to one step."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Heat mixture", "order": 1, "confidence": 0.90},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "Before heating",
                    "type": "precondition",
                    "attached_to": ["S1"],
                    "confidence": 0.88
                },
                {
                    "id": "C2",
                    "expression": "Temperature must not exceed 100°C",
                    "type": "safety_guard",
                    "attached_to": ["S1"],
                    "confidence": 0.90
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "Before heating, ensure setup. Heat to <100°C.", "manual")

        assert len(result.constraints) == 2
        attached_to_s1 = [c for c in result.constraints if "S1" in c.get("attached_to", [])]
        assert len(attached_to_s1) == 2

    def test_p2_identifies_human_actors(self):
        """Verify template mentions human actor identification."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        actor_keywords = ["Operator", "Technician", "Engineer", "Inspector", "HUMAN ACTOR"]
        template_text = template.lower()
        found_actor_mention = any(keyword.lower() in template_text for keyword in actor_keywords)
        assert found_actor_mention, "Template should mention identifying human actors"

    def test_p2_extracts_entities(self, strategy):
        """Verify P2 can extract entities with types and categories."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": [
                {"content": "Valve A", "type": "equipment", "category": "component", "confidence": 0.95},
                {"content": "50 PSI", "type": "specification", "category": "pressure", "confidence": 0.92},
            ]
        })

        result = strategy.run(mock_backend, "Adjust Valve A to 50 PSI", "manual")

        assert len(result.entities) == 2
        assert result.entities[0].content == "Valve A"
        assert result.entities[1].content == "50 PSI"

    def test_p2_handles_descriptive_content(self, strategy):
        """Verify P2 handles descriptive (non-procedural) content gracefully."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": [
                {"content": "Historical fact", "type": "information", "category": "general", "confidence": 0.85},
            ]
        })

        result = strategy.run(mock_backend, "Beetles are insects that eat grain", "manual")

        assert len(result.steps) == 0

    def test_p2_json_extraction_from_reasoning(self, strategy):
        """Verify P2 correctly extracts JSON from response with reasoning."""
        mock_backend = Mock()
        response_with_reasoning = """
CLASSIFY: This is procedural text about equipment maintenance.
The human actor is the Operator.

FILTER: Steps identified:
- S1: Clean the filter
- S2: Replace cartridge

ATTACH: Safety guard found - "Must wear gloves"

EMIT:
<json>
{
  "steps": [
    {"id": "S1", "text": "Clean the filter", "order": 1, "confidence": 0.91},
    {"id": "S2", "text": "Replace cartridge", "order": 2, "confidence": 0.89}
  ],
  "constraints": [
    {"id": "C1", "expression": "Must wear gloves", "type": "safety_guard", "attached_to": ["S1", "S2"], "confidence": 0.92}
  ],
  "entities": [
    {"content": "filter", "type": "component", "category": "component", "confidence": 0.95},
    {"content": "cartridge", "type": "component", "category": "component", "confidence": 0.93}
  ]
}
</json>
"""
        mock_backend.generate.return_value = response_with_reasoning

        result = strategy.run(mock_backend, "Clean filter. Replace cartridge. Must wear gloves.", "manual")

        assert len(result.steps) == 2
        assert len(result.constraints) == 1
        assert len(result.entities) == 2

    def test_p2_backward_compatible_with_simple_json(self, strategy):
        """Verify P2 handles simpler JSON formats without all fields."""
        mock_backend = Mock()
        simple_json = json.dumps({
            "steps": [
                {"id": "S1", "text": "Do action", "confidence": 0.9}
            ],
            "constraints": [
                {"text": "Guard", "confidence": 0.85}
            ],
            "entities": []
        })
        mock_backend.generate.return_value = simple_json

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) >= 1
        assert len(result.constraints) >= 1

    def test_p2_handles_empty_json(self, strategy):
        """Verify P2 handles empty extraction gracefully."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": []
        })

        result = strategy.run(mock_backend, "No procedural content here", "manual")

        assert len(result.steps) == 0
        assert len(result.constraints) == 0
        assert len(result.entities) == 0

    def test_p2_mentions_quantitative_specs(self):
        """Verify template guides extraction of quantitative specifications."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        spec_keywords = ["temperature", "pressure", "distance", "duration", "equipment"]
        for keyword in spec_keywords:
            assert keyword in template.lower()

    def test_p2_mentions_environmental_conditions(self):
        """Verify template guides extraction of environmental conditions."""
        config = UnifiedConfig.from_environment()
        strategy = CoTPromptStrategy(config)
        template = strategy.template

        env_keywords = ["when", "during", "conditions"]
        for keyword in env_keywords:
            assert keyword in template.lower()
