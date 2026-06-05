"""Tests for the enhanced P1 few-shot prompt strategy."""

import json
from unittest.mock import Mock

import pytest

from src.ai.prompting.few_shot import FewShotPromptStrategy
from src.core.unified_config import UnifiedConfig


class TestFewShotEnhanced:
    """Test the P1 few-shot strategy with constraint types and parameters."""

    @pytest.fixture
    def config(self):
        cfg = UnifiedConfig.from_environment()
        cfg.prompting_strategy = "P1"
        return cfg

    @pytest.fixture
    def strategy(self, config):
        return FewShotPromptStrategy(config)

    def test_p1_extracts_constraint_types(self, strategy):
        """Verify P1 can extract constraints with type field."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
            ],
            "constraints": [
                {"id": "C1", "expression": "Guard condition", "type": "guard", "attached_to": ["S1"], "confidence": 0.85},
                {"id": "C2", "expression": "Temporal constraint", "type": "temporal", "attached_to": ["S1"], "confidence": 0.87},
            ],
            "entities": [
                {"content": "Tool X", "type": "tool", "category": "tool", "confidence": 0.9},
            ]
        })

        result = strategy.run(mock_backend, "Some text", "manual")

        assert len(result.steps) == 1
        assert result.steps[0]["id"] == "S1"
        assert "order" in result.steps[0]

        assert len(result.constraints) == 2
        assert result.constraints[0]["type"] == "guard"
        assert result.constraints[0]["attached_to"] == ["S1"]
        assert result.constraints[1]["type"] == "temporal"

        assert len(result.entities) == 1
        assert result.entities[0].content == "Tool X"

    def test_p1_extracts_parameters(self, strategy):
        """Verify P1 preserves parameters in constraints."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Fill cavity", "order": 1, "confidence": 0.92},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "Fill 85% of cavity",
                    "type": "parameter",
                    "attached_to": ["S1"],
                    "confidence": 0.89,
                    "parameters": {"fill_level": {"value": 85, "unit": "%"}}
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert constraint["type"] == "parameter"
        assert "parameters" in constraint
        assert constraint["parameters"]["fill_level"]["value"] == 85
        assert constraint["parameters"]["fill_level"]["unit"] == "%"

    def test_p1_extracts_schedule_type(self, strategy):
        """Verify P1 can extract schedule constraints with frequency data."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Inspect", "order": 1, "confidence": 0.91},
                {"id": "S2", "text": "Clean", "order": 2, "confidence": 0.89},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "Daily",
                    "type": "schedule",
                    "attached_to": ["S1", "S2"],
                    "confidence": 0.90,
                    "parameters": {"frequency": {"value": 1, "unit": "day"}}
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert constraint["type"] == "schedule"
        assert constraint["attached_to"] == ["S1", "S2"]
        assert constraint["parameters"]["frequency"]["value"] == 1
        assert constraint["parameters"]["frequency"]["unit"] == "day"

    def test_p1_extracts_environmental_guard(self, strategy):
        """Verify P1 can extract environmental guard constraints."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Operation", "order": 1, "confidence": 0.91},
            ],
            "constraints": [
                {
                    "id": "C1",
                    "expression": "When dusty or fire-hazard",
                    "type": "environmental_guard",
                    "attached_to": ["S1"],
                    "confidence": 0.87,
                    "parameters": {"conditions": ["dusty", "fire_hazard"]}
                }
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert constraint["type"] == "environmental_guard"
        assert constraint["parameters"]["conditions"] == ["dusty", "fire_hazard"]

    def test_p1_maintains_entity_confidence(self, strategy):
        """Verify P1 maintains entity confidence scores."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": [
                {"content": "Material A", "type": "material", "category": "material", "confidence": 0.95},
                {"content": "Tool B", "type": "tool", "category": "tool", "confidence": 0.88},
            ]
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.entities) == 2
        assert result.entities[0].confidence == 0.95
        assert result.entities[1].confidence == 0.88

    def test_p1_handles_missing_parameters(self, strategy):
        """Verify P1 gracefully handles constraints without parameters."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
            ],
            "constraints": [
                {"id": "C1", "expression": "Guard", "type": "guard", "attached_to": ["S1"], "confidence": 0.85}
            ],
            "entities": []
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert constraint["type"] == "guard"

    def test_p1_handles_mixed_constraint_types(self, strategy):
        """Verify P1 can handle a mix of different constraint types in one extraction."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Mix and apply", "order": 1, "confidence": 0.92},
                {"id": "S2", "text": "Feather", "order": 2, "confidence": 0.90},
            ],
            "constraints": [
                {"id": "C1", "expression": "If needed", "type": "guard", "attached_to": ["S2"], "confidence": 0.85},
                {
                    "id": "C2",
                    "expression": "85% fill",
                    "type": "parameter",
                    "attached_to": ["S1"],
                    "confidence": 0.89,
                    "parameters": {"fill_level": {"value": 85, "unit": "%"}}
                },
                {
                    "id": "C3",
                    "expression": "After 30 minutes",
                    "type": "temporal",
                    "attached_to": ["S2"],
                    "confidence": 0.87,
                    "parameters": {"wait_time": {"value": 30, "unit": "minutes"}}
                }
            ],
            "entities": [
                {"content": "Filler", "type": "material", "category": "material", "confidence": 0.95},
            ]
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) == 2
        assert len(result.constraints) == 3
        assert result.constraints[0]["type"] == "guard"
        assert result.constraints[1]["type"] == "parameter"
        assert result.constraints[2]["type"] == "temporal"

    def test_p1_extracts_examples_from_template(self):
        """Verify the P1 examples are properly embedded in the template."""
        config = UnifiedConfig.from_environment()
        config.prompting_strategy = "P1"
        strategy = FewShotPromptStrategy(config)

        template = strategy.template

        assert "Example 1" in template
        assert "Example 2" in template
        assert "3M™ Marine Blister Repair Filler" in template
        assert "engine compartment" in template
        assert "constraint types" in template or "type" in template

    def test_p1_shows_guidance_on_constraint_types(self):
        """Verify the P1 template includes guidance on constraint types."""
        config = UnifiedConfig.from_environment()
        config.prompting_strategy = "P1"
        strategy = FewShotPromptStrategy(config)

        template = strategy.template

        constraint_types = ["guard", "parameter", "temporal", "schedule", "environmental_guard"]
        for ctype in constraint_types:
            assert ctype in template.lower(), f"Template should mention constraint type '{ctype}'"

    def test_p1_strategy_name(self):
        """Verify P1 strategy has correct name."""
        config = UnifiedConfig.from_environment()
        strategy = FewShotPromptStrategy(config)
        assert strategy.name == "P1"

    def test_p1_backward_compatible_with_old_format(self, strategy):
        """Verify P1 handles old format (without type and parameters) gracefully."""
        mock_backend = Mock()
        old_format_response = json.dumps({
            "steps": [
                {"id": "S1", "text": "Action", "confidence": 0.9}
            ],
            "constraints": [
                {"text": "Some constraint", "steps": ["S1"], "confidence": 0.85}
            ],
            "entities": [
                {"content": "Material", "type": "material", "category": "material", "confidence": 0.9}
            ]
        })
        mock_backend.generate.return_value = old_format_response

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) >= 1
        assert len(result.constraints) >= 1
        assert len(result.entities) >= 1
