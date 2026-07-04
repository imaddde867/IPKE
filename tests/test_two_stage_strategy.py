"""Tests for the P3 Two-Stage prompt strategy."""

import json
from unittest.mock import Mock, call

import pytest

from src.ai.prompting.two_stage import TwoStageSchemaStrategy
from src.core.unified_config import UnifiedConfig


class TestTwoStageStrategy:
    """Test the P3 Two-Stage strategy with skeleton and enrichment phases."""

    @pytest.fixture
    def config(self):
        cfg = UnifiedConfig.from_environment()
        cfg.prompting_strategy = "P3"
        return cfg

    @pytest.fixture
    def strategy(self, config):
        return TwoStageSchemaStrategy(config)

    def test_p3_strategy_name(self):
        """Verify P3 strategy has correct name."""
        config = UnifiedConfig.from_environment()
        strategy = TwoStageSchemaStrategy(config)
        assert strategy.name == "P3"

    def test_p3_stage1_template_focuses_on_steps(self, strategy):
        """Verify Stage 1 template focuses only on step extraction."""
        template = strategy.stage1_template
        assert "ONLY the ordered procedural steps" in template
        assert "Ignore constraints" in template
        assert "id" in template.lower()
        assert "order" in template.lower()

    def test_p3_stage2_template_enforces_step_id_validation(self, strategy):
        """Verify Stage 2 template enforces constraint-to-step linkage."""
        template = strategy.stage2_template
        assert "CRITICAL RULE" in template
        assert "MUST explicitly reference" in template
        assert "attached_to" in template
        assert "step id" in template.lower()

    def test_p3_stage2_template_mentions_locked_taxonomy(self, strategy):
        """Verify Stage 2 prompt requests the locked IPKE-Bench constraint schema."""
        template = strategy.stage2_template
        for expected in [
            "precondition",
            "postcondition",
            "guard",
            "parameter",
            "role_assignment",
            "reference",
            "enforcement",
            "must",
            "should",
            "may",
        ]:
            assert expected in template

    def test_p3_extracts_steps_in_stage1(self, strategy):
        """Verify P3 can extract steps in Stage 1."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [
                {"id": "S1", "text": "Inspect valve", "order": 1, "confidence": 0.92},
                {"id": "S2", "text": "Close valve", "order": 2, "confidence": 0.90},
            ],
            "constraints": [],
            "entities": []
        })

        result = strategy.run(mock_backend, "Inspect valve. Close valve.", "manual")

        assert len(result.steps) == 2
        assert result.steps[0]["id"] == "S1"
        assert result.steps[0]["text"] == "Inspect valve"
        assert result.steps[0]["order"] == 1
        assert result.steps[1]["id"] == "S2"

    def test_p3_two_stage_backend_calls(self, strategy):
        """Verify P3 makes exactly two backend calls."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [],
                "entities": []
            })
        ]

        strategy.run(mock_backend, "text", "manual")

        assert mock_backend.generate.call_count == 2

    def test_p3_passes_steps_to_stage2(self, strategy):
        """Verify Stage 2 receives extracted steps from Stage 1."""
        mock_backend = Mock()
        stage1_response = json.dumps({
            "steps": [
                {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                {"id": "S2", "text": "Action 2", "order": 2, "confidence": 0.88},
            ],
            "constraints": [],
            "entities": []
        })
        stage2_response = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": []
        })
        mock_backend.generate.side_effect = [stage1_response, stage2_response]

        strategy.run(mock_backend, "text", "manual")

        stage2_prompt = mock_backend.generate.call_args_list[1][0][0]
        assert "S1" in stage2_prompt
        assert "S2" in stage2_prompt
        assert "Action 1" in stage2_prompt

    def test_p3_validates_constraints_reference_valid_steps(self, strategy):
        """Verify P3 validates that constraints reference step IDs from Stage 1."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                    {"id": "S2", "text": "Action 2", "order": 2, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard for S1",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    },
                    {
                        "id": "C2",
                        "expression": "Guard for S2",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S2"],
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) == 2
        assert len(result.constraints) == 2
        assert result.constraints[0]["attached_to"] == ["S1"]
        assert result.constraints[1]["attached_to"] == ["S2"]

    def test_p3_filters_constraints_with_invalid_step_ids(self, strategy):
        """Verify P3 filters out constraints referencing non-existent step IDs."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Valid constraint",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    },
                    {
                        "id": "C2",
                        "expression": "Invalid constraint",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S99"],
                        "confidence": 0.85
                    },
                    {
                        "id": "C3",
                        "expression": "Another invalid",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S2", "S3"],
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert result.constraints[0]["id"] == "C1"
        assert result.constraints[0]["expression"] == "Valid constraint"

    def test_p3_filters_constraints_without_attachment(self, strategy):
        """Verify P3 filters constraints without attached_to field."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    },
                    {
                        "id": "C2",
                        "expression": "No attachment",
                        "type": "guard",
                        "enforcement": "must",
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert result.constraints[0]["id"] == "C1"

    def test_p3_filters_constraints_with_invalid_type(self, strategy):
        """Verify P3 filters constraints outside the locked benchmark taxonomy."""
        constraints = [
            {
                "id": "C1",
                "expression": "Wear gloves",
                "type": "warning",
                "enforcement": "must",
                "attached_to": ["S1"],
            }
        ]

        assert strategy._validate_constraints(constraints, {"S1"}) == []

    def test_p3_filters_constraints_without_enforcement(self, strategy):
        """Verify P3 filters constraints missing locked enforcement metadata."""
        constraints = [
            {
                "id": "C1",
                "expression": "Wear gloves",
                "type": "guard",
                "attached_to": ["S1"],
            }
        ]

        assert strategy._validate_constraints(constraints, {"S1"}) == []

    def test_p3_handles_empty_step_extraction(self, strategy):
        """Verify P3 handles gracefully when Stage 1 extracts no steps."""
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "steps": [],
            "constraints": [],
            "entities": []
        })

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) == 0
        assert len(result.constraints) == 0
        assert mock_backend.generate.call_count == 1

    def test_p3_uses_stage1_steps_if_stage2_empty(self, strategy):
        """Verify P3 preserves Stage 1 steps even if Stage 2 returns empty steps."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                    {"id": "S2", "text": "Action 2", "order": 2, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    }
                ],
                "entities": [
                    {"content": "Tool", "type": "tool", "category": "tool", "confidence": 0.9}
                ]
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) == 2
        assert result.steps[0]["id"] == "S1"
        assert result.steps[1]["id"] == "S2"
        assert len(result.constraints) == 1
        assert len(result.entities) == 1

    def test_p3_extracts_entities(self, strategy):
        """Verify P3 can extract entities in Stage 2."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Apply treatment", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [],
                "entities": [
                    {"content": "Chemical X", "type": "material", "category": "chemical", "confidence": 0.95},
                    {"content": "50°C", "type": "specification", "category": "temperature", "confidence": 0.92},
                ]
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.entities) == 2
        assert result.entities[0].content == "Chemical X"
        assert result.entities[1].content == "50°C"

    def test_p3_handles_multiple_constraints_per_step(self, strategy):
        """Verify P3 can attach multiple constraints to a single step."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Heat mixture", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Before heating",
                        "type": "precondition",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.88
                    },
                    {
                        "id": "C2",
                        "expression": "Temp < 100°C",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.90
                    },
                    {
                        "id": "C3",
                        "expression": "Stir continuously",
                        "type": "parameter",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 3
        for constraint in result.constraints:
            assert constraint["attached_to"] == ["S1"]

    def test_p3_handles_constraint_attached_to_multiple_steps(self, strategy):
        """Verify P3 can attach a constraint to multiple valid steps."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                    {"id": "S2", "text": "Action 2", "order": 2, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Daily maintenance",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1", "S2"],
                        "confidence": 0.90
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert set(result.constraints[0]["attached_to"]) == {"S1", "S2"}

    def test_p3_filters_mixed_valid_invalid_attachments(self, strategy):
        """Verify P3 filters partially valid attachments (keep valid, remove invalid)."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Mixed attachment",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1", "S99", "S2"],
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert result.constraints[0]["attached_to"] == ["S1"]

    def test_p3_handles_non_list_attachment(self, strategy):
        """Verify P3 handles non-list attached_to values."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": "S1",
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert isinstance(result.constraints[0]["attached_to"], list)

    def test_p3_respects_stop_tokens(self, strategy):
        """Verify P3 passes correct stop tokens to backend."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [],
                "entities": []
            })
        ]

        strategy.run(mock_backend, "text", "manual")

        calls = mock_backend.generate.call_args_list
        assert len(calls) == 2
        for call_obj in calls:
            kwargs = call_obj[1]
            assert "stop" in kwargs
            assert "</s>" in kwargs["stop"]
            assert "[/INST]" in kwargs["stop"]

    def test_p3_filters_simple_constraints_missing_schema_metadata(self, strategy):
        """Verify P3 filters constraints that omit locked schema metadata."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "text": "Guard",
                        "attached_to": ["S1"],
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 0

    def test_p3_stage1_template_format_compatible(self, strategy):
        """Verify Stage 1 template includes proper placeholders."""
        template = strategy.stage1_template
        assert "{document_type}" in template
        assert "{chunk}" in template
        assert "S1" in template
        assert "confidence" in template

    def test_p3_stage2_template_requires_steps_json(self, strategy):
        """Verify Stage 2 template includes placeholder for steps_json."""
        template = strategy.stage2_template
        assert "{steps_json}" in template

    def test_p3_extracts_constraint_types(self, strategy):
        """Verify P3 preserves constraint types."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action", "order": 1, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    },
                    {
                        "id": "C2",
                        "expression": "Temporal",
                        "type": "temporal",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.87
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.constraints) == 1
        assert result.constraints[0]["type"] == "guard"

    def test_p3_empty_steps_in_stage2_response(self, strategy):
        """Verify P3 correctly ignores empty steps from Stage 2."""
        mock_backend = Mock()
        mock_backend.generate.side_effect = [
            json.dumps({
                "steps": [
                    {"id": "S1", "text": "Action 1", "order": 1, "confidence": 0.9},
                    {"id": "S2", "text": "Action 2", "order": 2, "confidence": 0.9},
                ],
                "constraints": [],
                "entities": []
            }),
            json.dumps({
                "steps": [],
                "constraints": [
                    {
                        "id": "C1",
                        "expression": "Guard",
                        "type": "guard",
                        "enforcement": "must",
                        "attached_to": ["S1"],
                        "confidence": 0.85
                    }
                ],
                "entities": []
            })
        ]

        result = strategy.run(mock_backend, "text", "manual")

        assert len(result.steps) == 2
        assert result.steps[0]["text"] == "Action 1"
        assert result.steps[1]["text"] == "Action 2"
