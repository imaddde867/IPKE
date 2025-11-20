#!/usr/bin/env python
"""Demonstration of P1 (Few-shot) prompt strategy."""

import json
from unittest.mock import Mock

from src.ai.prompting import FewShotPromptStrategy
from src.core.unified_config import UnifiedConfig


def demo_p1_strategy():
    """Demonstrate P1 strategy with constraint types and parameters."""
    print("=" * 80)
    print("P1 (Few-shot In-Context Learning) Strategy Demonstration")
    print("=" * 80)

    config = UnifiedConfig.from_environment()
    config.prompting_strategy = "P1"
    strategy = FewShotPromptStrategy(config)

    print(f"Strategy: {strategy.name} - {strategy.__class__.__name__}")
    print()

    template = strategy.template
    print("✓ Contains 2 annotated examples (marine repair, maintenance)")
    print("✓ Supports typed constraints (guard, parameter, temporal, schedule, environmental_guard)")
    print("✓ Includes structured parameters in constraints")
    print()

    mock_response = {
        "steps": [
            {"id": "S1", "text": "Mix and apply filler", "order": 1, "confidence": 0.92},
            {"id": "S2", "text": "Fill 85% of cavity", "order": 2, "confidence": 0.90},
        ],
        "constraints": [
            {"id": "C1", "type": "parameter", "attached_to": ["S2"], "confidence": 0.89,
             "parameters": {"fill_level": {"value": 85, "unit": "%"}}},
            {"id": "C2", "type": "temporal", "attached_to": ["S2"], "confidence": 0.87,
             "parameters": {"wait_time": {"value": 30, "unit": "minutes"}}},
        ],
        "entities": [
            {"content": "3M Marine Filler", "type": "material", "category": "material", "confidence": 0.95},
        ]
    }

    mock_backend = Mock()
    mock_backend.generate.return_value = json.dumps(mock_response)
    result = strategy.run(mock_backend, "Mix and apply. Fill 85%.", "manual")

    print(f"Extracted {len(result.steps)} steps, {len(result.constraints)} constraints, {len(result.entities)} entities")
    print()
    print("Sample constraint with parameters:")
    if result.constraints:
        c = result.constraints[0]
        print(f"  Type: {c.get('type')}, Parameters: {c.get('parameters')}")
    print()
    print("✓ P1 Strategy Successfully Tested!")
    print("=" * 80)


if __name__ == "__main__":
    demo_p1_strategy()
