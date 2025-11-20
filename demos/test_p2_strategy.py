#!/usr/bin/env python
"""
Demonstration of the enhanced P2 (Chain-of-Thought) prompt strategy.

This script shows how the P2 strategy uses explicit reasoning decomposition
to guide the LLM through four steps: CLASSIFY, FILTER, ATTACH, EMIT.
"""

import json
from unittest.mock import Mock

from src.ai.prompting import CoTPromptStrategy
from src.core.unified_config import UnifiedConfig


def demo_p2_strategy():
    """Demonstrate P2 strategy with reasoning decomposition."""
    print("=" * 80)
    print("P2 (Chain-of-Thought) Strategy Demonstration")
    print("=" * 80)
    print()

    config = UnifiedConfig.from_environment()
    config.prompting_strategy = "P2"
    strategy = CoTPromptStrategy(config)

    print(f"Strategy Name: {strategy.name}")
    print(f"Strategy Class: {strategy.__class__.__name__}")
    print()

    print("1. Reasoning Decomposition Steps in Template:")
    print("-" * 80)
    template = strategy.template
    reasoning_stages = ["CLASSIFY", "FILTER", "ATTACH", "EMIT"]
    for i, stage in enumerate(reasoning_stages, 1):
        if stage in template:
            print(f"  ✓ STEP {i}: {stage}")
        else:
            print(f"  ✗ STEP {i}: {stage}")
    print()

    print("2. Template Features:")
    print("-" * 80)
    features = {
        "Descriptive vs Procedural": "Descriptive" in template and "Procedural" in template,
        "Human actor identification": "HUMAN ACTOR" in template,
        "Non-human action filtering": "non-human" in template.lower(),
        "Precondition guidance": "before" in template.lower(),
        "Safety guard guidance": "must" in template.lower(),
        "Quantitative specs": "temperature" in template.lower() or "pressure" in template.lower(),
        "Environmental conditions": "when" in template.lower() or "during" in template.lower(),
    }

    for feature, present in features.items():
        symbol = "✓" if present else "✗"
        print(f"  {symbol} {feature:40} {present}")
    print()

    print("3. Test Extraction with Mock Backend:")
    print("-" * 80)

    mock_response = """
CLASSIFY: This is procedural maintenance text.
The human actor is the Operator/Technician.

FILTER: Identified human actions:
- Inspect the hydraulic manifold
- Close supply valve
- Tag and lock out upstream pumps

ATTACH: Safety guards and preconditions:
- "Follow site lockout procedures" applies to locking out pumps
- "Before closing valve" precondition

EMIT:
<json>
{
  "steps": [
    {"id": "S1", "text": "Inspect the hydraulic manifold", "order": 1, "confidence": 0.91},
    {"id": "S2", "text": "Close the supply valve", "order": 2, "confidence": 0.89},
    {"id": "S3", "text": "Tag and lock out upstream pumps", "order": 3, "confidence": 0.87}
  ],
  "constraints": [
    {"id": "C1", "expression": "Before closing valve", "type": "precondition", "attached_to": ["S2"], "confidence": 0.85},
    {"id": "C2", "expression": "Follow site lockout procedures", "type": "safety_guard", "attached_to": ["S3"], "confidence": 0.88}
  ],
  "entities": [
    {"content": "Hydraulic manifold", "type": "equipment", "category": "component", "confidence": 0.94},
    {"content": "Supply valve", "type": "equipment", "category": "component", "confidence": 0.92},
    {"content": "Pumps", "type": "equipment", "category": "component", "confidence": 0.89}
  ]
}
</json>
"""

    mock_backend = Mock()
    mock_backend.generate.return_value = mock_response

    test_text = "Inspect manifold. Before closing, check pressure. Close valve. Tag and lock pumps per site procedures."
    result = strategy.run(mock_backend, test_text, "maintenance_manual")

    print(f"Input text: '{test_text[:60]}...'")
    print()

    print("Extracted Steps (with ordering):")
    for step in result.steps:
        print(f"  {step['id']}: {step['text']}")
        print(f"      Order: {step.get('order', 'N/A')}, Confidence: {step.get('confidence', 'N/A'):.2f}")
    print()

    print("Extracted Constraints (preconditions + safety guards):")
    for constraint in result.constraints:
        constraint_type = constraint.get("type", "unknown")
        expression = constraint.get("expression", constraint.get("text", "N/A"))
        attached = constraint.get("attached_to", [])
        print(f"  {constraint['id']}: {expression} [{constraint_type}]")
        print(f"      Attached to: {attached}, Confidence: {constraint.get('confidence', 'N/A'):.2f}")
    print()

    print("Extracted Entities:")
    for entity in result.entities:
        print(f"  {entity.content} ({entity.category})")
        print(f"      Type: {entity.entity_type}, Confidence: {entity.confidence:.2f}")
    print()

    print("4. Reasoning Stages Explained:")
    print("-" * 80)
    stages_desc = {
        "CLASSIFY": "Distinguish descriptive vs procedural content; identify human actors",
        "FILTER": "Extract actionable steps; exclude non-human events and passive states",
        "ATTACH": "Link preconditions, safety guards, specs to steps",
        "EMIT": "Generate strict JSON with proper confidence scores",
    }

    for stage, description in stages_desc.items():
        print(f"  • {stage:10} → {description}")
    print()

    print("=" * 80)
    print("P2 Strategy Successfully Configured!")
    print("Key Advantages:")
    print("  ✓ Explicit reasoning reduces hallucination")
    print("  ✓ Filters non-human actions (fewer false positives)")
    print("  ✓ Properly attaches context to steps")
    print("  ✓ Distinguishes descriptive from procedural content")
    print("  ✓ Comprehensive guidance for complex extractions")
    print("=" * 80)


if __name__ == "__main__":
    demo_p2_strategy()
