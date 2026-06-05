# Prompting Strategies

This module provides modular, extensible prompting strategies for procedural knowledge extraction. Each strategy can be selected via the `prompting_strategy` configuration parameter.

## Available Strategies

### P0: Zero-Shot JSON (ZeroShotJSONStrategy)

**Approach**: Single direct prompt without examples

**Characteristics**:
- Fastest inference
- Good for simple, well-structured procedures
- Baseline for comparison
- Minimal context requirements

**Use when**: Processing straightforward instructions with clear action sequences

**File**: `zero_shot.py`

---

### P1: Few-Shot In-Context Learning (FewShotPromptStrategy)

**Approach**: In-context learning with 2 annotated examples (Brown et al., 2020)

**Key Features**:
- **Constraint Types**: Distinguishes between guard, parameter, temporal, schedule, and environmental_guard constraints
- **Typed Constraints**: Each constraint includes a `type` field for classification
- **Structured Parameters**: Constraints can include a `parameters` dict with machine-readable data
- **Two Examples**: Demonstrate proper extraction of:
  - Marine repair procedure with parameter and temporal constraints
  - Maintenance procedure with schedule and environmental guard constraints

**Constraint Types**:
- `guard`: Conditional prerequisites ("If X", "Only if Y")
- `parameter`: Quantitative limits or specifications ("85% fill", "grade 80+")
- `temporal`: Timing or duration constraints ("After 30 min", "Wait until dry")
- `schedule`: Frequency or recurrence ("Daily", "Weekly")
- `environmental_guard`: Contextual conditions ("When dusty", "In high temperature")

**Use when**: 
- Constraint classification is important
- You want to improve accuracy through demonstrations
- Complex procedures with multiple constraint types

**File**: `few_shot.py`

---

### P2: Chain-of-Thought Reasoning (CoTPromptStrategy)

**Approach**: Explicit reasoning decomposition (Wei et al., 2022)

**Four Reasoning Stages**:

1. **CLASSIFY**: Distinguish between Descriptive (facts, definitions) and Procedural (instructions, rules)
   - Returns early for descriptive content (reduces false positives)
   - Identifies human actors (Operator, Technician, Engineer, etc.)

2. **FILTER**: Extract actionable human steps
   - Excludes non-human actions ("Beetle eats grain")
   - Excludes passive states ("Pump is running")
   - Excludes general observations ("Fire is dangerous")

3. **ATTACH**: Link constraints to steps
   - Preconditions: "before", "after", "ensure", "first", "prior to"
   - Safety guards: "must", "must not", "never", "if", "only if"
   - Quantitative specs: temperatures, pressures, distances, durations
   - Environmental conditions: "when", "during", "in case of"

4. **EMIT**: Generate structured JSON with confidence scores

**Use when**:
- Reducing hallucination is critical
- Distinguishing steps from descriptions matters
- Linking constraints accurately is important
- Safety-critical procedures

**File**: `chain_of_thought.py`

---

### P3: Two-Stage Schema (TwoStageSchemaStrategy)

**Approach**: Sequential extraction with schema enrichment

**Process**:
1. **Stage 1**: Extract steps only from procedural content
2. **Stage 2**: Enrich steps with constraints and entities using the extracted steps as context

**Use when**: 
- Simple extraction can fail without context
- Step extraction needs independent verification
- Multi-stage reasoning helps accuracy

**File**: `two_stage.py`

---

## Configuration

Set the strategy via environment variable or config object:

```python
from src.core.unified_config import UnifiedConfig

config = UnifiedConfig.from_environment()
config.prompting_strategy = "P1"  # or "P0", "P2", "P3"
```

Or in `.env`:
```
PROMPTING_STRATEGY=P2
```

## Strategy Factory

Use the factory function to instantiate strategies programmatically:

```python
from src.ai.prompting import build_prompt_strategy

config = UnifiedConfig.from_environment()
strategy = build_prompt_strategy(config)
result = strategy.run(backend, text, document_type)
```

The factory maps strategy names:
- `P0` → ZeroShotJSONStrategy
- `P1` → FewShotPromptStrategy
- `P2` → CoTPromptStrategy
- `P3` → TwoStageSchemaStrategy

## Output Schema

All strategies return `ChunkExtraction` with:

```python
@dataclass
class ChunkExtraction:
    entities: List[ExtractedEntity]      # Materials, tools, components
    steps: List[Dict[str, Any]]          # Procedural steps with order
    constraints: List[Dict[str, Any]]    # Guards, preconditions, specs
```

### Constraint Structure

**Enhanced format (P1, P2, P3)**:
```json
{
  "id": "C1",
  "expression": "Fill 85% of cavity",
  "type": "parameter",
  "attached_to": ["S1", "S2"],
  "confidence": 0.89,
  "parameters": {
    "fill_level": {"value": 85, "unit": "%"}
  }
}
```

**Legacy format (P0)**:
```json
{
  "id": "C1",
  "text": "Guard condition",
  "steps": ["S1"],
  "confidence": 0.85
}
```

## Comparison

| Feature | P0 | P1 | P2 | P3 |
|---------|----|----|----|----|
| Speed | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡ |
| Accuracy | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Constraint Types | No | ✓ | ✓ | Partial |
| Reasoning | No | Examples | Explicit | Implicit |
| False Positives | High | Medium | Low | Low |
| Descriptive Filter | No | No | ✓ | No |

## Extending

To add a new strategy:

1. Create a file in this directory (e.g., `custom.py`)
2. Implement the `PromptStrategy` base class
3. Register in `__init__.py` factory function

See `EXTENDING.md` for detailed instructions.
