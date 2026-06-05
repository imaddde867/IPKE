# Adding New Prompt Strategies

The prompting module is modular and extensible. To add a new prompt strategy:

## 1. Create a new strategy file

Create a new Python file in this directory (e.g., `my_strategy.py`):

```python
from src.ai.prompting.base import PromptStrategy, _escape_braces
from src.ai.types import ChunkExtraction

class MyPromptStrategy(PromptStrategy):
    """Description of your strategy."""
    
    name = "CUSTOM"  # Identifier used in config
    
    def run(self, backend, chunk: str, document_type: str) -> ChunkExtraction:
        # Your implementation here
        prompt = f"Your prompt template with {chunk} and {document_type}"
        response = backend.generate(prompt, stop=["</s>", "[/INST]"])
        return self._parse_json(response, chunk)
```

## 2. Update `__init__.py`

Add the import and update the factory function:

```python
from src.ai.prompting.my_strategy import MyPromptStrategy

def build_prompt_strategy(config):
    strategy = getattr(config, "prompting_strategy", "P0").upper()
    # ... existing conditions ...
    if strategy == "CUSTOM":
        return MyPromptStrategy(config)
    # ... rest of function ...
```

## 3. Update config

Set `prompting_strategy=CUSTOM` in your `.env` or config to use your new strategy.

## Available Utilities

The `base.py` module provides reusable utilities:

- `_escape_braces(text)` - Escape braces for format strings
- `_coerce_confidence(value)` - Normalize confidence scores to 0-1
- `_ensure_dict_list(value)` - Safely convert to list of dicts
- `_build_entities(items, chunk)` - Build ExtractedEntity objects
- `_extract_json_payload(text)` - Extract JSON from LLM responses

## Current Strategies

- **P0** (ZeroShotJSONStrategy) - Single-turn prompting without examples
- **P1** (FewShotPromptStrategy) - Few-shot learning with examples
- **P2** (CoTPromptStrategy) - Chain-of-Thought reasoning
- **P3** (TwoStageSchemaStrategy) - Two-stage extraction pipeline
