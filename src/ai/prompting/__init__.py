"""Prompt strategy factory and public exports."""

from __future__ import annotations

from typing import Any

from src.ai.prompting.base import LLMBackend, PromptStrategy
from src.ai.prompting.chain_of_thought import CoTPromptStrategy
from src.ai.prompting.few_shot import FewShotPromptStrategy
from src.ai.prompting.two_stage import TwoStageSchemaStrategy
from src.ai.prompting.zero_shot import ZeroShotJSONStrategy


def build_prompt_strategy(config: Any) -> PromptStrategy:
    """Return the configured prompt strategy implementation.
    
    Strategy mapping (as per STRATEGIES.md):
    - P0: ZeroShotJSONStrategy (baseline, zero-shot)
    - P1: FewShotPromptStrategy (few-shot with examples)
    - P2: CoTPromptStrategy (chain-of-thought reasoning)
    - P3: TwoStageSchemaStrategy (two-stage schema extraction)
    """
    strategy = getattr(config, "prompting_strategy", "P0").upper()
    if strategy == "P1":
        return FewShotPromptStrategy(config)
    elif strategy == "P2":
        return CoTPromptStrategy(config)
    elif strategy == "P3":
        return TwoStageSchemaStrategy(config)
    # P0 or default
    return ZeroShotJSONStrategy(config)


__all__ = [
    "LLMBackend",
    "PromptStrategy",
    "ZeroShotJSONStrategy",
    "FewShotPromptStrategy",
    "CoTPromptStrategy",
    "TwoStageSchemaStrategy",
    "build_prompt_strategy",
]
