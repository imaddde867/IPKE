"""Prompt strategy factory and public exports."""

from __future__ import annotations

from typing import Any

from src.ai.prompting.base import LLMBackend, PromptStrategy
from src.ai.prompting.chain_of_thought import CoTPromptStrategy
from src.ai.prompting.few_shot import FewShotPromptStrategy
from src.ai.prompting.two_stage import TwoStageSchemaStrategy
from src.ai.prompting.zero_shot import ZeroShotJSONStrategy


def build_prompt_strategy(config: Any) -> PromptStrategy:
    """Return the configured prompt strategy implementation."""

    strategy = getattr(config, "prompting_strategy", "P0").upper()
    if strategy == "P1":
        return FewShotPromptStrategy(config)
    if strategy == "P2":
        return CoTPromptStrategy(config)
    if strategy == "P3":
        return TwoStageSchemaStrategy(config)
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
