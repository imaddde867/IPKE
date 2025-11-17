"""Prompt strategy selection tests."""

from src.ai.prompting import (
    CoTPromptStrategy,
    FewShotPromptStrategy,
    TwoStageSchemaStrategy,
    ZeroShotJSONStrategy,
    build_prompt_strategy,
)
from src.core.unified_config import UnifiedConfig


def test_build_prompt_strategy_variants():
    cfg = UnifiedConfig.from_environment()
    cfg.prompting_strategy = "P0"
    assert isinstance(build_prompt_strategy(cfg), ZeroShotJSONStrategy)
    cfg.prompting_strategy = "P1"
    assert isinstance(build_prompt_strategy(cfg), FewShotPromptStrategy)
    cfg.prompting_strategy = "P2"
    assert isinstance(build_prompt_strategy(cfg), CoTPromptStrategy)
    cfg.prompting_strategy = "P3"
    assert isinstance(build_prompt_strategy(cfg), TwoStageSchemaStrategy)
