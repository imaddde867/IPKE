"""Tests for the unified knowledge engine orchestration."""

import pytest

from src.ai.knowledge_engine import UnifiedKnowledgeEngine
from src.ai.types import ChunkExtraction, ExtractedEntity


def make_chunk(step_text: str, constraint_text: str) -> ChunkExtraction:
    return ChunkExtraction(
        entities=[ExtractedEntity(content="entity", entity_type="type", category="cat", confidence=0.9)],
        steps=[{"id": "legacy", "text": step_text, "confidence": 0.8}],
        constraints=[{"id": "c0", "text": constraint_text, "steps": ["legacy"], "confidence": 0.7}],
    )


@pytest.mark.asyncio
async def test_engine_normalizes_steps(monkeypatch):
    engine = UnifiedKnowledgeEngine()

    async def fake_run_chunks(self, tasks):
        assert len(tasks) == 1
        return [make_chunk("Inspect manifold", "Wear PPE")]

    monkeypatch.setattr(UnifiedKnowledgeEngine, "_run_chunks", fake_run_chunks)
    result = await engine.extract_knowledge("Inspect manifold before work.")
    assert result.steps[0]["id"] == "S1"
    assert result.constraints[0]["steps"] == ["S1"]
    assert result.entities


@pytest.mark.asyncio
async def test_engine_cache_hits(monkeypatch):
    engine = UnifiedKnowledgeEngine()
    calls = {"count": 0}

    async def fake_run_chunks(self, tasks):
        calls["count"] += 1
        return [make_chunk("Step alpha", "Use gloves")]

    monkeypatch.setattr(UnifiedKnowledgeEngine, "_run_chunks", fake_run_chunks)
    await engine.extract_knowledge("Alpha procedure text.")
    await engine.extract_knowledge("Alpha procedure text.")  # cache hit
    assert calls["count"] == 1
    stats = engine.get_performance_stats()
    assert stats["cache_hits"] == 1


def test_clear_cache_shuts_down_pool():
    engine = UnifiedKnowledgeEngine()

    class DummyPool:
        def __init__(self):
            self.closed = False

        def shutdown(self):
            self.closed = True

    dummy = DummyPool()
    engine._pool = dummy  # type: ignore[attr-defined]
    engine.cache["key"] = None  # type: ignore[index]
    engine.clear_cache()
    assert engine._pool is None
    assert not engine.cache
    assert dummy.closed
