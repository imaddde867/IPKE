"""
Test knowledge extraction engine
"""
import pytest
from src.ai.knowledge_engine import UnifiedKnowledgeEngine, ExtractedEntity, ExtractionResult
from src.core import unified_config


class TestKnowledgeEngine:
    """Test knowledge extraction engine"""
    
    @pytest.fixture
    def engine(self, monkeypatch):
        """Create engine instance"""
        monkeypatch.setenv("EXPLAINIUM_ENV", "testing")
        monkeypatch.setenv("GPU_BACKEND", "cpu")
        monkeypatch.setenv("ENABLE_GPU", "false")
        unified_config.reload_config()
        try:
            return UnifiedKnowledgeEngine()
        except RuntimeError as e:
            pytest.skip(f"LLM not available: {e}")
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert engine.strategies is not None
        assert len(engine.strategies) > 0
    
    @pytest.mark.asyncio
    async def test_extract_from_simple_text(self, engine):
        """Test extraction from simple text"""
        text = """
        Safety Procedure:
        1. Wear protective equipment including gloves and goggles
        2. Ensure proper ventilation in the work area
        3. Keep fire extinguisher nearby
        
        Equipment Required:
        - Industrial grade gloves
        - Safety goggles
        - Fire extinguisher
        """
        
        result = await engine.extract_knowledge(text, document_type="manual")
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert result.processing_time > 0
        assert result.strategy_used is not None
    
    @pytest.mark.asyncio
    async def test_quality_threshold(self, engine):
        """Test quality threshold filtering"""
        text = "Simple test text without much structure."
        
        # High threshold should still return results
        result = await engine.extract_knowledge(text, quality_threshold=0.9)
        assert result is not None
        assert isinstance(result.entities, list)
    
    def test_performance_stats(self, engine):
        """Test performance statistics tracking"""
        stats = engine.get_performance_stats()
        assert 'total_extractions' in stats
        assert 'cache_hits' in stats
        assert 'strategy_usage' in stats


@pytest.mark.asyncio
@pytest.mark.parametrize("chunking_method", ["fixed", "breakpoint_semantic", "dsc"])
async def test_tiny_end_to_end_runs_for_all_chunkers(monkeypatch, chunking_method):
    """Ensure every chunking method can process a tiny document repeatedly."""
    for key, value in [
        ("EXPLAINIUM_ENV", "testing"),
        ("GPU_BACKEND", "cpu"),
        ("ENABLE_GPU", "false"),
        ("CHUNKING_METHOD", chunking_method),
    ]:
        monkeypatch.setenv(key, value)
    unified_config.reload_config()
    engine = UnifiedKnowledgeEngine()
    tiny_doc = "Step one. Step two. Step three. Final instruction."

    for _ in range(3):
        await engine.clear_cache()
        result = await engine.extract_knowledge(tiny_doc, document_type="dummy")
        assert isinstance(result, ExtractionResult)
        assert result.strategy_used == "mock"
        chunk_meta = result.metadata.get("chunking", {})
        assert chunk_meta.get("method") == chunking_method
        assert chunk_meta.get("count", 0) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
