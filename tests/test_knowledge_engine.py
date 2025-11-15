"""
Test knowledge extraction engine
"""
import pytest
from src.ai.knowledge_engine import UnifiedKnowledgeEngine, ExtractedEntity, ExtractionResult


@pytest.mark.integration
class TestKnowledgeEngine:
    """Test knowledge extraction engine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance"""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
