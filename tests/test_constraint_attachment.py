"""Regression: constraints with 'attached_to' key are normalised to 'steps'."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.ai.knowledge_engine import UnifiedKnowledgeEngine
from src.ai.types import ChunkExtraction


@pytest.mark.asyncio
async def test_constraint_attachment_normalised():
    """
    Regression: constraints produced by P1/P2/P3 with key 'attached_to' are
    preserved through the engine and normalised to the 'steps' key in the
    final output. Previously this case was silently dropped.
    """
    # Patch _embed_chunk_texts so the engine never reaches sentence_transformers.
    embed_stub = lambda self, texts: np.eye(max(len(texts), 1))[: len(texts)]

    with patch.object(UnifiedKnowledgeEngine, "_embed_chunk_texts", embed_stub), \
         patch("src.ai.knowledge_engine.get_chunker") as mock_get_chunker, \
         patch("src.ai.knowledge_engine.LLMWorkerPool") as mock_pool_cls:

        mock_chunk = MagicMock()
        mock_chunk.text = "Test chunk"
        mock_chunk.sentence_span = (0, 10)
        mock_chunk.meta = {}
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_get_chunker.return_value = mock_chunker

        extraction_result = ChunkExtraction(
            steps=[{"id": "S1", "text": "Step 1", "confidence": 0.9}],
            constraints=[{
                "id": "C1",
                "text": "Constraint 1",
                "attached_to": ["S1"],  # P1/P2/P3 produce this key
                "confidence": 0.8,
            }],
            entities=[],
        )
        mock_pool = MagicMock()
        mock_pool.process.return_value = [extraction_result]
        mock_pool_cls.return_value = mock_pool

        engine = UnifiedKnowledgeEngine()
        result = await engine.extract_knowledge("Test content")

        assert len(result.steps) == 1
        assert result.steps[0]["id"] == "S1"
        assert len(result.constraints) == 1
        assert result.constraints[0]["steps"] == ["S1"]
        assert "attached_to" not in result.constraints[0]
