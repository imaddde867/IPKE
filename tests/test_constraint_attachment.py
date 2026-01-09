import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ai.knowledge_engine import UnifiedKnowledgeEngine
from src.ai.types import ChunkExtraction, ExtractedEntity


@pytest.mark.asyncio
async def test_constraint_attachment_bug():
    """
    Reproduce the bug where constraints using 'attached_to' are dropped
    because the engine only looks for 'steps'.
    """
    # Mock sentence_transformers at the module level where it's imported
    mock_st_model = MagicMock()
    mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_st_model))}):
        # Mock the chunker to return a single chunk
        with patch("src.ai.knowledge_engine.get_chunker") as mock_get_chunker:
            mock_chunker = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.text = "Test chunk"
            mock_chunk.sentence_span = (0, 10)
            mock_chunk.meta = {}
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_get_chunker.return_value = mock_chunker

            # Mock the worker pool to return a specific extraction result
            with patch("src.ai.knowledge_engine.LLMWorkerPool") as mock_pool_cls:
                mock_pool = MagicMock()

                # Simulate an extraction result from P1/P2/P3 which uses "attached_to"
                extraction_result = ChunkExtraction(
                    steps=[
                        {"id": "S1", "text": "Step 1", "confidence": 0.9},
                    ],
                    constraints=[
                        {
                            "id": "C1",
                            "text": "Constraint 1",
                            "attached_to": ["S1"],  # This is what P1/P2/P3 produce
                            "confidence": 0.8
                        }
                    ],
                    entities=[]
                )

                mock_pool.process.return_value = [extraction_result]
                mock_pool_cls.return_value = mock_pool

                engine = UnifiedKnowledgeEngine()

                # Run extraction
                result = await engine.extract_knowledge("Test content")

                # Assertions
                assert len(result.steps) == 1
                assert result.steps[0]["id"] == "S1"

                # This assertion is expected to FAIL currently because "attached_to" is ignored
                # The engine looks for "steps" key in constraints
                assert len(result.constraints) == 1
                assert result.constraints[0]["steps"] == ["S1"]
