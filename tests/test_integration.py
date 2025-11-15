"""
Integration tests for the complete system
"""
from importlib import reload
from pathlib import Path
import tempfile

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.core import unified_config
from src.ai.knowledge_engine import ExtractionResult, ExtractedEntity


class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def client(self, monkeypatch):
        """Create test client"""
        monkeypatch.setenv("EXPLAINIUM_ENV", "testing")
        monkeypatch.setenv("GPU_BACKEND", "cpu")
        monkeypatch.setenv("ENABLE_GPU", "false")
        unified_config.reload_config()
        reload(app_module)
        async def _fake_extract(*args, **kwargs):
            return ExtractionResult(
                entities=[ExtractedEntity(content="Mock entity", entity_type="spec", category="general", confidence=0.9)],
                steps=[{"id": "S1", "text": "Mock step", "order": 1, "confidence": 0.9}],
                constraints=[{"id": "C1", "text": "Mock constraint", "confidence": 0.8, "steps": ["S1"]}],
                confidence_score=0.9,
                processing_time=0.01,
                strategy_used="test-double",
                quality_metrics={"entity_count": 1},
                metadata={"chunking": {"method": "fixed", "count": 1, "avg_sentences": 0, "avg_chars": 10, "avg_cohesion": None}},
            )
        monkeypatch.setattr(
            app_module.processor.knowledge_engine,
            "extract_knowledge",
            _fake_extract,
            raising=True,
        )
        return TestClient(app_module.app)
    
    def test_complete_text_extraction_flow(self, client):
        """Test complete extraction flow with text file"""
        # Create sample document
        content = """
        SAFETY PROTOCOL FOR INDUSTRIAL EQUIPMENT
        
        1. Pre-Operation Checklist:
           - Inspect all safety guards
           - Check emergency stop buttons
           - Verify proper ventilation
        
        2. Required Personal Protective Equipment:
           - Safety glasses (ANSI Z87.1 compliant)
           - Steel-toed boots
           - Hard hat
           - Hearing protection
        
        3. Emergency Procedures:
           - In case of malfunction, press red emergency stop button
           - Evacuate area if alarm sounds
           - Contact supervisor immediately
        
        Technical Specifications:
        - Operating Pressure: 150 PSI
        - Temperature Range: -20°C to 80°C
        - Power Requirements: 480V, 3-phase
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Upload and process
            with open(temp_path, 'rb') as f:
                files = {'file': ('safety_manual.txt', f, 'text/plain')}
                response = client.post("/extract", files=files)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert 'document_id' in data
            assert 'entities' in data
            assert 'confidence_score' in data
            assert 'processing_time' in data
            assert 'strategy_used' in data
            
            # Verify processing occurred
            assert data['processing_time'] > 0
            
            print(f"\n=== Integration Test Results ===")
            print(f"Document ID: {data['document_id']}")
            print(f"Entities Extracted: {len(data['entities'])}")
            print(f"Confidence Score: {data['confidence_score']:.2f}")
            print(f"Processing Time: {data['processing_time']:.2f}s")
            print(f"Strategy Used: {data['strategy_used']}")
            
            if data['entities']:
                print(f"\nSample Entities:")
                for i, entity in enumerate(data['entities'][:3], 1):
                    print(f"{i}. {entity['entity_type']}: {entity['content'][:100]}")
        
        finally:
            Path(temp_path).unlink()
    
    def test_system_health_check(self, client):
        """Test complete system health"""
        # Check root
        response = client.get("/")
        assert response.status_code == 200
        
        # Check health
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check stats
        response = client.get("/stats")
        assert response.status_code == 200
        
        print("\n=== System Health Check ===")
        print("✓ Root endpoint operational")
        print("✓ Health endpoint operational")
        print("✓ Stats endpoint operational")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
