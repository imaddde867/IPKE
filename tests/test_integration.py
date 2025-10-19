"""
Integration tests for the complete system
"""
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app


class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
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
