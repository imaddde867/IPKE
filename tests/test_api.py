"""
Test API endpoints and edge cases.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.unified_config import get_config
from src.ai.knowledge_engine import ExtractionResult
from src.processors.streamlined_processor import ProcessingResult
from src.exceptions import ProcessingError

import src.api.app as api_app


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns health info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'timestamp' in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['version'] == '2.0'
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert 'processing_stats' in data
        assert 'api_info' in data
    
    def test_extract_no_file(self, client):
        """Test extract endpoint without file"""
        response = client.post("/extract")
        assert response.status_code == 422  # Unprocessable entity
    
    def test_extract_unsupported_format(self, client):
        """Test extract endpoint with unsupported format"""
        files = {'file': ('test.xyz', b'test content', 'application/octet-stream')}
        response = client.post("/extract", files=files)
        assert response.status_code == 415  # Unsupported media type
    
    def test_extract_file_too_large(self, client):
        """Test extract endpoint returns 413 when file exceeds limit"""
        config = get_config()
        original_limit = config.max_file_size_mb
        try:
            config.max_file_size_mb = 0  # force zero-byte limit
            files = {'file': ('oversize.txt', b'data', 'text/plain')}
            response = client.post("/extract", files=files)
            assert response.status_code == 413
            data = response.json()
            assert "File too large" in data['detail']
        finally:
            config.max_file_size_mb = original_limit
    
    def test_extract_processing_error_returns_422(self, client, monkeypatch):
        """Ensure ProcessingError surfaces as 422 response"""
        
        async def fail_process(*_args, **_kwargs):
            raise ProcessingError("simulated failure")
        
        monkeypatch.setattr(api_app.processor, "process_document", fail_process)
        files = {'file': ('test.txt', b'content', 'text/plain')}
        response = client.post("/extract", files=files)
        assert response.status_code == 422
        assert "simulated failure" in response.json()['detail']
    
    def test_extract_handles_empty_entities(self, client, monkeypatch, tmp_path):
        """LLM parse fallback should still return 200 with empty entities"""
        
        async def fake_process(*_args, **_kwargs):
            return ProcessingResult(
                document_id="doc123",
                document_type="manual",
                content_extracted="sample",
                extraction_result=ExtractionResult(
                    entities=[],
                    confidence_score=0.0,
                    processing_time=0.01,
                    strategy_used="llm_default"
                ),
                processing_time=0.01,
                file_size=10,
                metadata={}
            )
        
        monkeypatch.setattr(api_app.processor, "process_document", fake_process)
        
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("dummy content")
        files = {'file': (sample_file.name, sample_file.read_bytes(), 'text/plain')}
        response = client.post("/extract", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data['entities'] == []
        assert data['confidence_score'] == 0.0
    
    def test_cors_headers(self, client):
        """Test CORS preflight includes headers"""
        headers = {
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET"
        }
        response = client.options("/", headers=headers)
        assert response.status_code == 200
        assert 'access-control-allow-origin' in response.headers
