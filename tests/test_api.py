"""
Test API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app


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
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        headers = {
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET"
        }
        response = client.options("/", headers=headers)
        assert response.status_code == 200
        assert 'access-control-allow-origin' in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
