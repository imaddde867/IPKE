"""
Test custom exceptions
"""
import pytest
from src.exceptions import (
    BaseKnowledgeExtractionError,
    ProcessingError,
    AIError,
    ValidationError,
    ConfigurationError
)


class TestExceptions:
    """Test custom exception classes"""
    
    def test_base_exception(self):
        """Test base exception"""
        exc = BaseKnowledgeExtractionError("Test error", error_code="TEST001")
        assert str(exc) == "Test error"
        assert exc.error_code == "TEST001"
        assert exc.details == {}
    
    def test_processing_error(self):
        """Test processing error"""
        exc = ProcessingError(
            "File processing failed",
            file_path="/test/file.pdf",
            file_type=".pdf",
            processing_stage="extraction"
        )
        assert "file.pdf" in exc.details['file_path']
        assert exc.details['file_type'] == ".pdf"
    
    def test_ai_error(self):
        """Test AI error"""
        exc = AIError(
            "Model inference failed",
            model_name="mistral-7b",
            ai_operation="extraction"
        )
        assert exc.details['model_name'] == "mistral-7b"
    
    def test_validation_error(self):
        """Test validation error"""
        exc = ValidationError(
            "Invalid input",
            field="file_size",
            value=1000000,
            validation_rule="max_size"
        )
        assert exc.details['field'] == "file_size"
    
    def test_exception_to_dict(self):
        """Test exception serialization"""
        exc = ProcessingError("Test error", file_path="/test.pdf")
        exc_dict = exc.to_dict()
        assert 'error_type' in exc_dict
        assert 'message' in exc_dict
        assert 'details' in exc_dict
        assert 'timestamp' in exc_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
