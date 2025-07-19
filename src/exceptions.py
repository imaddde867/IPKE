"""
Custom exception classes for the Industrial Knowledge Extraction System.
Provides structured error handling across different system components.
"""

from typing import Optional, Dict, Any
import traceback
from datetime import datetime


class BaseKnowledgeExtractionError(Exception):
    """Base exception class for all knowledge extraction system errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.correlation_id = correlation_id
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }


class DatabaseError(BaseKnowledgeExtractionError):
    """Raised when database operations fail."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if table:
            details['table'] = table
        
        super().__init__(message, details=details, **kwargs)


class ProcessingError(BaseKnowledgeExtractionError):
    """Raised when document or media processing fails."""
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if file_type:
            details['file_type'] = file_type
        if processing_stage:
            details['processing_stage'] = processing_stage
        
        super().__init__(message, details=details, **kwargs)


class AIError(BaseKnowledgeExtractionError):
    """Raised when AI processing operations fail."""
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        ai_operation: Optional[str] = None,
        input_length: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if ai_operation:
            details['ai_operation'] = ai_operation
        if input_length:
            details['input_length'] = input_length
        
        super().__init__(message, details=details, **kwargs)


class SearchError(BaseKnowledgeExtractionError):
    """Raised when Elasticsearch operations fail."""
    
    def __init__(
        self, 
        message: str, 
        index_name: Optional[str] = None,
        query: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if index_name:
            details['index_name'] = index_name
        if query:
            details['query'] = query
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)


class ValidationError(BaseKnowledgeExtractionError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        if validation_rule:
            details['validation_rule'] = validation_rule
        
        super().__init__(message, details=details, **kwargs)


class ConfigurationError(BaseKnowledgeExtractionError):
    """Raised when system configuration is invalid or missing."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if expected_type:
            details['expected_type'] = expected_type
        
        super().__init__(message, details=details, **kwargs)


class ServiceUnavailableError(BaseKnowledgeExtractionError):
    """Raised when external services are unavailable."""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None,
        service_url: Optional[str] = None,
        retry_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if service_url:
            details['service_url'] = service_url
        if retry_count is not None:
            details['retry_count'] = retry_count
        
        super().__init__(message, details=details, **kwargs)