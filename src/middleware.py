"""
Middleware for the Industrial Knowledge Extraction System.
Provides centralized error handling, request logging, and correlation ID management.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.logging_config import get_logger, set_correlation_id, get_correlation_id, log_error
from src.exceptions import (
    BaseKnowledgeExtractionError,
    DatabaseError,
    ProcessingError,
    AIError,
    SearchError,
    ValidationError,
    ConfigurationError,
    ServiceUnavailableError
)

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling and logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Set correlation ID for request tracking
        correlation_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'method': request.method,
                'path': request.url.path,
                'query_params': str(request.query_params),
                'client_ip': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent')
            }
        )
        
        try:
            response = await call_next(request)
            
            # Log successful request completion
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': response.status_code,
                    'duration_seconds': duration
                }
            )
            
            # Add correlation ID to response headers
            response.headers['X-Correlation-ID'] = correlation_id
            return response
            
        except Exception as exc:
            # Log error and return appropriate response
            duration = time.time() - start_time
            return await self._handle_exception(request, exc, duration)
    
    async def _handle_exception(self, request: Request, exc: Exception, duration: float) -> JSONResponse:
        """Handle exceptions and return appropriate JSON responses."""
        
        # Default error response
        status_code = 500
        error_response = {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'correlation_id': get_correlation_id(),
            'timestamp': time.time()
        }
        
        # Handle custom knowledge extraction errors
        if isinstance(exc, BaseKnowledgeExtractionError):
            error_dict = exc.to_dict()
            error_response.update({
                'error': exc.__class__.__name__,
                'message': exc.message,
                'error_code': exc.error_code,
                'details': exc.details
            })
            
            # Set appropriate HTTP status codes
            if isinstance(exc, ValidationError):
                status_code = 400
            elif isinstance(exc, DatabaseError):
                status_code = 500
            elif isinstance(exc, ProcessingError):
                status_code = 422
            elif isinstance(exc, AIError):
                status_code = 503
            elif isinstance(exc, SearchError):
                status_code = 503
            elif isinstance(exc, ConfigurationError):
                status_code = 500
            elif isinstance(exc, ServiceUnavailableError):
                status_code = 503
            
            log_error(
                logger,
                exc,
                f"Custom error in {request.method} {request.url.path}",
                extra_data={
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': status_code,
                    'duration_seconds': duration
                }
            )
        
        # Handle FastAPI HTTPException
        elif isinstance(exc, HTTPException):
            status_code = exc.status_code
            error_response.update({
                'error': 'HTTPException',
                'message': exc.detail
            })
            
            logger.warning(
                f"HTTP exception in {request.method} {request.url.path}: {exc.detail}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': status_code,
                    'duration_seconds': duration,
                    'exception_detail': exc.detail
                }
            )
        
        # Handle unexpected exceptions
        else:
            log_error(
                logger,
                exc,
                f"Unexpected error in {request.method} {request.url.path}",
                extra_data={
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': status_code,
                    'duration_seconds': duration
                }
            )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={'X-Correlation-ID': get_correlation_id() or 'unknown'}
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""
    
    def __init__(self, app: ASGIApp, log_request_body: bool = False, log_response_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request details
        extra_data = {
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'client_ip': request.client.host if request.client else None
        }
        
        # Optionally log request body (be careful with sensitive data)
        if self.log_request_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    extra_data['request_body_size'] = len(body)
                    # Only log first 1000 characters to avoid huge logs
                    if len(body) < 1000:
                        extra_data['request_body'] = body.decode('utf-8', errors='ignore')
            except Exception as e:
                extra_data['request_body_error'] = str(e)
        
        logger.debug("Detailed request information", extra=extra_data)
        
        response = await call_next(request)
        
        # Log response details
        response_extra = {
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'response_headers': dict(response.headers)
        }
        
        logger.debug("Detailed response information", extra=response_extra)
        
        return response


def create_error_handler_for_exception_type(exception_type: type, status_code: int, error_name: str):
    """Create a custom error handler for specific exception types."""
    
    async def handler(request: Request, exc: exception_type):
        correlation_id = get_correlation_id() or str(uuid.uuid4())
        
        error_response = {
            'error': error_name,
            'message': str(exc),
            'correlation_id': correlation_id,
            'timestamp': time.time()
        }
        
        log_error(
            logger,
            exc,
            f"{error_name} in {request.method} {request.url.path}",
            extra_data={
                'method': request.method,
                'path': request.url.path,
                'status_code': status_code
            }
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={'X-Correlation-ID': correlation_id}
        )
    
    return handler