"""
Centralized logging configuration for the Industrial Knowledge Extraction System.
Provides structured JSON logging with correlation IDs and appropriate log levels.
"""

import logging
import logging.config
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from contextvars import ContextVar

# Context variable for correlation ID tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = correlation_id_var.get() or 'no-correlation-id'
        return True


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'no-correlation-id'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


def setup_logging(
    log_level: str = None,
    log_format: str = None,
    enable_console: bool = True,
    enable_file: bool = True,
    log_file_path: str = "logs/app.log"
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'standard')
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        log_file_path: Path to log file
    """
    # Get configuration from environment variables with defaults
    log_level = log_level or os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = log_format or os.getenv('LOG_FORMAT', 'json').lower()
    
    # Ensure log directory exists
    if enable_file:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure formatters
    formatters = {
        'json': {
            '()': JSONFormatter
        },
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    }
    
    # Configure handlers
    handlers = {}
    
    if enable_console:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': log_format,
            'filters': ['correlation_id'],
            'stream': sys.stdout
        }
    
    if enable_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': log_format,
            'filters': ['correlation_id'],
            'filename': log_file_path,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': formatters,
        'filters': {
            'correlation_id': {
                '()': CorrelationIdFilter
            }
        },
        'handlers': handlers,
        'root': {
            'level': log_level,
            'handlers': list(handlers.keys())
        },
        'loggers': {
            'uvicorn': {
                'level': 'INFO',
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'celery': {
                'level': 'INFO',
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'sqlalchemy.engine': {
                'level': 'WARNING',
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'elasticsearch': {
                'level': 'WARNING',
                'handlers': list(handlers.keys()),
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        correlation_id: Custom correlation ID, or None to generate a new one
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def log_error(
    logger: logging.Logger,
    error: Exception,
    message: str = None,
    extra_data: Dict[str, Any] = None
) -> None:
    """
    Log an error with structured information.
    
    Args:
        logger: Logger instance
        error: Exception to log
        message: Custom message (optional)
        extra_data: Additional data to include in log
    """
    log_message = message or f"Error occurred: {str(error)}"
    
    extra = extra_data or {}
    extra.update({
        'error_type': error.__class__.__name__,
        'error_message': str(error)
    })
    
    # Add custom exception details if available
    if hasattr(error, 'to_dict'):
        extra['error_details'] = error.to_dict()
    
    logger.error(log_message, extra=extra, exc_info=True)


def log_processing_step(
    logger: logging.Logger,
    step: str,
    status: str,
    duration: float = None,
    extra_data: Dict[str, Any] = None
) -> None:
    """
    Log a processing step with structured information.
    
    Args:
        logger: Logger instance
        step: Processing step name
        status: Step status (started, completed, failed)
        duration: Processing duration in seconds
        extra_data: Additional data to include in log
    """
    extra = extra_data or {}
    extra.update({
        'processing_step': step,
        'step_status': status
    })
    
    if duration is not None:
        extra['duration_seconds'] = duration
    
    message = f"Processing step '{step}' {status}"
    if duration is not None:
        message += f" in {duration:.2f}s"
    
    if status == 'failed':
        logger.error(message, extra=extra)
    elif status == 'completed':
        logger.info(message, extra=extra)
    else:
        logger.debug(message, extra=extra)


# Initialize logging on module import
setup_logging()