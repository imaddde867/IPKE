"""
EXPLAINIUM - Unified Configuration System
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Environment-based configuration
class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


def _get_env_value(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable value from the provided keys."""
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value
    return default

@dataclass
class UnifiedConfig:
    """
    Unified configuration for the entire Explainium system
    
    All configuration is centralized here and loaded from environment variables
    with sensible defaults for each environment.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Application
    app_name: str = "Explainium"
    app_version: str = "2.0"
    debug: bool = True
    
    # File Processing
    upload_directory: str = "uploaded_files"
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        '.pdf', '.doc', '.docx', '.txt', '.rtf',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
        '.xls', '.xlsx', '.csv', '.ppt', '.pptx',
        '.mp3', '.wav', '.flac', '.aac'
    ])
    
    # AI Models
    spacy_model: str = "en_core_web_sm"
    llm_model_path: str = "models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    embedding_model: str = "models/embeddings/bge-small-en-v1.5"
    whisper_model: str = "base"
    
    # Quality Thresholds
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.8
    production_threshold: float = 0.85
    
    # Performance
    enable_gpu: bool = False
    max_workers: int = 4
    chunk_size: int = 2000
    cache_size: int = 1000
    processing_timeout: int = 300  # 5 minutes
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/explainium"
    database_pool_size: int = 10
    database_echo: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:8501",  # Streamlit
        "http://127.0.0.1:8501"
    ])
    
    # Frontend
    frontend_host: str = "0.0.0.0"
    frontend_port: int = 8501
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Security
    secret_key: str = "development-key-change-in-production"
    enable_auth: bool = False
    
    # Feature Flags
    enable_ocr: bool = True
    enable_audio_processing: bool = True
    enable_llm_processing: bool = True
    enable_caching: bool = True
    enable_metrics: bool = True
    
    @classmethod
    def from_environment(cls) -> 'UnifiedConfig':
        """Load configuration from environment variables"""
        
        # Detect environment
        env_name = _get_env_value(
            'EXPLAINIUM_ENV',
            'EXPLAINIUM_ENVIRONMENT',
            'ENVIRONMENT',
            default='development'
        ).lower()
        try:
            environment = Environment(env_name)
        except ValueError:
            environment = Environment.DEVELOPMENT
        
        # Load configuration based on environment
        if environment == Environment.PRODUCTION:
            return cls._production_config()
        elif environment == Environment.TESTING:
            return cls._testing_config()
        else:
            return cls._development_config()
    
    @classmethod
    def _development_config(cls) -> 'UnifiedConfig':
        """Development environment configuration"""
        max_file_size = int(_get_env_value(
            'MAX_FILE_SIZE_MB',
            'EXPLAINIUM_MAX_FILE_SIZE_MB',
            'EXPLAINIUM_MAX_FILE_SIZE',
            default='50'
        ))
        processing_timeout = int(_get_env_value(
            'PROCESSING_TIMEOUT',
            'EXPLAINIUM_PROCESSING_TIMEOUT',
            default='180'
        ))
        api_host = _get_env_value('API_HOST', 'EXPLAINIUM_API_HOST', default='127.0.0.1')
        database_url = _get_env_value(
            'DATABASE_URL',
            'EXPLAINIUM_DATABASE_URL',
            default='postgresql://postgres:password@localhost:5432/explainium_dev'
        )
        return cls(
            environment=Environment.DEVELOPMENT,
            debug=True,
            database_echo=True,
            log_level="DEBUG",
            enable_auth=False,
            max_file_size_mb=max_file_size,
            confidence_threshold=0.6,  # Lower for testing
            processing_timeout=processing_timeout,
            api_host=api_host,
            database_url=database_url
        )
    
    @classmethod
    def _testing_config(cls) -> 'UnifiedConfig':
        """Testing environment configuration"""
        max_file_size = int(_get_env_value(
            'TEST_MAX_FILE_SIZE_MB',
            'EXPLAINIUM_TEST_MAX_FILE_SIZE_MB',
            default='10'
        ))
        database_url = _get_env_value(
            'TEST_DATABASE_URL',
            'EXPLAINIUM_TEST_DATABASE_URL',
            default='postgresql://postgres:password@localhost:5432/explainium_test'
        )
        return cls(
            environment=Environment.TESTING,
            debug=False,
            database_echo=False,
            log_level="WARNING",
            enable_auth=False,
            enable_llm_processing=False,  # Faster tests
            enable_audio_processing=False,
            max_file_size_mb=max_file_size,  # Smaller for tests
            confidence_threshold=0.5,
            cache_size=100,
            database_url=database_url
        )
    
    @classmethod
    def _production_config(cls) -> 'UnifiedConfig':
        """Production environment configuration"""
        log_level = _get_env_value('LOG_LEVEL', 'EXPLAINIUM_LOG_LEVEL', default='INFO')
        max_file_size = int(_get_env_value(
            'MAX_FILE_SIZE_MB',
            'EXPLAINIUM_MAX_FILE_SIZE_MB',
            'EXPLAINIUM_MAX_FILE_SIZE',
            default='200'
        ))
        confidence_threshold = float(_get_env_value(
            'CONFIDENCE_THRESHOLD',
            'EXPLAINIUM_CONFIDENCE_THRESHOLD',
            default='0.8'
        ))
        quality_threshold = float(_get_env_value(
            'QUALITY_THRESHOLD',
            'EXPLAINIUM_QUALITY_THRESHOLD',
            default='0.85'
        ))
        processing_timeout = int(_get_env_value(
            'PROCESSING_TIMEOUT',
            'EXPLAINIUM_PROCESSING_TIMEOUT',
            default='600'
        ))
        secret_key = _get_env_value('SECRET_KEY', 'EXPLAINIUM_SECRET_KEY', default='CHANGE-THIS-IN-PRODUCTION')
        database_url = _get_env_value(
            'DATABASE_URL',
            'EXPLAINIUM_DATABASE_URL',
            default='postgresql://postgres:password@localhost:5432/explainium'
        )
        api_host = _get_env_value('API_HOST', 'EXPLAINIUM_API_HOST', default='0.0.0.0')
        cors_origins_raw = _get_env_value('CORS_ORIGINS', 'EXPLAINIUM_CORS_ORIGINS', default='')
        cors_origins = cors_origins_raw.split(',') if cors_origins_raw else []
        return cls(
            environment=Environment.PRODUCTION,
            debug=False,
            database_echo=False,
            log_level=log_level,
            enable_auth=True,
            max_file_size_mb=max_file_size,
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
            processing_timeout=processing_timeout,
            secret_key=secret_key,
            database_url=database_url,
            api_host=api_host,
            cors_origins=cors_origins
        )
    
    # Utility methods for backward compatibility
    def get_upload_directory(self) -> str:
        """Get upload directory path"""
        path = Path(self.upload_directory)
        path.mkdir(exist_ok=True)
        return str(path.absolute())
    
    def get_max_file_size(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins list"""
        return self.cors_origins
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return self.database_url
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration"""
        return {
            'spacy_model': self.spacy_model,
            'llm_model_path': self.llm_model_path,
            'embedding_model': self.embedding_model,
            'whisper_model': self.whisper_model,
            'confidence_threshold': self.confidence_threshold,
            'enable_gpu': self.enable_gpu,
            'enable_llm_processing': self.enable_llm_processing
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return {
            'upload_directory': self.upload_directory,
            'max_file_size_mb': self.max_file_size_mb,
            'supported_formats': self.supported_formats,
            'enable_ocr': self.enable_ocr,
            'enable_audio_processing': self.enable_audio_processing,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'processing_timeout': self.processing_timeout
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'host': self.api_host,
            'port': self.api_port,
            'cors_origins': self.cors_origins,
            'debug': self.debug,
            'enable_auth': self.enable_auth
        }


# Global configuration instance
_config_instance: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig.from_environment()
    return _config_instance


def reload_config() -> UnifiedConfig:
    """Reload configuration from environment"""
    global _config_instance
    _config_instance = UnifiedConfig.from_environment()
    return _config_instance


# Backward compatibility aliases
class ConfigManager:
    """Backward compatibility class for existing code"""
    
    def __init__(self):
        self._config = get_config()
    
    def get_upload_directory(self) -> str:
        return self._config.get_upload_directory()
    
    def get_max_file_size(self) -> int:
        return self._config.get_max_file_size()
    
    def get_cors_origins(self) -> List[str]:
        return self._config.get_cors_origins()
    
    def get_database_url(self) -> str:
        return self._config.get_database_url()


# Legacy support
config = ConfigManager()
config_manager = config


# Environment configuration for specific components
class AIConfig:
    """AI-specific configuration for backward compatibility"""
    
    def __init__(self):
        self._config = get_config()
    
    @property
    def spacy_model(self) -> str:
        return self._config.spacy_model
    
    @property
    def llm_path(self) -> str:
        return self._config.llm_model_path
    
    @property
    def confidence_threshold(self) -> float:
        return self._config.confidence_threshold
    
    @property
    def enable_gpu(self) -> bool:
        return self._config.enable_gpu


class ProcessingConfig:
    """Processing-specific configuration for backward compatibility"""
    
    def __init__(self):
        self._config = get_config()
    
    @property
    def upload_directory(self) -> str:
        return self._config.upload_directory
    
    @property
    def max_file_size_mb(self) -> int:
        return self._config.max_file_size_mb
    
    @property
    def supported_formats(self) -> List[str]:
        return self._config.supported_formats
    
    @property
    def enable_ocr(self) -> bool:
        return self._config.enable_ocr
    
    @property
    def enable_audio_processing(self) -> bool:
        return self._config.enable_audio_processing


# Export the unified config as the main interface
__all__ = [
    'UnifiedConfig', 
    'get_config', 
    'reload_config', 
    'Environment',
    'ConfigManager',
    'AIConfig', 
    'ProcessingConfig',
    'config',
    'config_manager'
]
