"""
EXPLAINIUM - Unified Configuration System
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback if dependency missing at runtime
    load_dotenv = None

if load_dotenv:
    load_dotenv()

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
    
    # LLM Configuration (GPU-Optimized for comprehensive extraction)
    llm_n_ctx: int = 8192  # Larger context for Mistral
    llm_n_threads: int = 4  # CPU threads for processing (fallback)
    llm_n_gpu_layers: int = -1  # Use all GPU layers (-1 = all layers on GPU)
    llm_max_tokens: int = 1536  # Increased for comprehensive extraction
    llm_temperature: float = 0.1  # Low for deterministic extraction
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_max_chunks: int = 10  # Increased for comprehensive extraction
    llm_f16_kv: bool = True  # Use f16 for key-value cache
    llm_use_mlock: bool = True  # Lock model in memory for better performance
    llm_use_mmap: bool = True  # Memory map model files
    
    # GPU Configuration
    gpu_backend: str = "auto"  # "metal", "cuda", "auto", or "cpu"
    enable_gpu: bool = True  # Enable GPU acceleration by default
    gpu_memory_fraction: float = 0.8  # Use 80% of available GPU memory
    
    # Quality Thresholds (Optimized for LLM extraction performance)
    confidence_threshold: float = 0.8  # Matches knowledge engine setting
    quality_threshold: float = 0.7   # Default extraction threshold
    production_threshold: float = 0.85
    
    # Performance (GPU-Optimized for LLM extraction)
    max_workers: int = 4
    chunk_size: int = 2000  # Matches knowledge engine chunk size
    cache_size: int = 1000  # Matches knowledge engine cache limit
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
        gpu_backend = _get_env_value(
            'GPU_BACKEND',
            'EXPLAINIUM_GPU_BACKEND',
            default='auto'
        )
        enable_gpu = _get_env_value(
            'ENABLE_GPU',
            'EXPLAINIUM_ENABLE_GPU',
            default='true'
        ).lower() == 'true'
        llm_gpu_layers_raw = _get_env_value(
            'LLM_GPU_LAYERS',
            'EXPLAINIUM_LLM_GPU_LAYERS',
            default='-1'
        )
        try:
            llm_n_gpu_layers = int(llm_gpu_layers_raw)
        except (TypeError, ValueError):
            llm_n_gpu_layers = -1
        gpu_memory_fraction_raw = _get_env_value(
            'GPU_MEMORY_FRACTION',
            'EXPLAINIUM_GPU_MEMORY_FRACTION',
            default='0.8'
        )
        try:
            gpu_memory_fraction = float(gpu_memory_fraction_raw)
        except (TypeError, ValueError):
            gpu_memory_fraction = 0.8
        else:
            if gpu_memory_fraction <= 0 or gpu_memory_fraction > 1:
                gpu_memory_fraction = 0.8
        
        return cls(
            environment=Environment.DEVELOPMENT,
            debug=True,
            database_echo=True,
            log_level="INFO",  # Changed from DEBUG to match production settings
            enable_auth=False,
            max_file_size_mb=max_file_size,
            confidence_threshold=0.8,  # Optimized setting from knowledge engine
            processing_timeout=processing_timeout,
            api_host=api_host,
            database_url=database_url,
            gpu_backend=gpu_backend,
            enable_gpu=enable_gpu,
            llm_n_gpu_layers=llm_n_gpu_layers,
            gpu_memory_fraction=gpu_memory_fraction
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
            enable_gpu=False,  # Disable GPU for tests to avoid conflicts
            llm_n_gpu_layers=0,  # CPU-only for tests
            gpu_backend="cpu",  # Force CPU for reliable testing
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
        
        # GPU Configuration for Production
        gpu_backend = _get_env_value(
            'GPU_BACKEND',
            'EXPLAINIUM_GPU_BACKEND',
            default='auto'
        )
        enable_gpu = _get_env_value(
            'ENABLE_GPU',
            'EXPLAINIUM_ENABLE_GPU',
            default='true'
        ).lower() == 'true'
        llm_n_gpu_layers = int(_get_env_value(
            'LLM_GPU_LAYERS',
            'EXPLAINIUM_LLM_GPU_LAYERS',
            default='-1'  # Use all GPU layers by default
        ))
        gpu_memory_fraction = float(_get_env_value(
            'GPU_MEMORY_FRACTION',
            'EXPLAINIUM_GPU_MEMORY_FRACTION',
            default='0.8'
        ))
        
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
            cors_origins=cors_origins,
            gpu_backend=gpu_backend,
            enable_gpu=enable_gpu,
            llm_n_gpu_layers=llm_n_gpu_layers,
            gpu_memory_fraction=gpu_memory_fraction
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
            'gpu_backend': self.gpu_backend,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'enable_llm_processing': self.enable_llm_processing
        }
    
    def detect_gpu_backend(self) -> str:
        """Detect the best available GPU backend"""
        import platform
        
        if not self.enable_gpu:
            return "cpu"
        
        system = platform.system()
        if system == "Darwin":  # macOS
            machine = platform.machine()
            if machine in ["arm64", "aarch64"]:
                return "metal"  # Apple Silicon
            else:
                return "cpu"  # Intel Mac
        else:
            # Check for NVIDIA GPU on Linux/Windows
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return "cuda"
                else:
                    return "cpu"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return "cpu"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration optimized for GPU-accelerated comprehensive extraction"""
        return {
            'model_path': self.llm_model_path,
            'n_ctx': self.llm_n_ctx,
            'n_threads': self.llm_n_threads,
            'n_gpu_layers': self.llm_n_gpu_layers,
            'max_tokens': self.llm_max_tokens,
            'temperature': self.llm_temperature,
            'top_p': self.llm_top_p,
            'repeat_penalty': self.llm_repeat_penalty,
            'max_chunks': self.llm_max_chunks,
            'f16_kv': self.llm_f16_kv,
            'use_mlock': self.llm_use_mlock,
            'use_mmap': self.llm_use_mmap,
            'confidence_threshold': self.confidence_threshold,
            'gpu_backend': self.gpu_backend,
            'enable_gpu': self.enable_gpu,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'verbose': False  # Disable verbose output for cleaner logs
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
