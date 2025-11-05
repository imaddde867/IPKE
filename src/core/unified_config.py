"""
EXPLAINIUM - Unified Configuration System
"""

import os
from typing import Dict, Any, List, Optional
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


def _safe_int(value: Optional[str], fallback: int) -> int:
    """Safe integer conversion allowing non-negative values only."""
    try:
        result = int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback
    return result if result >= 0 else 0

@dataclass
class UnifiedConfig:
    """
    Unified configuration for the entire Explainium system all settings are centralized here and loaded from environment variables
    with sensible defaults for each environment.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Application
    app_name: str = "Explainium"
    app_version: str = "2.0"
    debug: bool = True
    log_level: str = "INFO"
    
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
    llm_max_chunks: int = 0  # 0 means unlimited chunks; still allow env overrides
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
    max_workers: int = 8
    chunk_size: int = 2000  # Matches knowledge engine chunk size
    cache_size: int = 1000  # Matches knowledge engine cache limit
    processing_timeout: int = 300  # 5 minutes
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ])

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
        confidence_threshold_raw = _get_env_value(
            'CONFIDENCE_THRESHOLD',
            'EXPLAINIUM_CONFIDENCE_THRESHOLD',
            default='0.8'
        )
        try:
            confidence_threshold = float(confidence_threshold_raw)
        except (TypeError, ValueError):
            confidence_threshold = 0.8
        quality_threshold_raw = _get_env_value(
            'QUALITY_THRESHOLD',
            'EXPLAINIUM_QUALITY_THRESHOLD',
            default='0.7'
        )
        try:
            quality_threshold = float(quality_threshold_raw)
        except (TypeError, ValueError):
            quality_threshold = 0.7
        chunk_size_raw = _get_env_value(
            'CHUNK_SIZE',
            'EXPLAINIUM_CHUNK_SIZE',
            default='2000'
        )
        try:
            chunk_size = int(chunk_size_raw)
        except (TypeError, ValueError):
            chunk_size = 2000
        llm_n_ctx_raw = _get_env_value(
            'LLM_N_CTX',
            'EXPLAINIUM_LLM_N_CTX',
            default=str(cls.llm_n_ctx)
        )
        try:
            llm_n_ctx = int(llm_n_ctx_raw)
        except (TypeError, ValueError):
            llm_n_ctx = cls.llm_n_ctx
        llm_temperature_raw = _get_env_value(
            'LLM_TEMPERATURE',
            'EXPLAINIUM_LLM_TEMPERATURE',
            default=str(cls.llm_temperature)
        )
        try:
            llm_temperature = float(llm_temperature_raw)
        except (TypeError, ValueError):
            llm_temperature = cls.llm_temperature
        llm_max_tokens_raw = _get_env_value(
            'LLM_MAX_TOKENS',
            'EXPLAINIUM_LLM_MAX_TOKENS',
            default=str(cls.llm_max_tokens)
        )
        try:
            llm_max_tokens = int(llm_max_tokens_raw)
        except (TypeError, ValueError):
            llm_max_tokens = cls.llm_max_tokens
        llm_max_chunks_raw = _get_env_value(
            'LLM_MAX_CHUNKS',
            'EXPLAINIUM_LLM_MAX_CHUNKS',
            default=str(cls.llm_max_chunks)
        )
        try:
            llm_max_chunks = int(llm_max_chunks_raw)
        except (TypeError, ValueError):
            llm_max_chunks = cls.llm_max_chunks
        else:
            if llm_max_chunks < 0:
                llm_max_chunks = 0
        max_workers_raw = _get_env_value(
            'MAX_WORKERS',
            'EXPLAINIUM_MAX_WORKERS',
            default=str(cls.max_workers)
        )
        try:
            max_workers = int(max_workers_raw)
        except (TypeError, ValueError):
            max_workers = cls.max_workers
        else:
            if max_workers < 1:
                max_workers = cls.max_workers
        
        return cls(
            environment=Environment.DEVELOPMENT,
            debug=True,
            log_level="INFO",  # Changed from DEBUG to match production settings
            max_file_size_mb=max_file_size,
            confidence_threshold=confidence_threshold,  # Optimized setting from knowledge engine
            quality_threshold=quality_threshold,
            chunk_size=chunk_size,
            llm_n_ctx=llm_n_ctx,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_max_chunks=llm_max_chunks,
            max_workers=max_workers,
            processing_timeout=processing_timeout,
            api_host=api_host,
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
        return cls(
            environment=Environment.TESTING,
            debug=False,
            log_level="WARNING",
            enable_gpu=False,  # Disable GPU for tests to avoid conflicts
            llm_n_gpu_layers=0,  # CPU-only for tests
            gpu_backend="cpu",  # Force CPU for reliable testing
            max_file_size_mb=max_file_size,  # Smaller for tests
            confidence_threshold=0.5,
            cache_size=100
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
            log_level=log_level,
            max_file_size_mb=max_file_size,
            confidence_threshold=confidence_threshold,
            quality_threshold=quality_threshold,
            processing_timeout=processing_timeout,
            api_host=api_host,
            cors_origins=cors_origins,
            gpu_backend=gpu_backend,
            enable_gpu=enable_gpu,
            llm_n_gpu_layers=llm_n_gpu_layers,
            gpu_memory_fraction=gpu_memory_fraction,
            llm_max_chunks=_safe_int(
                _get_env_value(
                    'LLM_MAX_CHUNKS',
                    'EXPLAINIUM_LLM_MAX_CHUNKS',
                    default=str(cls.llm_max_chunks)
                ),
                fallback=cls.llm_max_chunks
            )
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

    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return {
            'upload_directory': self.upload_directory,
            'max_file_size_mb': self.max_file_size_mb,
            'supported_formats': self.supported_formats,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'processing_timeout': self.processing_timeout
        }

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
            'gpu_memory_fraction': self.gpu_memory_fraction
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
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'host': self.api_host,
            'port': self.api_port,
            'cors_origins': self.cors_origins,
            'debug': self.debug
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


# Export the unified config as the main interface
__all__ = [
    'UnifiedConfig',
    'get_config',
    'reload_config',
    'Environment'
]
