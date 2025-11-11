"""
Unified Configuration System for Industrial Procedural Knowledge Extraction (IPKE)
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


def _env_int(*keys: str, default: int, min_value: Optional[int] = None) -> int:
    """Fetch an int from env with optional lower bound."""
    raw = _get_env_value(*keys)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None and value < min_value:
        return min_value
    return value


def _env_float(
    *keys: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> float:
    """Fetch a float from env with optional bounds."""
    raw = _get_env_value(*keys)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def _env_bool(*keys: str, default: bool = False) -> bool:
    """Fetch a boolean flag from env."""
    raw = _get_env_value(*keys)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class UnifiedConfig:
    """
    Unified configuration for the entire thesis system (IPKE): all settings are centralized here and loaded from environment variables
    with sensible defaults for each environment.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Application
    app_name: str = "Industrial Procedural Knowledge Extraction (IPKE)"
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
    embedding_model: str = "models/embeddings/bge-small-en-v1.5"
    whisper_model: str = "base"

    # LLM Configuration (Llama.cpp backend)
    llm_model_path: str = "models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    llm_n_gpu_layers: int = -1
    llm_f16_kv: bool = True
    llm_use_mlock: bool = True
    llm_use_mmap: bool = True
    
    # LLM Configuration (Hugging Face Transformers backend)
    llm_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_quantization: str = "4bit"  # "4bit", "8bit", or "none"

    # LLM Common Configuration
    llm_n_ctx: int = 8192
    llm_n_threads: int = 4
    llm_max_tokens: int = 1536
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_max_chunks: int = 0
    
    # GPU Configuration
    gpu_backend: str = "auto"  # "metal", "cuda", "auto", or "cpu"
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Quality Thresholds
    confidence_threshold: float = 0.8
    quality_threshold: float = 0.7
    production_threshold: float = 0.85
    
    # Performance
    max_workers: int = 8
    chunk_size: int = 2000
    cache_size: int = 1000
    processing_timeout: int = 300
    
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
        
        env_name = _get_env_value(
            'EXPLAINIUM_ENV', 'ENVIRONMENT', default='development'
        ).lower()
        try:
            environment = Environment(env_name)
        except ValueError:
            environment = Environment.DEVELOPMENT
        
        if environment == Environment.PRODUCTION:
            return cls._production_config()
        elif environment == Environment.TESTING:
            return cls._testing_config()
        else:
            return cls._development_config()
    
    @classmethod
    def _development_config(cls) -> 'UnifiedConfig':
        """Development environment configuration"""
        return cls(
            environment=Environment.DEVELOPMENT,
            debug=True,
            log_level="INFO",
            max_file_size_mb=_env_int('MAX_FILE_SIZE_MB', default=50, min_value=1),
            processing_timeout=_env_int('PROCESSING_TIMEOUT', default=180, min_value=1),
            api_host=_get_env_value('API_HOST', default='127.0.0.1'),
            gpu_backend=_get_env_value('GPU_BACKEND', default='auto'),
            enable_gpu=_env_bool('ENABLE_GPU', default=True),
            llm_n_gpu_layers=_env_int('LLM_GPU_LAYERS', default=-1),
            gpu_memory_fraction=_env_float('GPU_MEMORY_FRACTION', default=0.8, min_value=0.0, max_value=1.0),
            confidence_threshold=_env_float('CONFIDENCE_THRESHOLD', default=0.8, min_value=0.0, max_value=1.0),
            quality_threshold=_env_float('QUALITY_THRESHOLD', default=0.7, min_value=0.0, max_value=1.0),
            chunk_size=_env_int('CHUNK_SIZE', default=2000, min_value=100),
            llm_n_ctx=_env_int('LLM_N_CTX', default=cls.llm_n_ctx),
            llm_temperature=_env_float('LLM_TEMPERATURE', default=cls.llm_temperature),
            llm_top_p=_env_float('LLM_TOP_P', default=cls.llm_top_p),
            llm_repeat_penalty=_env_float('LLM_REPEAT_PENALTY', default=cls.llm_repeat_penalty),
            llm_n_threads=_env_int('LLM_N_THREADS', default=cls.llm_n_threads, min_value=1),
            llm_max_tokens=_env_int('LLM_MAX_TOKENS', default=cls.llm_max_tokens, min_value=64),
            llm_max_chunks=_env_int('LLM_MAX_CHUNKS', default=cls.llm_max_chunks, min_value=0),
            max_workers=_env_int('MAX_WORKERS', default=cls.max_workers, min_value=1),
            llm_model_id=_get_env_value('LLM_MODEL_ID', default=cls.llm_model_id),
            llm_quantization=_get_env_value('LLM_QUANTIZATION', default=cls.llm_quantization),
        )
    
    @classmethod
    def _testing_config(cls) -> 'UnifiedConfig':
        """Testing environment configuration"""
        return cls(
            environment=Environment.TESTING,
            debug=False,
            log_level="WARNING",
            enable_gpu=False,
            llm_n_gpu_layers=0,
            gpu_backend="cpu",
            max_file_size_mb=_env_int('TEST_MAX_FILE_SIZE_MB', default=10, min_value=1),
            confidence_threshold=0.5,
            cache_size=100,
            llm_model_id="sshleifer/tiny-gpt2"
        )

    @classmethod
    def _production_config(cls) -> 'UnifiedConfig':
        """Production environment configuration"""
        cors_origins_raw = _get_env_value('CORS_ORIGINS', default='')
        cors_origins = cors_origins_raw.split(',') if cors_origins_raw else []
        
        return cls(
            environment=Environment.PRODUCTION,
            debug=False,
            log_level=_get_env_value('LOG_LEVEL', default='INFO'),
            max_file_size_mb=_env_int('MAX_FILE_SIZE_MB', default=200, min_value=1),
            confidence_threshold=_env_float('CONFIDENCE_THRESHOLD', default=0.8, min_value=0.0, max_value=1.0),
            quality_threshold=_env_float('QUALITY_THRESHOLD', default=0.85, min_value=0.0, max_value=1.0),
            processing_timeout=_env_int('PROCESSING_TIMEOUT', default=600, min_value=60),
            api_host=_get_env_value('API_HOST', default='0.0.0.0'),
            cors_origins=cors_origins,
            gpu_backend=_get_env_value('GPU_BACKEND', default='auto'),
            enable_gpu=_env_bool('ENABLE_GPU', default=True),
            llm_n_gpu_layers=_env_int('LLM_GPU_LAYERS', default=-1),
            gpu_memory_fraction=_env_float('GPU_MEMORY_FRACTION', default=0.8, min_value=0.0, max_value=1.0),
            llm_temperature=_env_float('LLM_TEMPERATURE', default=cls.llm_temperature),
            llm_top_p=_env_float('LLM_TOP_P', default=cls.llm_top_p),
            llm_repeat_penalty=_env_float('LLM_REPEAT_PENALTY', default=cls.llm_repeat_penalty),
            llm_n_threads=_env_int('LLM_N_THREADS', default=cls.llm_n_threads, min_value=1),
            llm_max_chunks=_env_int('LLM_MAX_CHUNKS', default=cls.llm_max_chunks, min_value=0),
            llm_model_id=_get_env_value('LLM_MODEL_ID', default=cls.llm_model_id),
            llm_quantization=_get_env_value('LLM_QUANTIZATION', default=cls.llm_quantization),
        )
    
    # Utility methods
    def get_upload_directory(self) -> str:
        path = Path(self.upload_directory)
        path.mkdir(exist_ok=True)
        return str(path.absolute())
    
    def get_max_file_size(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
    
    def get_cors_origins(self) -> List[str]:
        return self.cors_origins

    def get_processing_config(self) -> Dict[str, Any]:
        return {
            'upload_directory': self.upload_directory,
            'max_file_size_mb': self.max_file_size_mb,
            'supported_formats': self.supported_formats,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'processing_timeout': self.processing_timeout
        }

    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration"""
        return {
            'spacy_model': self.spacy_model,
            'llm_model_path': self.llm_model_path,
            'llm_model_id': self.llm_model_id,
            'embedding_model': self.embedding_model,
            'whisper_model': self.whisper_model,
            'confidence_threshold': self.confidence_threshold,
            'enable_gpu': self.enable_gpu,
            'gpu_backend': self.gpu_backend,
        }
    
    def detect_gpu_backend(self) -> str:
        """Detect the best available GPU backend if set to 'auto'."""
        if self.gpu_backend != 'auto':
            return self.gpu_backend
        
        if not self.enable_gpu:
            return "cpu"
        
        import platform
        system = platform.system()
        if system == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            return "metal"
        
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return "cuda"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        return "cpu"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get all LLM-related configuration."""
        return {
            # Common
            'n_ctx': self.llm_n_ctx,
            'n_threads': self.llm_n_threads,
            'max_tokens': self.llm_max_tokens,
            'temperature': self.llm_temperature,
            'top_p': self.llm_top_p,
            'repeat_penalty': self.llm_repeat_penalty,
            'max_chunks': self.llm_max_chunks,
            'confidence_threshold': self.confidence_threshold,
            'enable_gpu': self.enable_gpu,
            'gpu_backend': self.gpu_backend,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'verbose': False,

            # Llama.cpp specific
            'model_path': self.llm_model_path,
            'n_gpu_layers': self.llm_n_gpu_layers,
            'f16_kv': self.llm_f16_kv,
            'use_mlock': self.llm_use_mlock,
            'use_mmap': self.llm_use_mmap,

            # Transformers specific
            'model_id': self.llm_model_id,
            'quantization': self.llm_quantization,
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
