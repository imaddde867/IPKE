"""
EXPLAINIUM - Consolidated Configuration System

A clean, professional configuration management system that consolidates
all configuration logic into a single, well-organized module.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "explainium"
    user: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    upload_directory: str = "uploaded_files"
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        '.pdf', '.doc', '.docx', '.txt', '.rtf',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
        '.xls', '.xlsx', '.csv',
        '.ppt', '.pptx',
        '.mp3', '.wav', '.flac', '.aac',
        '.mp4', '.avi', '.mov', '.mkv'
    ])
    enable_ocr: bool = True
    enable_audio_processing: bool = True
    enable_scene_detection: bool = True  # basic histogram-based scene detection
    parallel_processing: bool = True
    batch_size: int = 10
    video_frame_interval_seconds: int = 5  # extract one frame every N seconds


@dataclass
class AIConfig:
    """AI model configuration"""
    spacy_model: str = "en_core_web_sm"
    classification_model: str = "facebook/bart-large-mnli"
    ner_model: str = "dslim/bert-base-NER"
    whisper_model: str = "base"
    confidence_threshold: float = 0.7
    enable_gpu: bool = False
    model_cache_dir: str = "models"
    # Multi-modal feature flags
    enable_clip: bool = True
    clip_model: str = "openai/clip-vit-base-patch32"
    enable_layout_detection: bool = True  # simple OpenCV-based layout block detection
    enable_speaker_diarization: bool = False  # optional unsupervised diarization
    # Local LLM integration (used by AdvancedKnowledgeEngine)
    quantization: str = "Q4_K_M"  # matches downloaded mistral quantization
    embedding_model: str = "bge-small"  # shorthand for BAAI/bge-small-en-v1.5
    llm_path: str = "models/llm/Mistral-7B-Instruct-v0.2-GGUF"  # directory containing .gguf file


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    rate_limit: int = 100  # requests per minute


@dataclass
class CeleryConfig:
    """Celery configuration"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    worker_concurrency: int = 4


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    enable_https: bool = False
    trusted_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])


class Config:
    """Main configuration class"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.processing = ProcessingConfig()
        self.ai = AIConfig()
        self.api = APIConfig()
        self.celery = CeleryConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load configuration from environment
        self._load_from_environment()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.name = os.getenv("DB_NAME", self.database.name)
        self.database.user = os.getenv("DB_USER", self.database.user)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        self.database.pool_size = int(os.getenv("DB_POOL_SIZE", self.database.pool_size))
        self.database.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", self.database.max_overflow))
        self.database.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        
        # Redis configuration
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.db = int(os.getenv("REDIS_DB", self.redis.db))
        self.redis.password = os.getenv("REDIS_PASSWORD")
        
        # Processing configuration
        self.processing.upload_directory = os.getenv("UPLOAD_DIRECTORY", self.processing.upload_directory)
        self.processing.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", self.processing.max_file_size_mb))
        self.processing.enable_ocr = os.getenv("ENABLE_OCR", "true").lower() == "true"
        self.processing.enable_audio_processing = os.getenv("ENABLE_AUDIO_PROCESSING", "true").lower() == "true"
        self.processing.enable_scene_detection = os.getenv("ENABLE_SCENE_DETECTION", str(self.processing.enable_scene_detection).lower()).lower() == "true"
        self.processing.parallel_processing = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
        self.processing.batch_size = int(os.getenv("BATCH_SIZE", self.processing.batch_size))
        self.processing.video_frame_interval_seconds = int(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS", self.processing.video_frame_interval_seconds))
        
        # AI configuration
        self.ai.spacy_model = os.getenv("SPACY_MODEL", self.ai.spacy_model)
        self.ai.classification_model = os.getenv("CLASSIFICATION_MODEL", self.ai.classification_model)
        self.ai.ner_model = os.getenv("NER_MODEL", self.ai.ner_model)
        self.ai.whisper_model = os.getenv("WHISPER_MODEL", self.ai.whisper_model)
        self.ai.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", self.ai.confidence_threshold))
        self.ai.enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"
        self.ai.model_cache_dir = os.getenv("MODEL_CACHE_DIR", self.ai.model_cache_dir)
        self.ai.enable_clip = os.getenv("ENABLE_CLIP", str(self.ai.enable_clip).lower()).lower() == "true"
        self.ai.clip_model = os.getenv("CLIP_MODEL", self.ai.clip_model)
        self.ai.enable_layout_detection = os.getenv("ENABLE_LAYOUT_DETECTION", str(self.ai.enable_layout_detection).lower()).lower() == "true"
        self.ai.enable_speaker_diarization = os.getenv("ENABLE_SPEAKER_DIARIZATION", str(self.ai.enable_speaker_diarization).lower()).lower() == "true"
        # Local LLM extras
        self.ai.quantization = os.getenv("LLM_QUANTIZATION", self.ai.quantization)
        self.ai.embedding_model = os.getenv("EMBEDDING_MODEL", self.ai.embedding_model)
        self.ai.llm_path = os.getenv("LLM_PATH", self.ai.llm_path)
        
        # API configuration
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        cors_origins = os.getenv("CORS_ORIGINS")
        if cors_origins:
            self.api.cors_origins = [origin.strip() for origin in cors_origins.split(",")]
        self.api.max_request_size = int(os.getenv("MAX_REQUEST_SIZE", self.api.max_request_size))
        self.api.rate_limit = int(os.getenv("RATE_LIMIT", self.api.rate_limit))
        
        # Celery configuration
        self.celery.broker_url = os.getenv("CELERY_BROKER_URL", self.redis.url)
        self.celery.result_backend = os.getenv("CELERY_RESULT_BACKEND", self.redis.url)
        self.celery.worker_concurrency = int(os.getenv("CELERY_CONCURRENCY", self.celery.worker_concurrency))
        
        # Logging configuration
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file_path = os.getenv("LOG_FILE_PATH")
        self.logging.max_file_size = int(os.getenv("LOG_MAX_FILE_SIZE", self.logging.max_file_size))
        self.logging.backup_count = int(os.getenv("LOG_BACKUP_COUNT", self.logging.backup_count))
        self.logging.enable_console = os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true"
        
        # Security configuration
        self.security.secret_key = os.getenv("SECRET_KEY", self.security.secret_key)
        self.security.jwt_algorithm = os.getenv("JWT_ALGORITHM", self.security.jwt_algorithm)
        self.security.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", self.security.jwt_expiration_hours))
        self.security.password_min_length = int(os.getenv("PASSWORD_MIN_LENGTH", self.security.password_min_length))
        self.security.enable_https = os.getenv("ENABLE_HTTPS", "false").lower() == "true"
        trusted_hosts = os.getenv("TRUSTED_HOSTS")
        if trusted_hosts:
            self.security.trusted_hosts = [host.strip() for host in trusted_hosts.split(",")]
    
    def _apply_environment_settings(self):
        """Apply environment-specific settings"""
        if self.environment == Environment.DEVELOPMENT:
            self.api.debug = True
            self.database.echo = True
            self.logging.level = "DEBUG"
            self.logging.enable_console = True
            
        elif self.environment == Environment.TESTING:
            self.database.name = "explainium_test"
            self.logging.level = "WARNING"
            self.processing.max_file_size_mb = 10  # Smaller for tests
            
        elif self.environment == Environment.PRODUCTION:
            self.api.debug = False
            self.database.echo = False
            self.logging.level = "INFO"
            self.security.enable_https = True
            self.api.cors_origins = []  # Restrict CORS in production
            
            # Ensure secure defaults
            if self.security.secret_key == "your-secret-key-change-in-production":
                raise ValueError("SECRET_KEY must be set in production environment")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate database configuration
        if not self.database.host:
            errors.append("Database host cannot be empty")
        if not (1 <= self.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")
        if not self.database.name:
            errors.append("Database name cannot be empty")
        
        # Validate processing configuration
        if self.processing.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")
        if self.processing.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate AI configuration
        if not (0.0 <= self.ai.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        if self.api.max_request_size <= 0:
            errors.append("Max request size must be positive")
        
        # Validate security configuration
        if len(self.security.secret_key) < 32:
            errors.append("Secret key should be at least 32 characters long")
        if self.security.password_min_length < 4:
            errors.append("Password minimum length should be at least 4")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_tika_url(self) -> Optional[str]:
        """Get Tika server URL"""
        tika_host = os.getenv("TIKA_HOST", "localhost")
        tika_port = os.getenv("TIKA_PORT", "9998")
        return f"http://{tika_host}:{tika_port}"
    
    def get_elasticsearch_url(self) -> Optional[str]:
        """Get Elasticsearch URL"""
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
        return f"http://{es_host}:{es_port}"
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.processing.upload_directory,
            self.ai.model_cache_dir,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'name': self.database.name,
                'user': self.database.user,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'echo': self.database.echo
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db
            },
            'processing': {
                'upload_directory': self.processing.upload_directory,
                'max_file_size_mb': self.processing.max_file_size_mb,
                'supported_formats': self.processing.supported_formats,
                'enable_ocr': self.processing.enable_ocr,
                'enable_audio_processing': self.processing.enable_audio_processing,
                'enable_scene_detection': self.processing.enable_scene_detection,
                'parallel_processing': self.processing.parallel_processing,
                'batch_size': self.processing.batch_size,
                'video_frame_interval_seconds': self.processing.video_frame_interval_seconds
            },
            'ai': {
                'spacy_model': self.ai.spacy_model,
                'classification_model': self.ai.classification_model,
                'ner_model': self.ai.ner_model,
                'whisper_model': self.ai.whisper_model,
                'confidence_threshold': self.ai.confidence_threshold,
                'enable_gpu': self.ai.enable_gpu,
                'model_cache_dir': self.ai.model_cache_dir,
                'enable_clip': self.ai.enable_clip,
                'clip_model': self.ai.clip_model,
                'enable_layout_detection': self.ai.enable_layout_detection,
                'enable_speaker_diarization': self.ai.enable_speaker_diarization
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'cors_origins': self.api.cors_origins,
                'max_request_size': self.api.max_request_size,
                'rate_limit': self.api.rate_limit
            },
            'celery': {
                'broker_url': self.celery.broker_url,
                'result_backend': self.celery.result_backend,
                'worker_concurrency': self.celery.worker_concurrency
            },
            'logging': {
                'level': self.logging.level,
                'file_path': self.logging.file_path,
                'enable_console': self.logging.enable_console
            }
        }
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("EXPLAINIUM Configuration Summary")
        print("=" * 60)
        print(f"Environment: {self.environment.value}")
        print(f"Database: {self.database.host}:{self.database.port}/{self.database.name}")
        print(f"Redis: {self.redis.host}:{self.redis.port}/{self.redis.db}")
        print(f"API: {self.api.host}:{self.api.port} (Debug: {self.api.debug})")
        print(f"Upload Directory: {self.processing.upload_directory}")
        print(f"Max File Size: {self.processing.max_file_size_mb}MB")
        print(f"OCR Enabled: {self.processing.enable_ocr}")
        print(f"Audio Processing: {self.processing.enable_audio_processing}")
        print(f"AI Models: spaCy={self.ai.spacy_model}, Whisper={self.ai.whisper_model}")
        print(f"Confidence Threshold: {self.ai.confidence_threshold}")
        print(f"Celery Workers: {self.celery.worker_concurrency}")
        print(f"Log Level: {self.logging.level}")
        print("=" * 60)

    # Convenience accessors expected by other modules
    def get_cors_origins(self) -> List[str]:  # maintains backward compatibility with previous API
        return self.api.cors_origins
    def get_upload_directory(self) -> str:
        return self.processing.upload_directory
    def get_max_file_size(self) -> int:
        return self.processing.max_file_size_mb * 1024 * 1024


# Global configuration instance
def get_environment() -> Environment:
    """Get current environment from environment variable"""
    env_name = os.getenv("ENVIRONMENT", "development").lower()
    try:
        return Environment(env_name)
    except ValueError:
        logger.warning(f"Unknown environment '{env_name}', defaulting to development")
        return Environment.DEVELOPMENT


# Initialize global configuration
config = Config(get_environment())

# Create necessary directories
config.create_directories()

# Print configuration summary on import
if os.getenv("PRINT_CONFIG_SUMMARY", "true").lower() == "true":
    config.print_summary()