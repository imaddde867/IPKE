"""
Configuration management module for the Industrial Knowledge Extraction System.
Handles environment-based configuration with validation and default values.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "db"
    port: int = 5432
    name: str = "knowledge_db"
    user: str = "user"
    password: str = "password"
    
    @property
    def url(self) -> str:
        """Generate database URL from components."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class RedisConfig:
    """Redis configuration settings."""
    host: str = "redis"
    port: int = 6379
    db: int = 0
    
    @property
    def url(self) -> str:
        """Generate Redis URL from components."""
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration settings."""
    host: str = "elasticsearch"
    port: int = 9200
    index_name: str = "documents"
    
    @property
    def url(self) -> str:
        """Generate Elasticsearch URL from components."""
        return f"http://{self.host}:{self.port}"

@dataclass
class TikaConfig:
    """Apache Tika configuration settings."""
    host: str = "tika"
    port: int = 9998
    
    @property
    def url(self) -> str:
        """Generate Tika URL from components."""
        return f"http://{self.host}:{self.port}"

@dataclass
class APIConfig:
    """API server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000"
    ])

@dataclass
class ProcessingConfig:
    """Document processing configuration settings."""
    upload_directory: str = "./uploaded_files"
    max_file_size_mb: int = 100
    supported_file_types: list = field(default_factory=lambda: [
        "pdf", "docx", "pptx", "txt", "jpg", "jpeg", "png", "mp4", "avi", "mov"
    ])
    # Timeout and retry settings
    tika_timeout_seconds: int = 300
    tika_ocr_timeout_seconds: int = 300
    service_connection_timeout_seconds: int = 5
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 2

@dataclass
class CeleryConfig:
    """Celery task queue configuration settings."""
    log_level: str = "info"
    concurrency: int = 2
    max_retries: int = 3
    retry_delay: int = 60
    task_soft_time_limit: int = 1800  # 30 minutes
    task_time_limit: int = 2400       # 40 minutes
    result_expires: int = 3600        # 1 hour

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None

@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    tika: TikaConfig = field(default_factory=TikaConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    celery: CeleryConfig = field(default_factory=CeleryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment settings
    environment: str = "development"
    debug: bool = True

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigManager:
    """Manages application configuration from environment variables."""
    
    def __init__(self):
        self.config = AppConfig()
        self._load_from_environment()
        self._validate_configuration()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        print("Loading configuration from environment variables")
        
        # Environment settings
        self.config.environment = os.getenv("ENVIRONMENT", "development")
        self.config.debug = os.getenv("DEBUG", "true").lower() == "true"
        
        # Database configuration
        self.config.database.host = os.getenv("DB_HOST", self.config.database.host)
        self.config.database.port = int(os.getenv("DB_PORT", str(self.config.database.port)))
        self.config.database.name = os.getenv("DB_NAME", self.config.database.name)
        self.config.database.user = os.getenv("DB_USER", self.config.database.user)
        self.config.database.password = os.getenv("DB_PASSWORD", self.config.database.password)
        
        # Redis configuration
        self.config.redis.host = os.getenv("REDIS_HOST", self.config.redis.host)
        self.config.redis.port = int(os.getenv("REDIS_PORT", str(self.config.redis.port)))
        self.config.redis.db = int(os.getenv("REDIS_DB", str(self.config.redis.db)))
        
        # Elasticsearch configuration
        self.config.elasticsearch.host = os.getenv("ELASTICSEARCH_HOST", self.config.elasticsearch.host)
        self.config.elasticsearch.port = int(os.getenv("ELASTICSEARCH_PORT", str(self.config.elasticsearch.port)))
        self.config.elasticsearch.index_name = os.getenv("ELASTICSEARCH_INDEX", self.config.elasticsearch.index_name)
        
        # Tika configuration
        self.config.tika.host = os.getenv("TIKA_HOST", self.config.tika.host)
        self.config.tika.port = int(os.getenv("TIKA_PORT", str(self.config.tika.port)))
        
        # API configuration
        self.config.api.host = os.getenv("API_HOST", self.config.api.host)
        self.config.api.port = int(os.getenv("API_PORT", str(self.config.api.port)))
        self.config.api.reload = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # CORS origins from environment (comma-separated)
        cors_origins_env = os.getenv("CORS_ORIGINS")
        if cors_origins_env:
            self.config.api.cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
        
        # Processing configuration
        self.config.processing.upload_directory = os.getenv("UPLOAD_DIRECTORY", self.config.processing.upload_directory)
        self.config.processing.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", str(self.config.processing.max_file_size_mb)))
        self.config.processing.tika_timeout_seconds = int(os.getenv("TIKA_TIMEOUT_SECONDS", str(self.config.processing.tika_timeout_seconds)))
        self.config.processing.tika_ocr_timeout_seconds = int(os.getenv("TIKA_OCR_TIMEOUT_SECONDS", str(self.config.processing.tika_ocr_timeout_seconds)))
        self.config.processing.service_connection_timeout_seconds = int(os.getenv("SERVICE_CONNECTION_TIMEOUT_SECONDS", str(self.config.processing.service_connection_timeout_seconds)))
        self.config.processing.max_retry_attempts = int(os.getenv("MAX_RETRY_ATTEMPTS", str(self.config.processing.max_retry_attempts)))
        self.config.processing.retry_delay_seconds = int(os.getenv("RETRY_DELAY_SECONDS", str(self.config.processing.retry_delay_seconds)))
        
        # Celery configuration
        self.config.celery.log_level = os.getenv("CELERY_LOG_LEVEL", self.config.celery.log_level)
        self.config.celery.concurrency = int(os.getenv("CELERY_CONCURRENCY", str(self.config.celery.concurrency)))
        self.config.celery.max_retries = int(os.getenv("CELERY_MAX_RETRIES", str(self.config.celery.max_retries)))
        self.config.celery.retry_delay = int(os.getenv("CELERY_RETRY_DELAY", str(self.config.celery.retry_delay)))
        
        # Logging configuration
        self.config.logging.level = os.getenv("LOG_LEVEL", self.config.logging.level)
        self.config.logging.format = os.getenv("LOG_FORMAT", self.config.logging.format)
        self.config.logging.file_path = os.getenv("LOG_FILE_PATH")
        
        print(f"Configuration loaded for environment: {self.config.environment}")
    
    def _validate_configuration(self):
        """Validate the loaded configuration."""
        print("Validating configuration")
        
        errors = []
        
        # Validate required string fields are not empty
        required_fields = [
            ("database.host", self.config.database.host),
            ("database.name", self.config.database.name),
            ("database.user", self.config.database.user),
            ("database.password", self.config.database.password),
            ("redis.host", self.config.redis.host),
            ("elasticsearch.host", self.config.elasticsearch.host),
            ("tika.host", self.config.tika.host),
        ]
        
        for field_name, value in required_fields:
            if not value or not value.strip():
                errors.append(f"Required field '{field_name}' is empty or not set")
        
        # Validate port numbers
        port_fields = [
            ("database.port", self.config.database.port),
            ("redis.port", self.config.redis.port),
            ("elasticsearch.port", self.config.elasticsearch.port),
            ("tika.port", self.config.tika.port),
            ("api.port", self.config.api.port),
        ]
        
        for field_name, port in port_fields:
            if not (1 <= port <= 65535):
                errors.append(f"Port '{field_name}' must be between 1 and 65535, got {port}")
        
        # Validate file size limit
        if self.config.processing.max_file_size_mb <= 0:
            errors.append(f"Max file size must be positive, got {self.config.processing.max_file_size_mb}")
        
        # Validate timeout and retry settings
        timeout_fields = [
            ("processing.tika_timeout_seconds", self.config.processing.tika_timeout_seconds),
            ("processing.tika_ocr_timeout_seconds", self.config.processing.tika_ocr_timeout_seconds),
            ("processing.service_connection_timeout_seconds", self.config.processing.service_connection_timeout_seconds),
        ]
        
        for field_name, timeout in timeout_fields:
            if timeout <= 0:
                errors.append(f"Timeout '{field_name}' must be positive, got {timeout}")
        
        if self.config.processing.max_retry_attempts <= 0:
            errors.append(f"Max retry attempts must be positive, got {self.config.processing.max_retry_attempts}")
        
        if self.config.processing.retry_delay_seconds < 0:
            errors.append(f"Retry delay must be non-negative, got {self.config.processing.retry_delay_seconds}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.level.upper() not in valid_log_levels:
            errors.append(f"Log level must be one of {valid_log_levels}, got {self.config.logging.level}")
        
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.config.environment not in valid_environments:
            errors.append(f"Environment must be one of {valid_environments}, got {self.config.environment}")
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            print(f"ERROR: {error_message}")
            raise ConfigurationError(error_message)
        
        print("Configuration validation successful")
    
    def get_config(self) -> AppConfig:
        """Get the validated configuration."""
        return self.config
    
    def get_database_url(self) -> str:
        """Get the database connection URL."""
        return self.config.database.url
    
    def get_redis_url(self) -> str:
        """Get the Redis connection URL."""
        return self.config.redis.url
    
    def get_elasticsearch_url(self) -> str:
        """Get the Elasticsearch connection URL."""
        return self.config.elasticsearch.url
    
    def get_tika_url(self) -> str:
        """Get the Tika server URL."""
        return self.config.tika.url
    
    def print_config_summary(self):
        """Print a summary of the current configuration (without sensitive data)."""
        print("Configuration Summary:")
        print(f"  Environment: {self.config.environment}")
        print(f"  Debug: {self.config.debug}")
        print(f"  Database: {self.config.database.host}:{self.config.database.port}/{self.config.database.name}")
        print(f"  Redis: {self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}")
        print(f"  Elasticsearch: {self.config.elasticsearch.host}:{self.config.elasticsearch.port}")
        print(f"  Tika: {self.config.tika.host}:{self.config.tika.port}")
        print(f"  API: {self.config.api.host}:{self.config.api.port}")
        print(f"  Upload Directory: {self.config.processing.upload_directory}")
        print(f"  Max File Size: {self.config.processing.max_file_size_mb}MB")
        print(f"  Log Level: {self.config.logging.level}")

# Global configuration manager instance
config_manager = ConfigManager()
config = config_manager.get_config()