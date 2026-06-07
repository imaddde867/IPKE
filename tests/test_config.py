"""
Test configuration system
"""
import pytest
import os
from src.core.unified_config import UnifiedConfig, get_config, reload_config, Environment


class TestUnifiedConfig:
    """Test configuration loading and environment detection"""
    
    def test_default_config_loads(self):
        """Test that default configuration loads successfully"""
        config = get_config()
        assert config is not None
        assert config.app_name == "Industrial Procedural Knowledge Extraction (IPKE)"
        assert config.app_version == "2.0"
    
    def test_environment_detection(self):
        """Test environment detection from env vars"""
        config = get_config()
        assert config.environment in [
            Environment.DEVELOPMENT,
            Environment.TESTING,
            Environment.PRODUCTION,
            Environment.CLOUD,
        ]
    
    def test_file_size_limits(self):
        """Test file size configuration"""
        config = get_config()
        max_size_bytes = config.get_max_file_size()
        assert max_size_bytes > 0
        assert max_size_bytes == config.max_file_size_mb * 1024 * 1024
    
    def test_supported_formats(self):
        """Test supported file formats are defined"""
        config = get_config()
        assert len(config.supported_formats) > 0
        assert '.pdf' in config.supported_formats
        assert '.docx' in config.supported_formats
    
    def test_model_paths_exist(self):
        """Test that model paths are defined"""
        config = get_config()
        assert config.llm_model_path is not None
        assert config.spacy_model is not None
    
    def test_quality_thresholds(self):
        """Test quality threshold ranges"""
        config = get_config()
        assert 0 <= config.confidence_threshold <= 1
        assert 0 <= config.quality_threshold <= 1
    
    def test_api_config(self):
        """Test API configuration"""
        config = get_config()
        api_config = config.get_api_config()
        assert 'host' in api_config
        assert 'port' in api_config
        assert isinstance(api_config['port'], int)
    
    def test_processing_config(self):
        """Test processing configuration"""
        config = get_config()
        proc_config = config.get_processing_config()
        assert 'max_file_size_mb' in proc_config
        assert 'supported_formats' in proc_config
        assert 'processing_timeout' in proc_config

    def test_gpu_env_overrides(self, monkeypatch):
        """GPU-related env vars should override config values."""
        monkeypatch.delenv('EXPLAINIUM_ENV', raising=False)
        monkeypatch.setenv('LLM_GPU_LAYERS', '10')
        monkeypatch.setenv('GPU_MEMORY_FRACTION', '0.5')
        monkeypatch.setenv('ENABLE_GPU', 'true')
        try:
            reload_config()
            config = get_config()
            llm_config = config.get_llm_config()
            assert llm_config['n_gpu_layers'] == 10
            assert abs(llm_config['gpu_memory_fraction'] - 0.5) < 1e-6
            assert llm_config['enable_gpu'] is True
        finally:
            monkeypatch.delenv('LLM_GPU_LAYERS', raising=False)
            monkeypatch.delenv('GPU_MEMORY_FRACTION', raising=False)
            monkeypatch.delenv('ENABLE_GPU', raising=False)
            reload_config()

    def test_cloud_environment_detected(self, monkeypatch):
        """Setting GCP hints or EXPLAINIUM_ENV=cloud should load the cloud preset."""
        try:
            monkeypatch.delenv('IPKE_ENV', raising=False)
            monkeypatch.delenv('EXPLAINIUM_ENV', raising=False)
            monkeypatch.delenv('ENVIRONMENT', raising=False)
            monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'demo-project')
            reload_config()
            config = get_config()
            assert config.environment == Environment.CLOUD
            assert config.enable_gpu is True
        finally:
            monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)
            reload_config()
        try:
            monkeypatch.delenv('IPKE_ENV', raising=False)
            monkeypatch.setenv('EXPLAINIUM_ENV', 'cloud')
            reload_config()
            config = get_config()
            assert config.environment == Environment.CLOUD
        finally:
            monkeypatch.delenv('EXPLAINIUM_ENV', raising=False)
            reload_config()

    def test_chunking_method_aliases(self, monkeypatch):
        """Chunking method aliases should normalize to canonical names."""
        try:
            monkeypatch.setenv('CHUNKING_METHOD', 'dsc')
            reload_config()
            config = get_config()
            assert config.chunking_method == 'dual_semantic'
        finally:
            monkeypatch.delenv('CHUNKING_METHOD', raising=False)
            reload_config()

    def test_chunk_size_alias(self, monkeypatch):
        """CHUNK_SIZE should act as a fallback for CHUNK_MAX_CHARS."""
        try:
            monkeypatch.delenv('CHUNK_MAX_CHARS', raising=False)
            monkeypatch.setenv('CHUNK_SIZE', '1234')
            reload_config()
            config = get_config()
            assert config.chunk_max_chars == 1234
        finally:
            monkeypatch.delenv('CHUNK_SIZE', raising=False)
            reload_config()


def test_factory_methods_roundtrip(monkeypatch):
    """Regression guard: after collapsing to _parse_env_vars(), the three factory
    methods must produce the same values they did before for the same env-var state.
    """
    baseline = {
        "LLM_NUM_WORKERS": "2",
        "LLM_GPU_LAYERS": "12",
        "GPU_MEMORY_FRACTION": "0.6",
        "CONFIDENCE_THRESHOLD": "0.75",
        "LLM_N_CTX": "4096",
        "LLM_TEMPERATURE": "0.2",
        "LLM_TOP_P": "0.85",
        "LLM_REPEAT_PENALTY": "1.2",
        "LLM_N_THREADS": "6",
        "LLM_MAX_TOKENS": "1024",
        "LLM_MAX_CHUNKS": "5",
        "MAX_WORKERS": "4",
        "LLM_MODEL_PATH": "models/llm/test.gguf",
        "LLM_MODEL_ID": "test/model",
        "LLM_QUANTIZATION": "8bit",
        "LLM_BACKEND": "transformers",
        "LLM_DEVICE_STRATEGY": "single",
        "LLM_RANDOM_SEED": "99",
        "STRICT_SCHEMA_VALIDATION": "1",
        "SCHEMA_AUTOFIX_ENABLED": "0",
        "VALIDATION_ERROR_LOG": "logs/test_errors.jsonl",
        "GPU_BACKEND": "cpu",
        "ENABLE_GPU": "0",
        "PROMPTING_STRATEGY": "P0",
    }
    for key, val in baseline.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("EXPLAINIUM_ENV", raising=False)
    monkeypatch.delenv("IPKE_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    dev = UnifiedConfig._development_config()
    assert dev.debug is True
    assert dev.llm_backend == "transformers"
    assert dev.llm_model_path == "models/llm/test.gguf"
    assert dev.llm_random_seed == 99
    assert dev.confidence_threshold == 0.75
    assert dev.strict_schema_validation is True
    assert dev.schema_autofix_enabled is False
    assert dev.llm_n_gpu_layers == 12
    assert dev.llm_max_chunks == 5
    assert dev.prompting_strategy == "P0"
    assert dev.cors_origins == ["http://localhost:8501", "http://127.0.0.1:8501"]

    prod = UnifiedConfig._production_config()
    assert prod.debug is False
    assert prod.llm_backend == "transformers"
    assert prod.llm_random_seed == 99
    assert prod.confidence_threshold == 0.75
    assert prod.quality_threshold == 0.85
    assert prod.cors_origins == []
    # env vars newly respected by prod mode (was dataclass-default before)
    assert prod.llm_n_ctx == 4096
    assert prod.llm_max_tokens == 1024
    assert prod.max_workers == 4
    assert prod.llm_model_path == "models/llm/test.gguf"

    testing = UnifiedConfig._testing_config()
    assert testing.debug is False
    assert testing.enable_gpu is False
    assert testing.llm_n_gpu_layers == 0
    assert testing.gpu_backend == "cpu"
    assert testing.llm_backend == "transformers"
    assert testing.llm_device_strategy == "single"
    assert testing.llm_num_workers == 1
    assert testing.llm_model_id == "sshleifer/tiny-gpt2"
    assert testing.cache_size == 100
    assert testing.confidence_threshold == 0.5
    assert testing.llm_random_seed == 42
    assert testing.strict_schema_validation is True
    assert testing.validation_error_log == "logs/test_errors.jsonl"
    assert testing.cors_origins == ["http://localhost:8501", "http://127.0.0.1:8501"]
    # env vars newly respected by testing mode (was dataclass-default before)
    assert testing.llm_n_ctx == 4096
    assert testing.llm_max_tokens == 1024
    assert testing.max_workers == 4
    assert testing.llm_model_path == "models/llm/test.gguf"
    assert testing.llm_quantization == "8bit"
    assert testing.llm_n_threads == 6
    assert testing.llm_temperature == 0.2


def test_prod_cors_origins_strips_whitespace(monkeypatch):
    """CORS_ORIGINS values with spaces after commas must be stripped."""
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com ,http://c.com")
    prod = UnifiedConfig._production_config()
    assert prod.cors_origins == ["http://a.com", "http://b.com", "http://c.com"]


def test_prod_cors_origins_empty_string_returns_empty_list(monkeypatch):
    """Empty or unset CORS_ORIGINS must produce [] not ['']. """
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    prod = UnifiedConfig._production_config()
    assert prod.cors_origins == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
