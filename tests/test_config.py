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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
