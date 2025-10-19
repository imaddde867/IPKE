"""
Test configuration system
"""
import pytest
import os
from src.core.unified_config import UnifiedConfig, get_config, Environment


class TestUnifiedConfig:
    """Test configuration loading and environment detection"""
    
    def test_default_config_loads(self):
        """Test that default configuration loads successfully"""
        config = get_config()
        assert config is not None
        assert config.app_name == "Explainium"
        assert config.app_version == "2.0"
    
    def test_environment_detection(self):
        """Test environment detection from env vars"""
        config = get_config()
        assert config.environment in [Environment.DEVELOPMENT, Environment.TESTING, Environment.PRODUCTION]
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
