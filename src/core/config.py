"""
EXPLAINIUM - Configuration (Redirects to Unified Config)

This file now redirects to the new unified configuration for backward compatibility.
For new development, use unified_config directly.
"""

# Import the new unified configuration
from src.core.unified_config import (
    get_config,
    UnifiedConfig,
    ConfigManager,
    AIConfig,
    ProcessingConfig,
    config,
    config_manager
)

# Backward compatibility - export everything
__all__ = [
    'get_config',
    'UnifiedConfig', 
    'ConfigManager',
    'AIConfig',
    'ProcessingConfig',
    'config',
    'config_manager'
]