"""
DEPRECATED: Use src.ai.unified_knowledge_engine instead.
This module provides backward compatibility only.
"""

from src.ai.legacy_compatibility import EnhancedExtractionEngine, OptimizedEnhancedExtractionEngine

# Legacy classes for backward compatibility
ExtractedEntity = None  # Import from unified_knowledge_engine

# Re-export for backward compatibility
__all__ = ['EnhancedExtractionEngine', 'OptimizedEnhancedExtractionEngine', 'ExtractedEntity']