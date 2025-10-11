"""
EXPLAINIUM - Backward Compatibility Shims

These modules provide backward compatibility with the old architecture.
They redirect to the new unified engine while maintaining the same interface.

DEPRECATED: Use src.ai.unified_knowledge_engine directly for new code.
"""

import warnings
import asyncio
from typing import Dict, List, Any, Optional

from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine, ExtractionResult, ExtractedEntity
from src.core.unified_config import get_config

# Show deprecation warning once per module
_warned = False

def _show_deprecation_warning(old_class: str):
    global _warned
    if not _warned:
        warnings.warn(
            f"{old_class} is deprecated. Use UnifiedKnowledgeEngine instead.",
            DeprecationWarning,
            stacklevel=3
        )
        _warned = True


class AdvancedKnowledgeEngine:
    """Compatibility shim for AdvancedKnowledgeEngine"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("AdvancedKnowledgeEngine")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def extract_knowledge(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        return await self._engine.extract_knowledge(content, document_type)
    
    # Legacy sync method support
    def process_document(self, content: str, document_type: str = "unknown") -> Dict[str, Any]:
        result = asyncio.run(self._engine.extract_knowledge(content, document_type))
        return {
            'entities': [
                {
                    'content': e.content,
                    'entity_type': e.entity_type,
                    'category': e.category,
                    'confidence': e.confidence,
                    'context': e.context,
                    'metadata': e.metadata
                }
                for e in result.entities
            ],
            'confidence_score': result.confidence_score,
            'processing_time': result.processing_time,
            'strategy_used': result.strategy_used
        }


class LLMProcessingEngine:
    """Compatibility shim for LLMProcessingEngine"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("LLMProcessingEngine")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def process_with_llm(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        return await self._engine.extract_knowledge(content, document_type, strategy_preference="llm")
    
    # Legacy method names
    async def process_document_chunked(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        return await self.process_with_llm(content, document_type)


class EnhancedExtractionEngine:
    """Compatibility shim for EnhancedExtractionEngine"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("EnhancedExtractionEngine")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    def extract_comprehensive_knowledge(self, content: str, document_type: str = "unknown") -> List[ExtractedEntity]:
        result = asyncio.run(self._engine.extract_knowledge(content, document_type, strategy_preference="nlp"))
        return result.entities
    
    # Legacy sync methods
    def process_content(self, content: str, document_type: str = "unknown") -> List[Dict[str, Any]]:
        entities = self.extract_comprehensive_knowledge(content, document_type)
        return [
            {
                'content': e.content,
                'entity_type': e.entity_type,
                'category': e.category,
                'confidence': e.confidence,
                'context': e.context,
                'metadata': e.metadata
            }
            for e in entities
        ]


class OptimizedEnhancedExtractionEngine(EnhancedExtractionEngine):
    """Compatibility alias"""
    pass


class KnowledgeCategorizationEngine:
    """Compatibility shim for KnowledgeCategorizationEngine"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("KnowledgeCategorizationEngine")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def categorize_knowledge(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        return await self._engine.extract_knowledge(content, document_type, strategy_preference="nlp")
    
    # Legacy method support
    async def analyze_document_structure(self, content: str) -> Dict[str, Any]:
        result = await self.categorize_knowledge(content)
        return {
            'categories': list(set(e.category for e in result.entities)),
            'entity_types': list(set(e.entity_type for e in result.entities)),
            'confidence': result.confidence_score
        }


class DocumentIntelligenceAnalyzer:
    """Compatibility shim for DocumentIntelligenceAnalyzer"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("DocumentIntelligenceAnalyzer")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def analyze_document_intelligence(self, content: str, filename: str = "", 
                                          document_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        result = await self._engine.extract_knowledge(content, "document")
        return {
            'document_type': 'document',
            'confidence': result.confidence_score,
            'entities_found': len(result.entities),
            'processing_time': result.processing_time,
            'strategy_used': result.strategy_used
        }


class DatabaseOutputGenerator:
    """Compatibility shim for DatabaseOutputGenerator"""
    
    def __init__(self, *args, **kwargs):
        _show_deprecation_warning("DatabaseOutputGenerator")
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def generate_database_entries(self, extraction_result: Any, document_id: str) -> List[Dict[str, Any]]:
        # Convert to our format if needed
        if hasattr(extraction_result, 'entities'):
            entities = extraction_result.entities
        else:
            # Handle legacy format
            entities = extraction_result.get('entities', [])
        
        return [
            {
                'document_id': document_id,
                'content': e.content if hasattr(e, 'content') else e.get('content', ''),
                'entity_type': e.entity_type if hasattr(e, 'entity_type') else e.get('entity_type', 'unknown'),
                'category': e.category if hasattr(e, 'category') else e.get('category', 'general'),
                'confidence': e.confidence if hasattr(e, 'confidence') else e.get('confidence', 0.5),
                'metadata': e.metadata if hasattr(e, 'metadata') else e.get('metadata', {})
            }
            for e in entities
        ]


# Export all legacy classes for backward compatibility
__all__ = [
    'AdvancedKnowledgeEngine',
    'LLMProcessingEngine',
    'EnhancedExtractionEngine',
    'OptimizedEnhancedExtractionEngine',
    'KnowledgeCategorizationEngine', 
    'DocumentIntelligenceAnalyzer',
    'DatabaseOutputGenerator'
]