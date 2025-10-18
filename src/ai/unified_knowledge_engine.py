"""
EXPLAINIUM - Unified Knowledge Extraction Engine

A single, consolidated engine that replaces the 4 separate AI engines with a clean,
pluggable architecture based on extraction strategies and dependency injection.

CONSOLIDATES:
- AdvancedKnowledgeEngine (1,083 lines)
- LLMProcessingEngine (844 lines) 
- EnhancedExtractionEngine (573 lines)
- KnowledgeCategorizationEngine (1,439 lines)

TOTAL REDUCTION: ~3,939 lines → ~800 lines (80% reduction)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import json
import re
import time

# External dependencies (lazy loading)
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

from src.logging_config import get_logger
from src.core.unified_config import UnifiedConfig, get_config

logger = get_logger(__name__)


# Core Data Models
@dataclass
class ExtractedEntity:
    """Unified entity model for all extraction types"""
    content: str
    entity_type: str
    category: str
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    source_location: str = ""


@dataclass
class ExtractionResult:
    """Unified extraction result with quality metrics"""
    entities: List[ExtractedEntity]
    confidence_score: float
    processing_time: float
    strategy_used: str
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExtractionStrategy(ABC):
    """Abstract base class for extraction strategies"""
    
    @abstractmethod
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        """Extract entities from content"""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for this strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name"""
        pass


class LLMExtractionStrategy(ExtractionStrategy):
    """LLM-powered extraction for complex understanding"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.llm = None
        self.confidence_threshold = 0.8
        self._initialized = False
    
    async def _initialize(self):
        """Lazy load LLM model"""
        if self._initialized:
            return
        
        if not LLAMA_AVAILABLE:
            self._initialized = True
            raise RuntimeError("llama_cpp backend not available during model load.")
        
        if not self.model_path:
            self._initialized = True
            raise RuntimeError("LLM model path is not configured.")
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,  # Larger context for Mistral
                n_threads=4,
                n_gpu_layers=0,  # CPU only for compatibility
                verbose=False,
                f16_kv=True  # Use f16 for key-value cache
            )
            self._initialized = True
            logger.info(f"Loaded LLM model: {self.model_path}")
        except Exception as e:
            self._initialized = True
            raise RuntimeError(f"Failed to load LLM model: {e}") from e
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        """LLM-powered extraction"""
        start_time = time.time()
        await self._initialize()
        
        if not self.llm:
            raise RuntimeError("LLM model failed to load.")
        
        # Chunk content for LLM processing
        chunks = self._chunk_content(content, 2000)
        all_entities = []
        
        for chunk in chunks[:5]:  # Limit to 5 chunks for performance
            entities = await self._process_chunk_with_llm(chunk, document_type)
            all_entities.extend(entities)
        
        processing_time = time.time() - start_time
        confidence_score = sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0
        
        return ExtractionResult(
            entities=all_entities,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used="llm_extraction",
            quality_metrics={
                'entity_count': len(all_entities),
                'avg_confidence': confidence_score,
                'chunks_processed': len(chunks[:5])
            }
        )
    
    def _chunk_content(self, content: str, chunk_size: int) -> List[str]:
        """Split content into manageable chunks"""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            # Try to break at sentence boundaries
            if i + chunk_size < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:  # Only if we don't lose too much
                    chunk = chunk[:last_period + 1]
            chunks.append(chunk)
        return chunks
    
    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> List[ExtractedEntity]:
        """Process a single chunk with LLM"""
        prompt = f"""[INST] Extract key information from this {document_type} document text and return ONLY a valid JSON array.

Analyze this text and identify:
- procedures (steps or actions)
- requirements (conditions or constraints)  
- equipment (tools or machines)
- personnel roles (people or responsibilities)
- safety measures (safety conditions or rules)

Text: {chunk}

Return ONLY a JSON array in this exact format:
[
  {{"content": "extracted text", "type": "procedure|requirement|equipment|personnel|safety", "category": "descriptive category", "confidence": 0.85}}
]
[/INST]"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["</s>", "[/INST]"]
            )
            
            return self._parse_llm_response(response['choices'][0]['text'], chunk)
        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str, chunk: str) -> List[ExtractedEntity]:
        """Parse LLM JSON response into entities"""
        entities = []
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                for item in data:
                    if isinstance(item, dict) and 'content' in item:
                        entity = ExtractedEntity(
                            content=item['content'],
                            entity_type=item.get('type', 'unknown'),
                            category=item.get('category', 'general'),
                            confidence=float(item.get('confidence', self.confidence_threshold)),
                            context=chunk[:200] + "...",
                            metadata={'llm_extracted': True, 'source': 'llama'}
                        )
                        entities.append(entity)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return entities
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold
    
    def get_strategy_name(self) -> str:
        return "LLM-Powered Extraction"


class UnifiedKnowledgeEngine:
    """
    Unified Knowledge Extraction Engine
    
    Dedicated LLM-based extraction pipeline powered by a single llama.cpp-backed model.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.strategies = {}
        self.cache = {}
        self.cache_lock = asyncio.Lock()
        self.performance_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'strategy_usage': {},
            'avg_processing_time': 0.0
        }
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize only LLM extraction strategy"""
        # Only LLM strategy - check if available
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp backend not available; LLM-only extraction cannot proceed.")
        
        # Use the Mistral model path from config
        llm_path = getattr(self.config, 'llm_model_path', 'models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf')
        self.strategies['llm'] = LLMExtractionStrategy(llm_path)
        logger.info("Initialized LLM extraction strategy")
        
        logger.info(f"Initialized {len(self.strategies)} extraction strategies: {list(self.strategies.keys())}")
    
    async def extract_knowledge(
        self, 
        content: str, 
        document_type: str = "unknown",
        quality_threshold: float = 0.7
    ) -> ExtractionResult:
        """
        Extract knowledge using the best available strategy
        
        Args:
            content: Document content to process
            document_type: Type of document for context
            quality_threshold: Minimum quality threshold for results
        """
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{quality_threshold}"
        
        async with self.cache_lock:
            if cache_key in self.cache:
                self.performance_stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        # Select strategy (LLM-only)
        strategy = self._select_best_strategy(content, document_type, quality_threshold)
        
        # Extract knowledge
        start_time = time.time()
        result = await strategy.extract(content, document_type)
        
        # Validate quality
        if result.confidence_score < quality_threshold:
            logger.warning(
                "Extraction confidence below threshold",
                extra={
                    "confidence_score": result.confidence_score,
                    "quality_threshold": quality_threshold
                }
            )
        
        # Update statistics
        self.performance_stats['total_extractions'] += 1
        strategy_name = result.strategy_used
        self.performance_stats['strategy_usage'][strategy_name] = \
            self.performance_stats['strategy_usage'].get(strategy_name, 0) + 1
        
        # Cache result
        async with self.cache_lock:
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[cache_key] = result
        
        logger.info(f"Extracted {len(result.entities)} entities using {result.strategy_used} "
                   f"in {result.processing_time:.2f}s (confidence: {result.confidence_score:.2f})")
        
        return result
    
    def _select_best_strategy(self, content: str, document_type: str, quality_threshold: float) -> ExtractionStrategy:
        """Select extraction strategy - always use LLM"""
        return self.strategies['llm']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """Clear the extraction cache"""
        asyncio.create_task(self._clear_cache_async())
    
    async def _clear_cache_async(self):
        """Async cache clearing"""
        async with self.cache_lock:
            self.cache.clear()
            logger.info("Extraction cache cleared")


# Backward compatibility: Export the main interface
KnowledgeExtractor = UnifiedKnowledgeEngine


# Factory function for easy instantiation
def create_knowledge_engine(config: Optional[UnifiedConfig] = None) -> UnifiedKnowledgeEngine:
    """Create a new knowledge extraction engine with the given configuration"""
    return UnifiedKnowledgeEngine(config)
