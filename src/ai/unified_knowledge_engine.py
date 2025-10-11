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
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from src.logging_config import get_logger
from src.core.config import AIConfig

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


class PatternExtractionStrategy(ExtractionStrategy):
    """Fast pattern-based extraction for common entities"""
    
    def __init__(self):
        self.patterns = self._load_extraction_patterns()
        self.confidence_threshold = 0.6
    
    def _load_extraction_patterns(self) -> Dict[str, List[str]]:
        """Load optimized regex patterns for entity extraction"""
        return {
            'procedures': [
                r'(?:step\s+\d+|procedure|process):\s*([^.!?]+[.!?])',
                r'(?:to\s+\w+|must\s+\w+|shall\s+\w+)[^.!?]*[.!?]',
            ],
            'requirements': [
                r'(?:must|shall|required|mandatory)\s+([^.!?]+[.!?])',
                r'(?:compliance|regulation|standard)\s+([^.!?]+[.!?])',
            ],
            'equipment': [
                r'(?:equipment|device|tool|instrument):\s*([^.!?]+[.!?])',
                r'(?:model|serial|part)\s+(?:number|#):\s*([^\s,;]+)',
            ],
            'personnel': [
                r'(?:operator|technician|engineer|manager):\s*([^.!?]+[.!?])',
                r'(?:responsible|assigned|performed)\s+by\s+([^.!?]+[.!?])',
            ],
            'safety': [
                r'(?:danger|warning|caution|hazard):\s*([^.!?]+[.!?])',
                r'(?:ppe|protective|safety)\s+([^.!?]+[.!?])',
            ]
        }
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        """Fast pattern-based extraction"""
        start_time = time.time()
        entities = []
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    entity = ExtractedEntity(
                        content=match.group(1).strip() if match.groups() else match.group(0).strip(),
                        entity_type=category,
                        category=category,
                        confidence=self.confidence_threshold + 0.1,
                        context=self._extract_context(content, match.start(), match.end()),
                        metadata={'pattern': pattern, 'document_type': document_type}
                    )
                    entities.append(entity)
        
        processing_time = time.time() - start_time
        confidence_score = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        
        return ExtractionResult(
            entities=entities,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used="pattern_extraction",
            quality_metrics={'entity_count': len(entities), 'avg_confidence': confidence_score}
        )
    
    def _extract_context(self, content: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around matched entity"""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end].strip()
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold
    
    def get_strategy_name(self) -> str:
        return "Pattern-Based Extraction"


class NLPExtractionStrategy(ExtractionStrategy):
    """NLP-enhanced extraction using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.confidence_threshold = 0.7
        self._initialized = False
    
    async def _initialize(self):
        """Lazy load spaCy model"""
        if not self._initialized and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.model_name)
                self._initialized = True
                logger.info(f"Loaded spaCy model: {self.model_name}")
            except OSError:
                logger.warning(f"spaCy model {self.model_name} not found. Using pattern fallback.")
                self.nlp = None
                self._initialized = True
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        """NLP-enhanced extraction"""
        start_time = time.time()
        await self._initialize()
        
        if not self.nlp:
            # Fallback to pattern extraction
            pattern_strategy = PatternExtractionStrategy()
            result = await pattern_strategy.extract(content, document_type)
            result.strategy_used = "nlp_fallback_to_pattern"
            return result
        
        # Process with spaCy
        doc = self.nlp(content[:1000000])  # Limit to 1M chars for performance
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity = ExtractedEntity(
                content=ent.text,
                entity_type=ent.label_.lower(),
                category=self._map_spacy_label_to_category(ent.label_),
                confidence=self.confidence_threshold + 0.05,
                context=str(ent.sent),
                metadata={'spacy_label': ent.label_, 'start': ent.start, 'end': ent.end}
            )
            entities.append(entity)
        
        # Extract process steps and procedures
        entities.extend(await self._extract_processes(doc))
        
        processing_time = time.time() - start_time
        confidence_score = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        
        return ExtractionResult(
            entities=entities,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used="nlp_extraction",
            quality_metrics={
                'entity_count': len(entities),
                'avg_confidence': confidence_score,
                'named_entities': len([e for e in entities if e.metadata.get('spacy_label')])
            }
        )
    
    async def _extract_processes(self, doc: Doc) -> List[ExtractedEntity]:
        """Extract process-related entities from spaCy doc"""
        entities = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Look for process indicators
            if re.search(r'\b(step|procedure|process|method|approach)\b', sent_text, re.I):
                entity = ExtractedEntity(
                    content=sent_text,
                    entity_type="procedure",
                    category="process",
                    confidence=self.confidence_threshold,
                    context=sent_text,
                    metadata={'sentence_idx': sent.start, 'nlp_extracted': True}
                )
                entities.append(entity)
        
        return entities
    
    def _map_spacy_label_to_category(self, label: str) -> str:
        """Map spaCy entity labels to our categories"""
        mapping = {
            'PERSON': 'personnel',
            'ORG': 'organization',
            'GPE': 'location',
            'PRODUCT': 'equipment',
            'EVENT': 'process',
            'LAW': 'requirement',
            'MONEY': 'financial',
            'PERCENT': 'metric',
            'QUANTITY': 'metric',
            'TIME': 'schedule',
            'DATE': 'schedule',
        }
        return mapping.get(label, 'general')
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold
    
    def get_strategy_name(self) -> str:
        return "NLP-Enhanced Extraction"


class LLMExtractionStrategy(ExtractionStrategy):
    """LLM-powered extraction for complex understanding"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.llm = None
        self.confidence_threshold = 0.8
        self._initialized = False
    
    async def _initialize(self):
        """Lazy load LLM model"""
        if not self._initialized and LLAMA_AVAILABLE and self.model_path:
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False
                )
                self._initialized = True
                logger.info(f"Loaded LLM model: {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LLM model: {e}")
                self.llm = None
                self._initialized = True
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        """LLM-powered extraction"""
        start_time = time.time()
        await self._initialize()
        
        if not self.llm:
            # Fallback to NLP extraction
            nlp_strategy = NLPExtractionStrategy()
            result = await nlp_strategy.extract(content, document_type)
            result.strategy_used = "llm_fallback_to_nlp"
            return result
        
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
        prompt = f"""Extract key information from this {document_type} document text. 
        Identify: procedures, requirements, equipment, personnel roles, safety measures.
        
        Text: {chunk}
        
        Format as JSON list with fields: content, type, category, confidence.
        """
        
        try:
            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                stop=["</s>"]
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
    
    Replaces 4 separate engines with a single, configurable engine using strategy pattern.
    Automatically selects the best extraction strategy based on content and performance requirements.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
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
        """Initialize all available extraction strategies"""
        # Always available
        self.strategies['pattern'] = PatternExtractionStrategy()
        
        # NLP strategy (if spaCy available)
        if SPACY_AVAILABLE:
            self.strategies['nlp'] = NLPExtractionStrategy(self.config.spacy_model)
        
        # LLM strategy (if available and configured)
        if LLAMA_AVAILABLE and hasattr(self.config, 'llm_path'):
            llm_path = getattr(self.config, 'llm_path', None)
            if llm_path:
                self.strategies['llm'] = LLMExtractionStrategy(llm_path)
        
        logger.info(f"Initialized {len(self.strategies)} extraction strategies: {list(self.strategies.keys())}")
    
    async def extract_knowledge(
        self, 
        content: str, 
        document_type: str = "unknown",
        strategy_preference: Optional[str] = None,
        quality_threshold: float = 0.7
    ) -> ExtractionResult:
        """
        Extract knowledge using the best available strategy
        
        Args:
            content: Document content to process
            document_type: Type of document for context
            strategy_preference: Force specific strategy ('pattern', 'nlp', 'llm')
            quality_threshold: Minimum quality threshold for results
        """
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{strategy_preference}_{quality_threshold}"
        
        async with self.cache_lock:
            if cache_key in self.cache:
                self.performance_stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        # Select strategy
        if strategy_preference and strategy_preference in self.strategies:
            strategy = self.strategies[strategy_preference]
        else:
            strategy = self._select_best_strategy(content, document_type, quality_threshold)
        
        # Extract knowledge
        start_time = time.time()
        result = await strategy.extract(content, document_type)
        
        # Validate quality
        if result.confidence_score < quality_threshold:
            # Try fallback strategy
            fallback_strategy = self._get_fallback_strategy(strategy)
            if fallback_strategy and fallback_strategy != strategy:
                logger.info(f"Quality below threshold ({result.confidence_score:.2f} < {quality_threshold:.2f}), trying fallback")
                fallback_result = await fallback_strategy.extract(content, document_type)
                if fallback_result.confidence_score > result.confidence_score:
                    result = fallback_result
                    result.strategy_used += "_with_fallback"
        
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
        """Select the best extraction strategy based on content and requirements"""
        content_length = len(content)
        
        # For high quality requirements and sufficient content, prefer LLM
        if quality_threshold >= 0.8 and content_length > 1000 and 'llm' in self.strategies:
            return self.strategies['llm']
        
        # For medium quality and moderate content, prefer NLP
        if quality_threshold >= 0.6 and content_length > 500 and 'nlp' in self.strategies:
            return self.strategies['nlp']
        
        # Default to pattern extraction for speed
        return self.strategies['pattern']
    
    def _get_fallback_strategy(self, current_strategy: ExtractionStrategy) -> Optional[ExtractionStrategy]:
        """Get fallback strategy when current strategy quality is insufficient"""
        if isinstance(current_strategy, LLMExtractionStrategy) and 'nlp' in self.strategies:
            return self.strategies['nlp']
        elif isinstance(current_strategy, NLPExtractionStrategy) and 'pattern' in self.strategies:
            return self.strategies['pattern']
        return None
    
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
def create_knowledge_engine(config: Optional[AIConfig] = None) -> UnifiedKnowledgeEngine:
    """Create a new knowledge extraction engine with the given configuration"""
    return UnifiedKnowledgeEngine(config)