#!/usr/bin/env python3
"""
OPTIMIZED LLM Processing Engine - Performance-First Implementation
================================================================

This engine is optimized for SPEED FIRST, then quality. Target: 2 minutes max per document.
Current: 10+ minutes per document. Improvement needed: 5x speed increase minimum.

OPTIMIZATION STRATEGIES:
1. Async Processing Pipeline
2. Content Chunking & Streaming  
3. Intelligent Caching System
4. Parallel Entity Extraction
5. Smart Processing Decisions
6. M4 Chip Optimizations
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Import LLM backends
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

from src.logging_config import get_logger
from src.core.config import AIConfig
from src.ai.enhanced_extraction_engine import EnhancedExtractionEngine, ExtractedEntity

logger = get_logger(__name__)

class ProcessingPriority(Enum):
    """Processing priority levels for different extraction methods"""
    CRITICAL = "critical"    # LLM-based processing (0.95+ reliability)
    HIGH = "high"           # Enhanced pattern + LLM validation (0.80+ reliability)
    MEDIUM = "medium"       # Pattern recognition with NLP (0.65+ reliability)
    LOW = "low"             # Basic pattern matching (0.50+ reliability)

class QualityThreshold(Enum):
    """Quality thresholds for different processing stages"""
    LLM_MINIMUM = 0.75          # Minimum LLM confidence
    ENHANCED_MINIMUM = 0.60     # Enhanced extraction minimum
    COMBINED_MINIMUM = 0.80     # Combined analysis threshold
    ENTITY_VALIDATION = 0.70    # Entity validation score
    PRODUCTION_READY = 0.85     # Production deployment threshold

@dataclass
class ProcessingRule:
    """Defines a processing rule with conditions and actions"""
    name: str
    condition: str
    action: str
    priority: ProcessingPriority
    threshold: float
    description: str

@dataclass
class ProcessingResult:
    """Comprehensive processing result with quality metrics"""
    entities: List[ExtractedEntity]
    processing_method: str
    confidence_score: float
    quality_metrics: Dict[str, float]
    processing_time: float
    llm_enhanced: bool
    validation_passed: bool
    metadata: Dict[str, Any]

class ProcessingCache:
    """Intelligent caching system for processed patterns and entities"""
    
    def __init__(self, max_size: int = 1000):
        self.pattern_cache = {}
        self.entity_cache = {}
        self.similarity_cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_cached_result(self, content_hash: str, pattern_type: str) -> Optional[Any]:
        """Get cached result for similar content"""
        if content_hash in self.entity_cache:
            self.access_count[content_hash] = self.access_count.get(content_hash, 0) + 1
            return self.entity_cache[content_hash]
        return None
    
    def cache_result(self, content_hash: str, pattern_type: str, result: Any):
        """Cache processing result"""
        if len(self.entity_cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])
            del self.entity_cache[least_accessed[0]]
            del self.access_count[least_accessed[0]]
        
        self.entity_cache[content_hash] = result
        self.access_count[content_hash] = 1
    
    def get_similar_content(self, content: str, threshold: float = 0.8) -> Optional[Any]:
        """Find similar cached content using simple similarity"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        for cached_hash, cached_result in self.entity_cache.items():
            # Simple similarity check - can be enhanced with embeddings
            if self._calculate_similarity(content, cached_result.get('content', '')) > threshold:
                return cached_result
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class OptimizedLLMProcessingEngine:
    """
    OPTIMIZED LLM Processing Engine - Performance-First Implementation
    
    CORE OPTIMIZATION PRINCIPLES:
    ============================
    1. SPEED FIRST: Target 2 minutes max per document
    2. ASYNC PROCESSING: Parallel execution of independent tasks
    3. CONTENT CHUNKING: Process large documents in manageable pieces
    4. INTELLIGENT CACHING: Avoid re-processing similar content
    5. SMART ROUTING: Choose fastest processing method for content type
    6. M4 OPTIMIZATION: Leverage Apple Silicon architecture
    """
    
    # Core Processing Rules - OPTIMIZED FOR SPEED
    PROCESSING_RULES = [
        ProcessingRule(
            name="fast_pattern_first",
            condition="content_length < 2000 and complexity < 0.3",
            action="use_fast_patterns_only",
            priority=ProcessingPriority.LOW,
            threshold=0.50,
            description="Fast pattern matching for simple content"
        ),
        ProcessingRule(
            name="enhanced_fallback",
            condition="content_length < 5000 and complexity < 0.6",
            action="use_enhanced_patterns",
            priority=ProcessingPriority.MEDIUM,
            threshold=0.60,
            description="Enhanced patterns for medium complexity"
        ),
        ProcessingRule(
            name="llm_optimized",
            condition="content_length >= 5000 or complexity >= 0.6",
            action="use_llm_with_chunking",
            priority=ProcessingPriority.CRITICAL,
            threshold=0.75,
            description="LLM processing with content chunking for complex content"
        )
    ]
    
    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.llm_model = None
        self.enhanced_engine = None
        self.initialized = False
        self.processing_cache = ProcessingCache()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Optimize for M4
        self.processing_stats = {
            "total_documents": 0,
            "total_processing_time": 0.0,
            "method_usage": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Performance monitoring
        self.start_time = None
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the processing engine asynchronously"""
        if self.initialized:
            return
        
        logger.info("🚀 Initializing Optimized LLM Processing Engine")
        
        # Initialize enhanced engine
        self.enhanced_engine = EnhancedExtractionEngine()
        
        # Initialize LLM model (if available)
        if LLAMA_AVAILABLE:
            await self._initialize_primary_llm()
            
            self.initialized = True
        logger.info("✅ Optimized LLM Processing Engine initialized")
    
    async def _initialize_primary_llm(self):
        """Initialize primary LLM model with M4 optimizations"""
        try:
            # Use smaller, faster model for M4 optimization
            model_path = self.config.get("llm_model_path", "models/llama-2-7b-chat.gguf")
            
            if Path(model_path).exists():
                # M4-optimized settings
                    self.llm_model = Llama(
                    model_path=model_path,
                    n_ctx=2048,  # Reduced context for speed
                    n_batch=512,  # Optimized batch size for M4
                    n_threads=8,  # Optimize for M4 chip
                    n_gpu_layers=0,  # CPU-only for M4 optimization
                    verbose=False
                )
                logger.info("✅ LLM model initialized with M4 optimizations")
                else:
                logger.warning("⚠️ LLM model not found, using enhanced patterns only")
                self.llm_model = None
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            self.llm_model = None
    
    async def process_document(self, content: str, document_type: str = "unknown", 
                             metadata: Dict[str, Any] = None) -> ProcessingResult:
        """
        OPTIMIZED PROCESSING FUNCTION - Target: 2 minutes max
        
        OPTIMIZED FLOW:
        1. Fast Content Analysis (async)
        2. Smart Processing Method Selection
        3. Parallel Content Processing
        4. Fast Validation & Enhancement
        5. Structured Output Generation
        """
        if not self.initialized:
            await self.initialize()
        
        self.start_time = time.time()
        metadata = metadata or {}
        
        logger.info(f"🚀 Starting OPTIMIZED processing for {document_type} document")
        
        # STEP 1: Fast Content Analysis (async)
        complexity_task = asyncio.create_task(self._assess_document_complexity_fast(content, document_type))
        
        # STEP 2: Check Cache First (immediate)
        cache_result = self.processing_cache.get_similar_content(content)
        if cache_result:
            self.processing_stats["cache_hits"] += 1
            logger.info("🎯 Cache hit - using cached results")
            return self._create_result_from_cache(cache_result, document_type)
        
        self.processing_stats["cache_misses"] += 1
        
        # STEP 3: Get Complexity Score
        complexity_score = await complexity_task
        
        # STEP 4: Smart Processing Method Selection
        processing_method = self._determine_processing_method_fast(complexity_score, len(content), metadata)
        
        # STEP 5: Execute Processing (with timeout)
        try:
            entities = await asyncio.wait_for(
                self._execute_processing_optimized(content, document_type, processing_method, metadata),
                timeout=90.0  # 90 second timeout for 2-minute target
            )
        except asyncio.TimeoutError:
            logger.warning("⏰ Processing timeout, falling back to fast patterns")
            entities = await self._execute_fast_pattern_fallback(content, document_type)
        
        # STEP 6: Fast Validation (minimal)
        validated_entities = await self._validate_entities_fast(entities, content)
        
        # STEP 7: Calculate Quality Metrics (fast)
        quality_metrics = self._calculate_quality_metrics_fast(validated_entities, processing_method)
        
        # STEP 8: Generate Result
        processing_time = time.time() - self.start_time
        
        result = ProcessingResult(
            entities=validated_entities,
            processing_method=processing_method,
            confidence_score=quality_metrics["overall_confidence"],
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            llm_enhanced=processing_method.startswith("llm"),
            validation_passed=quality_metrics["validation_score"] >= QualityThreshold.ENTITY_VALIDATION.value,
            metadata={
                "document_type": document_type,
                "complexity_score": complexity_score,
                "processing_stats": self._update_processing_stats(processing_method, quality_metrics),
                "cache_used": False
            }
        )
        
        # Cache the result
        content_hash = hashlib.md5(content.encode()).hexdigest()
        self.processing_cache.cache_result(content_hash, "document", {
            "entities": validated_entities,
            "content": content[:1000],  # Store first 1000 chars for similarity
            "result": result
        })
        
        # Update stats
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        logger.info(f"✅ OPTIMIZED processing complete: {len(validated_entities)} entities, "
                   f"{quality_metrics['overall_confidence']:.2f} confidence, "
                   f"{processing_time:.2f}s (Target: <120s)")
        
        return result
    
    async def _assess_document_complexity_fast(self, content: str, document_type: str) -> float:
        """Fast document complexity assessment"""
        # Simple, fast complexity calculation
        length_factor = min(len(content) / 10000, 1.0) * 0.4
        technical_factor = min(content.count('specification') + content.count('procedure') + 
                              content.count('requirement'), 10) / 10 * 0.3
        structure_factor = min(content.count('\n\n') + content.count('•') + 
                              content.count('-'), 20) / 20 * 0.3
        
        complexity_score = length_factor + technical_factor + structure_factor
        return min(complexity_score, 1.0)
    
    def _determine_processing_method_fast(self, complexity_score: float, content_length: int, 
                                        metadata: Dict[str, Any]) -> str:
        """Fast processing method determination"""
        
        # Rule 1: Fast patterns for simple content
        if content_length < 2000 and complexity_score < 0.3:
            return "fast_patterns"
        
        # Rule 2: Enhanced patterns for medium content
        if content_length < 5000 and complexity_score < 0.6:
            return "enhanced_patterns"
        
        # Rule 3: LLM with chunking for complex content
        if self.llm_model is not None:
            return "llm_chunked"
        
        # Rule 4: Enhanced patterns as fallback
        return "enhanced_patterns"
    
    async def _execute_processing_optimized(self, content: str, document_type: str, 
                                          method: str, metadata: Dict[str, Any]) -> List[ExtractedEntity]:
        """Execute optimized processing method"""
        
        if method == "fast_patterns":
            return await self._execute_fast_patterns(content, document_type)
        elif method == "enhanced_patterns":
            return await self._execute_enhanced_patterns(content, document_type)
        elif method == "llm_chunked":
            return await self._execute_llm_chunked(content, document_type)
        else:
            return await self._execute_fast_patterns(content, document_type)
    
    async def _execute_fast_patterns(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Execute fast pattern matching"""
        entities = []
        
        # Fast, simple patterns
        patterns = {
            "specifications": r'(\d+\.?\d*)\s*(HP|V|PSI|GPM|°F|°C)\s*([^\.]{0,50})',
            "procedures": r'(must|shall|should|required)\s+([^\.]{20,100})',
            "personnel": r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*-\s*([^,\n]{10,50})',
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity = ExtractedEntity(
                    content=match.group(0).strip(),
                    entity_type=pattern_type,
                    category="fast_extraction",
                    confidence=0.60,
                    context=content[max(0, match.start()-50):match.end()+50],
                    metadata={
                        "processing_method": "fast_patterns",
                        "pattern_type": pattern_type,
                        "optimized": True
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                )
                entities.append(entity)
        
        return entities
    
    async def _execute_enhanced_patterns(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Execute enhanced pattern processing"""
        if self.enhanced_engine:
            return self.enhanced_engine.extract_comprehensive_knowledge(content, document_type)
        else:
            return await self._execute_fast_patterns(content, document_type)
    
    async def _execute_llm_chunked(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Execute LLM processing with content chunking"""
        if not self.llm_model:
            return await self._execute_enhanced_patterns(content, document_type)
        
        # Chunk content for faster processing
        chunks = self._chunk_content_optimized(content, max_chunk_size=1500)
        
        # Process chunks in parallel
        chunk_tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_chunk_with_llm(chunk, document_type))
            chunk_tasks.append(task)
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Merge results
        all_entities = []
        for result in chunk_results:
            if isinstance(result, list):
                all_entities.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Chunk processing error: {result}")
        
        return all_entities
    
    def _chunk_content_optimized(self, content: str, max_chunk_size: int = 1500) -> List[str]:
        """Optimized content chunking for M4 processing"""
        chunks = []
        
        # Simple chunking by sentences and paragraphs
        sentences = re.split(r'[.!?]+', content)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [content[:max_chunk_size]]
        
        return chunks
    
    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> List[ExtractedEntity]:
        """Process a single chunk with LLM"""
        try:
            # Simple LLM prompt for speed
            prompt = f"""Extract key information from this {document_type} document chunk:
            
{chunk}

Extract entities in this format:
- Technical specifications
- Procedures and processes  
- Safety requirements
- Personnel information
- Equipment details

Format: [Entity Type]: [Content]"""

            # Query LLM with timeout
            response = await asyncio.wait_for(
                self._query_llm_async(prompt),
                timeout=15.0  # 15 second timeout per chunk
            )
            
            # Parse response into entities
            entities = self._parse_llm_response_fast(response, chunk, document_type)
            return entities
            
        except Exception as e:
            logger.warning(f"LLM chunk processing error: {e}")
            return []
    
    async def _query_llm_async(self, prompt: str) -> str:
        """Async LLM query with thread pool execution"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            self._query_llm_sync,
            prompt
        )
        return response
    
    def _query_llm_sync(self, prompt: str) -> str:
        """Synchronous LLM query"""
        try:
            response = self.llm_model(prompt, max_tokens=500, temperature=0.1)
            return response.get('choices', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return ""
    
    def _parse_llm_response_fast(self, response: str, chunk: str, document_type: str) -> List[ExtractedEntity]:
        """Fast parsing of LLM response"""
        entities = []
        
        # Simple parsing by lines
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line or not ':' in line:
                continue
            
            try:
                entity_type, content = line.split(':', 1)
                entity_type = entity_type.strip().strip('-').strip()
                content = content.strip()
                
                if len(content) > 10:  # Minimum content length
                    entity = ExtractedEntity(
                        content=content,
                        entity_type=entity_type,
                        category="llm_extraction",
                        confidence=0.80,
                        context=chunk,
                        metadata={
                            "processing_method": "llm_chunked",
                            "chunk_processed": True,
                            "optimized": True
                        },
                        relationships=[],
                        source_location="llm_generated"
                    )
                    entities.append(entity)
            except:
                continue
        
        return entities
    
    async def _execute_fast_pattern_fallback(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Fast pattern fallback for timeout scenarios"""
        return await self._execute_fast_patterns(content, document_type)
    
    async def _validate_entities_fast(self, entities: List[ExtractedEntity], content: str) -> List[ExtractedEntity]:
        """Fast entity validation (minimal)"""
        validated_entities = []
        
        for entity in entities:
            # Fast validation checks
            if (entity.confidence >= 0.5 and 
                len(entity.content) >= 5 and 
                entity.category != "unknown"):
                validated_entities.append(entity)
        
        return validated_entities
    
    def _calculate_quality_metrics_fast(self, entities: List[ExtractedEntity], 
                                 processing_method: str) -> Dict[str, float]:
        """Fast quality metrics calculation"""
        if not entities:
            return {
                "overall_confidence": 0.0,
                "validation_score": 0.0,
                "entity_count": 0,
                "processing_efficiency": 0.0
            }
        
        # Fast confidence calculation
        confidences = [entity.confidence for entity in entities]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Fast validation score
        validation_score = sum(1 for entity in entities if entity.confidence >= 0.6) / len(entities)
        
        return {
            "overall_confidence": overall_confidence,
            "validation_score": validation_score,
            "entity_count": len(entities),
            "processing_efficiency": 1.0 if processing_method.startswith("llm") else 0.8
        }
    
    def _create_result_from_cache(self, cache_result: Dict[str, Any], document_type: str) -> ProcessingResult:
        """Create result from cached data"""
        cached_data = cache_result.get("result")
        if cached_data:
            # Update metadata for new document type
            cached_data.metadata["document_type"] = document_type
            cached_data.metadata["cache_used"] = True
            cached_data.processing_time = 0.1  # Very fast from cache
            return cached_data
        
        # Fallback to creating new result
        return ProcessingResult(
            entities=cache_result.get("entities", []),
            processing_method="cached",
            confidence_score=0.75,
            quality_metrics={"overall_confidence": 0.75, "cache_hit": True},
            processing_time=0.1,
            llm_enhanced=False,
            validation_passed=True,
            metadata={"document_type": document_type, "cache_used": True}
        )
    
    def _update_processing_stats(self, method: str, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update processing statistics"""
        if method not in self.processing_stats["method_usage"]:
            self.processing_stats["method_usage"][method] = 0
        
        self.processing_stats["method_usage"][method] += 1
        
        return {
            "method_usage": self.processing_stats["method_usage"],
            "total_documents": self.processing_stats["total_documents"],
            "cache_hit_rate": (self.processing_stats["cache_hits"] / 
                              (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]))
            if (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]) > 0 else 0.0
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing performance summary"""
        total_docs = self.processing_stats["total_documents"]
        avg_time = (self.processing_stats["total_processing_time"] / total_docs 
                   if total_docs > 0 else 0.0)
        
        return {
            "total_documents_processed": total_docs,
            "average_processing_time": avg_time,
            "cache_hit_rate": (self.processing_stats["cache_hits"] / 
                              (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]))
            if (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]) > 0 else 0.0,
            "method_distribution": self.processing_stats["method_usage"],
            "performance_target_met": avg_time <= 120.0,  # 2 minutes target
            "speed_improvement": (600.0 / avg_time) if avg_time > 0 else 0.0  # 10 min -> target
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("🧹 Optimized LLM Processing Engine cleaned up")

<<<<<<< Current (Your changes)
# Export the main processing engine
__all__ = ["LLMProcessingEngine", "ProcessingResult", "ProcessingPriority", "QualityThreshold"]
=======
# Backward compatibility
LLMProcessingEngine = OptimizedLLMProcessingEngine
>>>>>>> Incoming (Background Agent changes)
