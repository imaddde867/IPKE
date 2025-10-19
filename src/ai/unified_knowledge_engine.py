"""LLM-driven knowledge extraction engine."""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import json
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
    @abstractmethod
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass


class LLMExtractionStrategy(ExtractionStrategy):
    """LLM-powered extraction for complex understanding."""
    
    MAX_CHUNKS = 10  # Increased for comprehensive extraction
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.llm = None
        self.confidence_threshold = 0.8
        self._initialized = False
    
    async def _initialize(self):
        if self._initialized:
            return

        self._initialized = True
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp backend not available during model load.")
        if not self.model_path:
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
            raise RuntimeError(f"Failed to load LLM model: {e}") from e
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        start_time = time.time()
        await self._initialize()

        if not self.llm:
            raise RuntimeError("LLM model failed to load.")

        chunks = list(self._iter_chunks(content, 2000))
        all_entities = []

        for chunk in chunks:
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
                'chunks_processed': len(chunks)
            }
        )
    
    def _iter_chunks(self, content: str, chunk_size: int):
        """Yield limited chunks, biased toward sentence boundaries."""
        for index, start in enumerate(range(0, len(content), chunk_size)):
            if index >= self.MAX_CHUNKS:
                break
            chunk = content[start:start + chunk_size]
            if start + chunk_size < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]
            yield chunk
    
    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> List[ExtractedEntity]:
        prompt = f"""[INST] Extract ALL key information from this {document_type} document text. Return one array element per distinct item you identify. If you find many items, list them ALL individually.

Comprehensively analyze this text and identify EVERY:
- procedure_step (individual action or step in a process)
- safety_requirement (PPE, protection, hazard warnings)
- equipment_tool (specific tools, machines, products with model numbers)
- specification (measurements, grades, settings, parameters, distances, speeds, pressures)
- material_product (consumables, chemicals, supplies with part numbers)
- requirement_condition (prerequisites, constraints, standards)
- time_duration (timing, curing times, intervals, waiting periods)
- measurement_value (distances, dimensions, percentages, temperatures, RPM, grades)
- contact_info (addresses, phone numbers, emails, websites)
- warning_caution (disclaimers, limitations, important notices)
- quality_standard (grades, ratings, performance criteria)
- maintenance_instruction (cleaning, preparation, finishing steps)
- process_sequence (numbered steps, workflows, procedures)

Text: {chunk}

Extract EVERY technical detail, measurement, part number, grade, timing, distance, speed, and procedure. Be extremely granular:
[
  {{"content": "Hold spray gun 15cm-20cm from the surface", "type": "specification", "category": "distance_requirement", "confidence": 0.95}},
  {{"content": "Use rotary polisher at 1400-1800 rpm", "type": "specification", "category": "speed_setting", "confidence": 0.95}},
  {{"content": "3M Hookit 775L Cubitron II abrasive discs, grade 80+", "type": "material_product", "category": "abrasive", "confidence": 0.95}}
]
[/INST]"""
        
        loop = asyncio.get_running_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    prompt,
                    max_tokens=1536,  # Increased further for comprehensive extraction
                    temperature=0.1,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=["</s>", "[/INST]"]
                )
            )
            
            logger.debug("LLM raw response: %s", response['choices'][0]['text'])
            return self._parse_llm_response(response['choices'][0]['text'], chunk)
        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str, chunk: str) -> List[ExtractedEntity]:
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return [
                    ExtractedEntity(
                        content=item['content'],
                        entity_type=item.get('type', 'unknown'),
                        category=item.get('category', 'general'),
                        confidence=float(item.get('confidence', self.confidence_threshold)),
                        context=chunk[:200] + "...",
                        metadata={'llm_extracted': True, 'source': 'llama'}
                    )
                    for item in data
                    if isinstance(item, dict) and 'content' in item
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return []
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold
    
    def get_strategy_name(self) -> str:
        return "LLM-Powered Extraction"


class UnifiedKnowledgeEngine:
    """LLM-only knowledge extraction pipeline."""
    
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
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp backend not available; LLM-only extraction cannot proceed.")
        llm_path = getattr(self.config, 'llm_model_path', 'models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf')
        self.strategies['llm'] = LLMExtractionStrategy(llm_path)
        logger.info("LLM extraction strategy ready: %s", llm_path)
    
    async def extract_knowledge(
        self, 
        content: str, 
        document_type: str = "unknown",
        quality_threshold: float = 0.7
    ) -> ExtractionResult:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{quality_threshold}"

        async with self.cache_lock:
            if cache_key in self.cache:
                self.performance_stats['cache_hits'] += 1
                return self.cache[cache_key]

        result = await self.strategies['llm'].extract(content, document_type)

        if result.confidence_score < quality_threshold:
            logger.warning(
                "Extraction confidence below threshold",
                extra={
                    "confidence_score": result.confidence_score,
                    "quality_threshold": quality_threshold
                }
            )

        stats = self.performance_stats
        total_before = stats['total_extractions']
        stats['total_extractions'] = total_before + 1
        stats['avg_processing_time'] = (
            (stats['avg_processing_time'] * total_before + result.processing_time)
            / stats['total_extractions']
        )
        strategy_usage = stats['strategy_usage']
        strategy_usage[result.strategy_used] = strategy_usage.get(result.strategy_used, 0) + 1

        async with self.cache_lock:
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[cache_key] = result

        logger.info(f"Extracted {len(result.entities)} entities using {result.strategy_used} "
                   f"in {result.processing_time:.2f}s (confidence: {result.confidence_score:.2f})")
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_stats.copy()
    
    def clear_cache(self):
        asyncio.create_task(self._clear_cache_async())
    
    async def _clear_cache_async(self):
        async with self.cache_lock:
            self.cache.clear()
            logger.info("Extraction cache cleared")


# Backward compatibility: Export the main interface
KnowledgeExtractor = UnifiedKnowledgeEngine


# Factory function for easy instantiation
def create_knowledge_engine(config: Optional[UnifiedConfig] = None) -> UnifiedKnowledgeEngine:
    """Create a new knowledge extraction engine with the given configuration"""
    return UnifiedKnowledgeEngine(config)
