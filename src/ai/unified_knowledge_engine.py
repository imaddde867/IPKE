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
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        llm_config = self.config.get_llm_config()
        
        self.model_path = llm_config['model_path']
        self.max_chunks = llm_config['max_chunks']
        
        # GPU-optimized LLM parameters
        self.llm_params = {
            'n_ctx': llm_config['n_ctx'],
            'n_threads': llm_config['n_threads'],
            'n_gpu_layers': llm_config['n_gpu_layers'],
            'f16_kv': llm_config['f16_kv'],
            'use_mlock': llm_config['use_mlock'],
            'use_mmap': llm_config['use_mmap'],
            'verbose': llm_config['verbose']
        }
        
        self.generation_params = {
            'max_tokens': llm_config['max_tokens'],
            'temperature': llm_config['temperature'],
            'top_p': llm_config['top_p'],
            'repeat_penalty': llm_config['repeat_penalty']
        }
        
        self.confidence_threshold = llm_config['confidence_threshold']
        self.enable_gpu = llm_config['enable_gpu']
        self.gpu_backend = llm_config['gpu_backend']
        self.llm = None
        self._initialized = False
    
    async def _initialize(self):
        if self._initialized:
            return

        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp backend not available during model load.")
        if not self.model_path:
            raise RuntimeError("LLM model path is not configured.")

        # Auto-detect and configure GPU backend
        gpu_params = self._configure_gpu()
        final_params = {**self.llm_params, **gpu_params}

        try:
            self.llm = Llama(
                model_path=self.model_path,
                **final_params
            )
            self._initialized = True
            
            backend = final_params.get('backend', 'unknown')
            layers = final_params.get('n_gpu_layers', 0)
            try:
                layers_int = int(layers)
            except (TypeError, ValueError):
                layers_int = 0
            if layers_int != 0:
                logger.info(f"Loaded LLM model on GPU ({backend}, {layers_int} layers): {self.model_path}")
            else:
                logger.info(f"Loaded LLM model on CPU ({backend}): {self.model_path}")
                
        except Exception as e:
            # Fallback to CPU if GPU initialization fails
            layers = final_params.get('n_gpu_layers', 0)
            try:
                layers_int = int(layers)
            except (TypeError, ValueError):
                layers_int = 0
            if self.enable_gpu and layers_int != 0:
                logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
                fallback_params = {**self.llm_params, 'n_gpu_layers': 0}
                try:
                    self.llm = Llama(
                        model_path=self.model_path,
                        **fallback_params
                    )
                    self._initialized = True
                    logger.info(f"Loaded LLM model on CPU (fallback): {self.model_path}")
                except Exception as fallback_error:
                    raise RuntimeError(f"Failed to load LLM model on both GPU and CPU: {fallback_error}") from fallback_error
            else:
                raise RuntimeError(f"Failed to load LLM model: {e}") from e
    
    def _configure_gpu(self) -> Dict[str, Any]:
        """Configure GPU parameters based on backend detection"""
        gpu_params = {}
        
        if not self.enable_gpu or self.gpu_backend == "cpu":
            gpu_params['n_gpu_layers'] = 0
            gpu_params['backend'] = 'cpu'
            return gpu_params
        
        import platform
        
        # Auto-detect GPU backend
        if self.gpu_backend == "auto":
            system = platform.system()
            if system == "Darwin":  # macOS
                # Check for Apple Silicon
                machine = platform.machine()
                if machine in ["arm64", "aarch64"]:
                    detected_backend = "metal"
                else:
                    detected_backend = "cpu"  # Intel Mac
            else:
                # Check for NVIDIA GPU on Linux/Windows
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        detected_backend = "cuda"
                    else:
                        detected_backend = "cpu"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    detected_backend = "cpu"
        else:
            detected_backend = self.gpu_backend
        
        # Set GPU parameters based on detected backend
        if detected_backend == "metal":
            gpu_params['n_gpu_layers'] = self.llm_params['n_gpu_layers']
            logger.info("Using Metal GPU backend for Apple Silicon")
        elif detected_backend == "cuda":
            gpu_params['n_gpu_layers'] = self.llm_params['n_gpu_layers'] 
            logger.info("Using CUDA GPU backend for NVIDIA")
        else:
            gpu_params['n_gpu_layers'] = 0
            logger.info("Using CPU backend")
        
        gpu_params['backend'] = detected_backend
        
        return gpu_params
    
    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        start_time = time.time()
        await self._initialize()

        if not self.llm:
            raise RuntimeError("LLM model failed to load.")

        chunks = list(self._iter_chunks(content, self.config.chunk_size))
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
            if index >= self.max_chunks:
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
                    stop=["</s>", "[/INST]"],
                    **self.generation_params  # Use config-driven parameters
                )
            )
            
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
        self.strategies['llm'] = LLMExtractionStrategy(self.config)
        logger.info("LLM extraction strategy ready: %s", self.config.llm_model_path)
    
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
            if len(self.cache) < self.config.cache_size:  # Use config-driven cache size
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
