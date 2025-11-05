"""LLM-driven knowledge extraction engine."""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
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
    steps: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkExtraction:
    """Intermediate structure returned per LLM chunk."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)


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
        self.max_chunks = max(0, llm_config['max_chunks'])
        
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
    
    def _load_llm_model(self, params: Dict[str, Any]):
        """Instantiate the llama.cpp model with shared parameters."""
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama_cpp backend not available during model load.")
        return Llama(model_path=self.model_path, **params)
    
    def _try_load_gpu(self, base_params: Dict[str, Any], requested_layers: Any, default_backend: str):
        """Attempt to load the model on GPU with progressive layer counts."""
        last_error: Optional[Exception] = None
        for layers in self._generate_gpu_layer_candidates(requested_layers):
            candidate_params = {**base_params, 'n_gpu_layers': layers}
            try:
                self.llm = self._load_llm_model(candidate_params)
                self._initialized = True
                logger.info(
                    "Loaded LLM model on GPU (%s, %s layers): %s",
                    candidate_params.get('backend', default_backend),
                    "all" if layers == -1 else layers,
                    self.model_path
                )
                return True, None
            except Exception as gpu_error:
                last_error = gpu_error
                logger.warning(
                    "GPU initialization attempt failed (backend=%s, layers=%s): %s",
                    candidate_params.get('backend', default_backend),
                    layers,
                    gpu_error
                )
        return False, last_error
    
    async def _initialize(self):
        if self._initialized:
            return

        if not self.model_path:
            raise RuntimeError("LLM model path is not configured.")

        # Auto-detect and configure GPU backend
        gpu_params = self._configure_gpu()
        final_params = {**self.llm_params, **gpu_params}

        backend = final_params.get('backend', 'unknown')
        requested_layers = final_params.get('n_gpu_layers', 0)
        gpu_attempted = False
        last_gpu_error: Optional[Exception] = None

        if self.enable_gpu and requested_layers not in (0, None):
            gpu_attempted = True
            loaded, last_gpu_error = self._try_load_gpu(final_params, requested_layers, backend)
            if loaded:
                return

        # Fallback to CPU if GPU attempts failed or were not allowed
        fallback_params = {**self.llm_params}
        fallback_params['n_gpu_layers'] = 0
        fallback_params['backend'] = 'cpu'
        fallback_params['use_mlock'] = False  # avoid mlock on restricted environments
        fallback_params['use_mmap'] = False   # mmap can fail on network filesystems

        try:
            self.llm = self._load_llm_model(fallback_params)
            self._initialized = True
            logger.info("Loaded LLM model on CPU (fallback): %s", self.model_path)
        except Exception as fallback_error:
            if gpu_attempted and last_gpu_error:
                raise RuntimeError(
                    f"Failed to load LLM model on GPU (last error: {last_gpu_error}) "
                    f"and CPU fallback: {fallback_error}"
                ) from fallback_error
            raise RuntimeError(f"Failed to load LLM model: {fallback_error}") from fallback_error

    def _generate_gpu_layer_candidates(self, requested_layers: Any) -> List[int]:
        """Yield progressively smaller GPU layer counts for constrained devices."""
        try:
            layers = int(requested_layers)
        except (TypeError, ValueError):
            layers = -1

        if layers == 0:
            return []

        candidates: List[int] = []
        if layers == -1:
            candidates = [-1, 80, 64, 48, 40, 32, 24, 16, 12, 8, 4, 2]
        else:
            current = max(layers, 1)
            while current > 0:
                candidates.append(current)
                if current == 1:
                    break
                current = max(current // 2, 1)
            if -1 not in candidates and layers > 0:
                candidates.insert(0, layers)

        # Ensure unique ordering
        seen = set()
        unique_candidates = []
        for value in candidates:
            if value not in seen:
                unique_candidates.append(value)
                seen.add(value)
        return unique_candidates
    
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
        chunk_results: List[ChunkExtraction] = []

        for chunk in chunks:
            chunk_result = await self._process_chunk_with_llm(chunk, document_type)
            chunk_results.append(chunk_result)

        all_entities: List[ExtractedEntity] = []
        raw_steps: List[Dict[str, Any]] = []
        raw_constraints: List[Dict[str, Any]] = []

        for chunk_result in chunk_results:
            all_entities.extend(chunk_result.entities)
            raw_steps.extend(chunk_result.steps)
            raw_constraints.extend(chunk_result.constraints)

        normalized_steps, step_id_map = self._normalize_steps(raw_steps)
        if not normalized_steps:
            normalized_steps, step_id_map = self._steps_from_entities(all_entities)

        normalized_constraints = self._normalize_constraints(raw_constraints, step_id_map)

        if not all_entities and normalized_steps:
            all_entities.extend(
                self._entities_from_structured_steps(normalized_steps, content)
            )

        processing_time = time.time() - start_time
        confidence_score = sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0

        return ExtractionResult(
            entities=all_entities,
            steps=normalized_steps,
            constraints=normalized_constraints,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used="llm_default",
            quality_metrics={
                'entity_count': len(all_entities),
                'avg_confidence': confidence_score,
                'chunks_processed': len(chunks),
                'step_count': len(normalized_steps),
                'constraint_count': len(normalized_constraints),
            }
        )
    
    def _iter_chunks(self, content: str, chunk_size: int):
        """Yield limited chunks, biased toward sentence boundaries."""
        for index, start in enumerate(range(0, len(content), chunk_size)):
            if self.max_chunks and index >= self.max_chunks:
                break
            chunk = content[start:start + chunk_size]
            if start + chunk_size < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]
            yield chunk
    
    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = f"""[INST] You produce lightweight procedural structure from {document_type} documents.

Read the provided text and return a SINGLE JSON object with concise fields:
{{
  "steps": [
    {{"id": "S1", "text": "Action statement written as an imperative.", "type": "procedure_step", "confidence": 0.9}},
    {{"id": "S2", "text": "Next ordered action.", "type": "procedure_step", "confidence": 0.88}}
  ],
  "constraints": [
    {{"id": "C1", "text": "Condition, warning, or requirement.", "steps": ["S1"], "confidence": 0.85}}
  ],
  "entities": [
    {{"content": "Supporting fact or measurement.", "type": "specification", "category": "distance_requirement", "confidence": 0.9}}
  ]
}}

Guidance:
- Keep IDs sequential (S1, S2, ..., C1, C2, ...).
- Steps must be actual actions or instructions in execution order.
- Constraints capture requirements, cautions, or prerequisites. Refer to step IDs when possible.
- Include additional granular facts in `entities` when helpful (measurements, tools, materials).
- Prefer short phrases; avoid restating large paragraphs.

Text:
\"\"\"{chunk}\"\"\"

Return only the JSON object described above with no commentary.
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
            return ChunkExtraction()
    
    def _parse_llm_response(self, response: str, chunk: str) -> ChunkExtraction:
        extraction = ChunkExtraction()
        try:
            # Extract JSON from response
            json_start = response.find('[')
            obj_start = response.find('{')
            json_end = response.rfind(']') + 1
            obj_end = response.rfind('}') + 1

            json_str = None
            if obj_start != -1 and obj_end > obj_start:
                json_str = response[obj_start:obj_end]
            elif json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]

            if json_str:
                data = json.loads(json_str)

                if isinstance(data, dict):
                    raw_steps = self._ensure_dict_list(data.get("steps"))
                    raw_constraints = self._ensure_dict_list(data.get("constraints"))
                    raw_entities = self._ensure_dict_list(data.get("entities"))

                    entities = self._build_entities(raw_entities, chunk)
                    extraction = ChunkExtraction(
                        entities=entities,
                        steps=raw_steps,
                        constraints=raw_constraints,
                    )
                elif isinstance(data, list):
                    entities = self._build_entities(data, chunk)
                    step_candidates = [
                        item for item in data
                        if isinstance(item, dict) and item.get("type") in {"procedure_step", "process_sequence"}
                    ]
                    extraction = ChunkExtraction(
                        entities=entities,
                        steps=self._ensure_dict_list(step_candidates),
                        constraints=[],
                    )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return extraction
    
    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold
    
    def get_strategy_name(self) -> str:
        return "LLM-Powered Extraction"

    @staticmethod
    def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
        if not value:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _build_entities(self, items: List[Dict[str, Any]], chunk: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        for item in items:
            content = (item.get("content") or item.get("text") or "").strip()
            if not content:
                continue
            entity_type = item.get("type") or item.get("entity_type") or "unknown"
            category = item.get("category") or item.get("type") or "general"
            confidence = self._coerce_confidence(item.get("confidence"))
            entities.append(
                ExtractedEntity(
                    content=content,
                    entity_type=entity_type,
                    category=category,
                    confidence=confidence,
                    context=chunk[:200] + "...",
                    metadata={'llm_extracted': True, 'source': 'llama'}
                )
            )
        return entities

    def _entities_from_structured_steps(self, steps: List[Dict[str, Any]], document_text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        for step in steps:
            content = (step.get("text") or step.get("description") or "").strip()
            if not content:
                continue
            confidence = self._coerce_confidence(step.get("confidence"))
            entities.append(
                ExtractedEntity(
                    content=content,
                    entity_type="procedure_step",
                    category="procedure",
                    confidence=confidence,
                    context=document_text[:200] + "...",
                    metadata={'llm_extracted': True, 'source': 'llama'}
                )
            )
        return entities

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(self.confidence_threshold)
        return max(0.0, min(1.0, confidence))

    def _normalize_steps(self, steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        normalized: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}

        for step in steps:
            content = (step.get("text") or step.get("description") or step.get("summary") or step.get("content") or "").strip()
            if not content:
                continue
            new_id = f"S{len(normalized) + 1}"
            original_id = str(step.get("id") or new_id)
            id_map[original_id] = new_id
            id_map[new_id] = new_id

            normalized_step = {
                "id": new_id,
                "text": content,
                "order": len(normalized) + 1,
                "confidence": self._coerce_confidence(step.get("confidence")),
            }

            if step.get("type"):
                normalized_step["type"] = step["type"]
            if step.get("inputs"):
                normalized_step["inputs"] = step["inputs"]
            if step.get("outputs"):
                normalized_step["outputs"] = step["outputs"]

            normalized.append(normalized_step)

        return normalized, id_map

    def _steps_from_entities(
        self,
        entities: List[ExtractedEntity],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        steps: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}

        for entity in entities:
            if entity.entity_type not in {"procedure_step", "process_sequence", "maintenance_instruction"}:
                continue
            content = entity.content.strip()
            if not content:
                continue
            new_id = f"S{len(steps) + 1}"
            id_map[new_id] = new_id
            steps.append(
                {
                    "id": new_id,
                    "text": content,
                    "order": len(steps) + 1,
                    "confidence": self._coerce_confidence(entity.confidence),
                    "type": "procedure_step",
                }
            )

        return steps, id_map

    def _normalize_constraints(
        self,
        constraints: List[Dict[str, Any]],
        step_id_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for constraint in constraints:
            text = (constraint.get("text") or constraint.get("description") or "").strip()
            if not text:
                continue
            new_id = f"C{len(normalized) + 1}"

            raw_refs = constraint.get("steps") or constraint.get("attached_to") or constraint.get("scope") or []
            if isinstance(raw_refs, str):
                raw_refs = [raw_refs]
            if isinstance(raw_refs, dict):
                raw_refs = [raw_refs.get("id")] if raw_refs.get("id") else []

            attached_steps: List[str] = []
            for ref in raw_refs or []:
                if not ref:
                    continue
                if isinstance(ref, dict):
                    candidate = ref.get("id")
                    if candidate and candidate in step_id_map:
                        attached_steps.append(step_id_map[candidate])
                elif isinstance(ref, str):
                    candidate = ref.strip()
                    if candidate in step_id_map:
                        attached_steps.append(step_id_map[candidate])

            normalized_constraint: Dict[str, Any] = {
                "id": new_id,
                "text": text,
                "confidence": self._coerce_confidence(constraint.get("confidence")),
            }
            if attached_steps:
                normalized_constraint["steps"] = attached_steps
            normalized.append(normalized_constraint)

        return normalized


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
