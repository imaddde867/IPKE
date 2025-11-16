"""LLM-driven knowledge extraction engine."""
from src.ai.llm_env_setup import *  # Must be first!

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import json
import time
import platform

# External dependencies (lazy loading)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig = None, None, None

from src.logging_config import get_logger
from src.core.unified_config import UnifiedConfig, get_config
from src.ai.chunkers import get_chunker

logger = get_logger(__name__)


# Core Data Models
@dataclass
class ExtractedEntity:
    content: str
    entity_type: str
    category: str
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractionResult:
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
    entities: List[ExtractedEntity] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)


class BaseExtractionStrategy(ABC):
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.llm_config = self.config.get_llm_config()
        self.max_chunks = max(0, self.llm_config['max_chunks'])
        self.confidence_threshold = self.llm_config['confidence_threshold']
        self._initialized = False

    @abstractmethod
    async def _initialize(self):
        pass

    @abstractmethod
    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        pass

    async def extract(self, content: str, document_type: str = "unknown") -> ExtractionResult:
        start_time = time.time()
        await self._initialize()

        chunker = get_chunker(self.config)
        chunk_start = time.time()
        chunk_objects = chunker.chunk(content)
        chunks = [chunk.text for chunk in chunk_objects]
        chunk_duration = time.time() - chunk_start

        # enforce max_chunks if configured to avoid concurrent llama.cpp calls blowing context
        if self.max_chunks > 0 and len(chunk_objects) > self.max_chunks:
            logger.info(
                "Limiting LLM processing to %d chunks (from %d) based on configuration.",
                self.max_chunks,
                len(chunk_objects),
            )
            chunk_objects = chunk_objects[: self.max_chunks]
            chunks = [chunk.text for chunk in chunk_objects]

        chunk_count = len(chunk_objects)
        avg_chunk_size = (
            sum(len(chunk.text) for chunk in chunk_objects) / chunk_count
            if chunk_count > 0 else 0
        )
        avg_sentences = (
            sum(max(1, chunk.sentence_span[1] - chunk.sentence_span[0]) for chunk in chunk_objects)
            / chunk_count if chunk_count else 0
        )
        cohesion_values = [
            chunk.meta.get("cohesion")
            for chunk in chunk_objects
            if isinstance(chunk.meta, dict) and isinstance(chunk.meta.get("cohesion"), (int, float))
        ]
        avg_cohesion = sum(cohesion_values) / len(cohesion_values) if cohesion_values else 0.0

        log_message = (
            f"Chunked document using {self.config.chunking_method} method: "
            f"{chunk_count} chunks, avg size {avg_chunk_size:.2f} chars, "
            f"avg sentences {avg_sentences:.2f}, avg cohesion {avg_cohesion:.2f}, "
            f"chunk_time {chunk_duration:.3f}s"
        ) if chunk_count else f"Chunked document using {self.config.chunking_method} method: no content"

        logger.info(
            log_message,
            extra={
                "chunking_method": self.config.chunking_method,
                "chunk_count": chunk_count,
                "avg_chunk_size": round(avg_chunk_size, 2) if chunk_count else 0,
                "avg_sentences_per_chunk": round(avg_sentences, 2) if chunk_count else 0,
                "avg_chunk_cohesion": round(avg_cohesion, 2) if cohesion_values else 0,
                "chunking_duration": round(chunk_duration, 4),
            }
        )

        chunk_results = []
        for chunk in chunks:
            result = await self._process_chunk_with_llm(chunk, document_type)
            chunk_results.append(result)

        all_entities: List[ExtractedEntity] = []
        raw_steps: List[Dict[str, Any]] = []
        raw_constraints: List[Dict[str, Any]] = []

        for result in chunk_results:
            if result:
                all_entities.extend(result.entities)
                raw_steps.extend(result.steps)
                raw_constraints.extend(result.constraints)

        normalized_steps, step_id_map = self._normalize_steps(raw_steps)
        normalized_constraints = self._normalize_constraints(raw_constraints, step_id_map)

        processing_time = time.time() - start_time
        confidence_score = sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0

        return ExtractionResult(
            entities=all_entities,
            steps=normalized_steps,
            constraints=normalized_constraints,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used=self.get_strategy_name(),
            quality_metrics={
                'entity_count': len(all_entities),
                'avg_confidence': confidence_score,
                'chunk_count': chunk_count,
                'chunks_processed': len(chunks),
                'avg_chunk_size': round(avg_chunk_size, 2) if chunk_count else 0,
                'avg_sentences_per_chunk': round(avg_sentences, 2) if chunk_count else 0,
                'avg_chunk_cohesion': round(avg_cohesion, 2) if chunk_count else 0,
                'chunking_duration': round(chunk_duration, 4),
                'chunking_method': self.config.chunking_method,
            }
        )

    def _parse_llm_response(self, response: str, chunk: str) -> ChunkExtraction:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return ChunkExtraction()

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            raw_steps = self._ensure_dict_list(data.get("steps"))
            raw_constraints = self._ensure_dict_list(data.get("constraints"))
            raw_entities = self._ensure_dict_list(data.get("entities"))

            entities = self._build_entities(raw_entities, chunk)
            return ChunkExtraction(entities=entities, steps=raw_steps, constraints=raw_constraints)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}\nResponse: {response[:200]}...")
            return ChunkExtraction()

    def _build_entities(self, items: List[Dict[str, Any]], chunk: str) -> List[ExtractedEntity]:
        entities = []
        for item in items:
            content = (item.get("content") or item.get("text") or "").strip()
            if not content:
                continue
            entities.append(ExtractedEntity(
                content=content,
                entity_type=item.get("type", "unknown"),
                category=item.get("category", "general"),
                confidence=self._coerce_confidence(item.get("confidence")),
                context=chunk[:200] + "...",
                metadata={'llm_extracted': True, 'source': self.get_strategy_name()}
            ))
        return entities

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(self.confidence_threshold)

    def _normalize_steps(self, steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        normalized, id_map = [], {}
        for i, step in enumerate(steps):
            content = (step.get("text") or "").strip()
            if not content:
                continue
            new_id = f"S{i + 1}"
            original_id = str(step.get("id") or new_id)
            id_map[original_id] = new_id
            normalized.append({
                "id": new_id,
                "text": content,
                "order": i + 1,
                "confidence": self._coerce_confidence(step.get("confidence")),
            })
        return normalized, id_map

    def _normalize_constraints(self, constraints: List[Dict[str, Any]], step_id_map: Dict[str, str]) -> List[Dict[str, Any]]:
        normalized = []
        for i, constraint in enumerate(constraints):
            text = (constraint.get("text") or "").strip()
            if not text:
                continue
            raw_refs = constraint.get("steps", [])
            attached_steps = [step_id_map[ref] for ref in raw_refs if ref in step_id_map]
            normalized.append({
                "id": f"C{i + 1}",
                "text": text,
                "confidence": self._coerce_confidence(constraint.get("confidence")),
                "steps": attached_steps,
            })
        return normalized

    @staticmethod
    def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, list) and all(isinstance(i, dict) for i in value):
            return value
        return []

    def get_confidence_threshold(self) -> float:
        return self.confidence_threshold

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass


class LlamaCppStrategy(BaseExtractionStrategy):
    """Strategy using llama-cpp-python, optimized for Metal on Apple Silicon."""
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        super().__init__(config)
        self.llm = None
        self.generation_params = {
            'max_tokens': self.llm_config['max_tokens'],
            'temperature': self.llm_config['temperature'],
            'top_p': self.llm_config['top_p'],
            'repeat_penalty': self.llm_config['repeat_penalty']
        }

    async def _initialize(self):
        if self._initialized:
            return
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed. Please install it for the 'metal' backend.")
        
        params = {
            'model_path': self.llm_config['model_path'],
            'n_ctx': self.llm_config['n_ctx'],
            'n_threads': self.llm_config['n_threads'],
            'n_gpu_layers': self.llm_config['n_gpu_layers'],
            'f16_kv': self.llm_config['f16_kv'],
            'use_mlock': self.llm_config['use_mlock'],
            'use_mmap': self.llm_config['use_mmap'],
            'verbose': self.llm_config['verbose']
        }
        
        try:
            self.llm = Llama(**params)
            self._initialized = True
            logger.info(f"Initialized LlamaCppStrategy with model: {params['model_path']}")
        except Exception as e:
            logger.error(f"Failed to initialize Llama.cpp model: {e}")
            raise RuntimeError("Failed to load Llama.cpp model") from e

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self._create_prompt(chunk, document_type)
        loop = asyncio.get_running_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(prompt, stop=["</s>", "[/INST]"], **self.generation_params)
            )
            return self._parse_llm_response(response['choices'][0]['text'], chunk)
        except Exception as e:
            logger.warning(f"Llama.cpp processing failed: {e}")
            return ChunkExtraction()

    def get_strategy_name(self) -> str:
        return "llama.cpp"

    @staticmethod
    def _create_prompt(chunk: str, document_type: str) -> str:
        return f"""[INST] You produce lightweight procedural structure from {document_type} documents.

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


class TransformersStrategy(BaseExtractionStrategy):
    """Strategy using Hugging Face Transformers, optimized for CUDA."""

    def __init__(self, config: Optional[UnifiedConfig] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.cpu_max_tokens_cap = self.llm_config.get("cpu_max_tokens_cap", 512)
        self.generation_params = {
            'max_new_tokens': self.llm_config['max_tokens'],
            'temperature': self.llm_config['temperature'],
            'top_p': self.llm_config['top_p'],
            'repetition_penalty': self.llm_config['repeat_penalty'],
            'do_sample': True,
        }

    def _resolve_device(self) -> str:
        """Determine the target device for inference."""
        if torch.cuda.is_available() and self.llm_config['enable_gpu']:
            return "cuda"
        return "cpu"

    async def _initialize(self):
        if self._initialized:
            return
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Hugging Face transformers library is not installed. Please install it for the 'cuda' backend.")

        # Force single-threaded mode for tokenizers/OMP
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = self._resolve_device()
        logger.info(f"Initializing TransformersStrategy on device: {self.device}")
        if self.device == "cpu" and self.cpu_max_tokens_cap and self.cpu_max_tokens_cap > 0:
            capped_tokens = min(self.generation_params['max_new_tokens'], self.cpu_max_tokens_cap)
            if capped_tokens < self.generation_params['max_new_tokens']:
                logger.info(
                    "Clamping max_new_tokens from %d to %d for CPU inference (override via LLM_CPU_MAX_TOKENS_CAP).",
                    self.generation_params['max_new_tokens'],
                    capped_tokens,
                )
                self.generation_params['max_new_tokens'] = capped_tokens

        # Small delay to ensure env vars propagate in native libs
        await asyncio.sleep(0.1)

        quant_config = None
        if self.device == "cuda":
            if self.llm_config['quantization'] == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif self.llm_config['quantization'] == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            # Disable parallelism to avoid mutex issues on macOS
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_config['model_id'],
                use_fast=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_config['model_id'],
                quantization_config=quant_config,
                device_map=None,
                dtype="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            if self.device == "cpu":
                self.model.to(self.device)
                logger.info(f"Moved model to {self.device}")
            self._initialized = True
            logger.info(f"Loaded Transformers model '{self.llm_config['model_id']}' on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Transformers model: {e}") from e

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = LlamaCppStrategy._create_prompt(chunk, document_type)
        loop = asyncio.get_running_loop()

        def generate():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **self.generation_params)
            # Decode only the generated part, excluding the prompt
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        try:
            response_text = await loop.run_in_executor(None, generate)
            return self._parse_llm_response(response_text, chunk)
        except Exception as e:
            logger.warning(f"Transformers processing failed: {e}")
            return ChunkExtraction()

    def get_strategy_name(self) -> str:
        return "transformers"


class UnifiedKnowledgeEngine:
    """LLM-only knowledge extraction pipeline with selectable backend."""
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self._strategy: Optional[BaseExtractionStrategy] = None
        self.cache = {}
        self.cache_lock = asyncio.Lock()
        self.performance_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'strategy_usage': {},
            'avg_processing_time': 0.0
        }

    @property
    def strategy(self) -> BaseExtractionStrategy:
        """Lazy-load the extraction strategy on first access."""
        if self._strategy is None:
            self._initialize_strategy()
        return self._strategy

    @property
    def strategies(self) -> List[BaseExtractionStrategy]:
        """Backward-compatible accessor returning the active strategy as a list."""
        if self._strategy is None:
            try:
                return [self.strategy]
            except RuntimeError:
                return []
        return [self._strategy]

    def _initialize_strategy(self):
        backend = self.config.detect_gpu_backend()
        logger.info(f"Detected GPU backend: {backend}. Initializing corresponding strategy.")
        # Note: On macOS Metal, we run the LLM on CPU to avoid tokenizer mutex issues.
        # MPS acceleration will be reserved for embedding workloads where we control parallelism.

        if backend == "metal":
            # On Apple Silicon, prefer llama.cpp on Metal / CPU to avoid tokenizer mutex issues.
            if LLAMA_CPP_AVAILABLE:
                self._strategy = LlamaCppStrategy(self.config)
                logger.info("Using LlamaCppStrategy on Metal backend.")
            elif TRANSFORMERS_AVAILABLE:
                self._strategy = TransformersStrategy(self.config)
                logger.info("Fallback to TransformersStrategy on CPU for Metal backend.")
            else:
                raise RuntimeError(
                    "No suitable LLM backend found. Please install 'llama-cpp-python' or 'transformers'."
                )
        elif backend == "cuda":
            self._strategy = TransformersStrategy(self.config)
            logger.info("Using TransformersStrategy for CUDA backend.")
        else: # cpu
            # Prefer transformers on CPU if available, otherwise fallback to llama.cpp
            if TRANSFORMERS_AVAILABLE:
                self._strategy = TransformersStrategy(self.config)
                logger.info("Using TransformersStrategy for CPU backend.")
            elif LLAMA_CPP_AVAILABLE:
                self._strategy = LlamaCppStrategy(self.config)
                logger.info("Transformers not found. Using LlamaCppStrategy for CPU backend.")
            else:
                raise RuntimeError("No suitable LLM backend found. Please install either 'transformers' or 'llama-cpp-python'.")

    async def extract_knowledge(
        self, 
        content: str, 
        document_type: str = "unknown",
        quality_threshold: Optional[float] = None
    ) -> ExtractionResult:
        strategy = self.strategy
        if not strategy:
            raise RuntimeError("Knowledge extraction strategy not initialized.")
            
        final_quality_threshold = quality_threshold if quality_threshold is not None else self.config.quality_threshold
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{final_quality_threshold}"

        async with self.cache_lock:
            if cache_key in self.cache:
                self.performance_stats['cache_hits'] += 1
                return self.cache[cache_key]

        result = await strategy.extract(content, document_type)

        if result.confidence_score < final_quality_threshold:
            logger.warning(
                "Extraction confidence below threshold",
                extra={"confidence_score": result.confidence_score, "quality_threshold": final_quality_threshold}
            )

        self._update_stats(result)

        async with self.cache_lock:
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = result

        logger.info(f"Extracted {len(result.entities)} entities using {result.strategy_used} "
                   f"in {result.processing_time:.2f}s (confidence: {result.confidence_score:.2f})")
        
        return result

    def _update_stats(self, result: ExtractionResult):
        stats = self.performance_stats
        total_before = stats['total_extractions']
        stats['total_extractions'] = total_before + 1
        stats['avg_processing_time'] = (
            (stats['avg_processing_time'] * total_before + result.processing_time)
            / (total_before + 1)
        )
        strategy_name = result.strategy_used
        stats['strategy_usage'][strategy_name] = stats['strategy_usage'].get(strategy_name, 0) + 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_stats.copy()
    
    async def clear_cache(self):
        async with self.cache_lock:
            self.cache.clear()
            logger.info("Extraction cache cleared")
