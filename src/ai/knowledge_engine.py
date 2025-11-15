"""LLM-driven knowledge extraction engine."""

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
from src.core.unified_config import UnifiedConfig, get_config, Environment
from src.ai.chunkers import Chunk, FixedChunker, get_chunker

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
        self.parallel_instances = max(1, int(self.llm_config.get('parallel_instances', 1)))
        self.chunker = get_chunker(self.config)
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

        chunk_objects = self._chunk_content(content)
        if self.max_chunks > 0:
            chunk_objects = chunk_objects[:self.max_chunks]
        chunk_texts = [chunk.text for chunk in chunk_objects] or ([content] if content else [])

        if not chunk_texts:
            chunk_results = []
        else:
            max_parallel = min(self.parallel_instances, len(chunk_texts))
            semaphore = asyncio.Semaphore(max_parallel or 1)

            async def process_chunk(chunk: str) -> ChunkExtraction:
                async with semaphore:
                    return await self._process_chunk_with_llm(chunk, document_type)

            chunk_results = await asyncio.gather(*(process_chunk(chunk) for chunk in chunk_texts))

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
        chunk_stats = self._compute_chunk_stats(chunk_objects, len(chunk_texts))
        metadata: Dict[str, Any] = {"chunking": chunk_stats}
        if self.config.debug_chunking:
            metadata["chunks"] = [
                {
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "sentence_span": chunk.sentence_span,
                    "meta": chunk.meta,
                }
                for chunk in chunk_objects
            ]

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
                'chunks_processed': len(chunk_texts),
            },
            metadata=metadata,
        )
    def _chunk_content(self, content: str) -> List[Chunk]:
        if not content:
            return []
        try:
            chunks = self.chunker.chunk(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Chunker '%s' failed (%s). Falling back to fixed-size splitting.",
                self.config.chunking_method,
                exc,
            )
            fallback = FixedChunker(max_chars=self.config.chunk_max_chars)
            chunks = fallback.chunk(content)
        if not chunks:
            return [
                Chunk(
                    text=content,
                    start_char=0,
                    end_char=len(content),
                    sentence_span=(-1, -1),
                    meta={
                        "chunking_method": "fallback",
                        "sentence_count": None,
                        "cohesion": None,
                        "char_length": len(content),
                    },
                )
            ]
        return chunks

    def _compute_chunk_stats(self, chunks: List[Chunk], processed_count: int) -> Dict[str, Any]:
        if not chunks:
            return {
                "method": self.config.chunking_method,
                "count": 0,
                "avg_sentences": 0.0,
                "avg_chars": 0.0,
                "avg_cohesion": None,
            }
        sentence_counts = [
            chunk.meta.get("sentence_count")
            for chunk in chunks
            if chunk.meta.get("sentence_count") is not None
        ]
        cohesions = [
            chunk.meta.get("cohesion")
            for chunk in chunks
            if chunk.meta.get("cohesion") is not None
        ]
        avg_sentences = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0.0
        avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks)
        avg_cohesion = (
            sum(cohesions) / len(cohesions) if cohesions else None
        )
        return {
            "method": chunks[0].meta.get("chunking_method", self.config.chunking_method),
            "count": processed_count,
            "avg_sentences": round(avg_sentences, 2),
            "avg_chars": round(avg_chars, 2),
            "avg_cohesion": round(avg_cohesion, 3) if avg_cohesion is not None else None,
        }

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
        self.pool_size = max(1, self.parallel_instances)
        self._llm_pool: List[Any] = []
        self._instance_queue: Optional[asyncio.Queue] = None
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
            self._llm_pool = []
            for idx in range(self.pool_size):
                instance = Llama(**params)
                self._llm_pool.append(instance)
                logger.debug(f"Loaded llama.cpp instance {idx + 1}/{self.pool_size}")

            self.llm = self._llm_pool[0]
            if self.pool_size > 1:
                self._instance_queue = asyncio.Queue(maxsize=self.pool_size)
                for instance in self._llm_pool:
                    self._instance_queue.put_nowait(instance)

            self._initialized = True
            logger.info(
                f"Initialized {self.pool_size} llama.cpp instance(s) with model: {params['model_path']}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Llama.cpp model: {e}")
            raise RuntimeError("Failed to load Llama.cpp model") from e

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = self._create_prompt(chunk, document_type)
        loop = asyncio.get_running_loop()

        llm_instance = await self._acquire_llm_instance()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: llm_instance(prompt, stop=["</s>", "[/INST]"], **self.generation_params)
            )
            return self._parse_llm_response(response['choices'][0]['text'], chunk)
        except Exception as e:
            logger.warning(f"Llama.cpp processing failed: {e}")
            return ChunkExtraction()
        finally:
            await self._release_llm_instance(llm_instance)

    def get_strategy_name(self) -> str:
        return "llama.cpp"

    async def _acquire_llm_instance(self):
        if self.pool_size == 1 or not self._instance_queue:
            return self.llm
        return await self._instance_queue.get()

    async def _release_llm_instance(self, instance):
        if self.pool_size == 1 or not self._instance_queue:
            return
        await self._instance_queue.put(instance)

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
        self.generation_params = {
            'max_new_tokens': self.llm_config['max_tokens'],
            'temperature': self.llm_config['temperature'],
            'top_p': self.llm_config['top_p'],
            'repetition_penalty': self.llm_config['repeat_penalty'],
            'do_sample': True,
        }

    async def _initialize(self):
        if self._initialized:
            return
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Hugging Face transformers library is not installed. Please install it for the 'cuda' backend.")

        device = "cuda" if torch.cuda.is_available() and self.llm_config['enable_gpu'] else "cpu"
        logger.info(f"Initializing TransformersStrategy on device: {device}")

        quant_config = None
        if device == "cuda":
            if self.llm_config['quantization'] == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif self.llm_config['quantization'] == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config['model_id'])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_config['model_id'],
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                self.model.to(device)
            self._initialized = True
            logger.info(f"Loaded Transformers model '{self.llm_config['model_id']}' on {self.model.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Transformers model: {e}") from e

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        prompt = LlamaCppStrategy._create_prompt(chunk, document_type)
        loop = asyncio.get_running_loop()

        def generate():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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


class MockExtractionStrategy(BaseExtractionStrategy):
    """Deterministic strategy for testing environments (no model dependency)."""

    async def _initialize(self):
        self._initialized = True

    async def _process_chunk_with_llm(self, chunk: str, document_type: str) -> ChunkExtraction:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        steps = []
        for idx, line in enumerate(lines[:5]):
            steps.append({
                "id": f"S{idx + 1}",
                "text": line,
                "confidence": 0.8,
            })
        entities = [
            ExtractedEntity(
                content=line,
                entity_type="fact",
                category=document_type or "general",
                confidence=0.85,
                context=line[:120] + "...",
                metadata={"llm_extracted": False, "source": "mock"},
            )
            for line in lines[:3]
        ]
        constraints = [
            {"id": "C1", "text": "Mock constraint", "confidence": 0.7, "steps": ["S1"]},
        ] if steps else []
        return ChunkExtraction(entities=entities, steps=steps, constraints=constraints)

    def get_strategy_name(self) -> str:
        return "mock"


class UnifiedKnowledgeEngine:
    """LLM-only knowledge extraction pipeline with selectable backend."""
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.strategy: Optional[BaseExtractionStrategy] = None
        self.cache = {}
        self.cache_lock = asyncio.Lock()
        self.performance_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'strategy_usage': {},
            'avg_processing_time': 0.0
        }
        self._initialize_strategy()

    @property
    def strategies(self) -> List[BaseExtractionStrategy]:
        """Backward-compatible accessor returning the active strategy as a list."""
        return [self.strategy] if self.strategy is not None else []

    def _initialize_strategy(self):
        backend = self.config.detect_gpu_backend()
        logger.info(f"Detected GPU backend: {backend}. Initializing corresponding strategy.")

        if self.config.environment == Environment.TESTING:
            self.strategy = MockExtractionStrategy(self.config)
            logger.info("Using MockExtractionStrategy for testing backend.")
            return

        if backend == "metal":
            self.strategy = LlamaCppStrategy(self.config)
            logger.info("Using LlamaCppStrategy for Metal backend.")
        elif backend == "cuda":
            self.strategy = TransformersStrategy(self.config)
            logger.info("Using TransformersStrategy for CUDA backend.")
        else: # cpu
            # Prefer transformers on CPU if available, otherwise fallback to llama.cpp
            if TRANSFORMERS_AVAILABLE:
                self.strategy = TransformersStrategy(self.config)
                logger.info("Using TransformersStrategy for CPU backend.")
            elif LLAMA_CPP_AVAILABLE:
                self.strategy = LlamaCppStrategy(self.config)
                logger.info("Transformers not found. Using LlamaCppStrategy for CPU backend.")
            else:
                raise RuntimeError("No suitable LLM backend found. Please install either 'transformers' or 'llama-cpp-python'.")

    async def extract_knowledge(
        self, 
        content: str, 
        document_type: str = "unknown",
        quality_threshold: Optional[float] = None
    ) -> ExtractionResult:
        if not self.strategy:
            raise RuntimeError("Knowledge extraction strategy not initialized.")
            
        final_quality_threshold = quality_threshold if quality_threshold is not None else self.config.quality_threshold
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{final_quality_threshold}"

        async with self.cache_lock:
            if cache_key in self.cache:
                self.performance_stats['cache_hits'] += 1
                return self.cache[cache_key]

        result = await self.strategy.extract(content, document_type)

        if result.confidence_score < final_quality_threshold:
            logger.warning(
                "Extraction confidence below threshold",
                extra={"confidence_score": result.confidence_score, "quality_threshold": final_quality_threshold}
            )

        self._update_stats(result)

        async with self.cache_lock:
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = result

        chunk_meta = result.metadata.get("chunking", {}) if result.metadata else {}
        logger.info(
            "Extracted %s entities using %s in %.2fs (confidence: %.2f) | chunking=%s count=%s avg_sent=%s avg_cohesion=%s",
            len(result.entities),
            result.strategy_used,
            result.processing_time,
            result.confidence_score,
            chunk_meta.get("method"),
            chunk_meta.get("count"),
            chunk_meta.get("avg_sentences"),
            chunk_meta.get("avg_cohesion"),
        )
        
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
