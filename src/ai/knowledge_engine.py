"""LLM-driven knowledge extraction engine."""
from src.ai.llm_env_setup import *  # Must be first!

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from src.ai.types import ChunkExtraction, ExtractedEntity, ExtractionResult
from src.ai.worker_pool import ChunkTask, LLMWorkerPool
from src.core.unified_config import UnifiedConfig, get_config
from src.logging_config import get_logger
from src.processors.chunkers import get_chunker

logger = get_logger(__name__)


class UnifiedKnowledgeEngine:
    """Parallel, prompt-aware knowledge extraction pipeline."""

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.cache: Dict[str, ExtractionResult] = {}
        self.cache_lock = asyncio.Lock()
        self.performance_stats = {
            "total_extractions": 0,
            "cache_hits": 0,
            "strategy_usage": {},
            "avg_processing_time": 0.0,
        }
        self._pool: Optional[LLMWorkerPool] = None
        self._backend_name: Optional[str] = None

    def _determine_backend(self) -> str:
        requested = getattr(self.config, "llm_backend", None)
        if requested:
            return requested
        backend = self.config.detect_gpu_backend()
        if backend == "cuda":
            return "llama_cpp"
        return "transformers"

    def _ensure_pool(self):
        if self._pool is not None:
            return
        backend_name = self._determine_backend()
        worker_count = max(1, getattr(self.config, "llm_num_workers", self.config.max_workers))
        device_strategy = getattr(self.config, "llm_device_strategy", "single")
        self._pool = LLMWorkerPool(
            config=self.config,
            backend_name=backend_name,
            num_workers=worker_count,
            device_strategy=device_strategy,
            timeout=self.config.processing_timeout,
        )
        self._backend_name = backend_name
        logger.info(
            "Initialized LLM worker pool backend=%s workers=%s device_strategy=%s prompting=%s",
            backend_name,
            worker_count,
            device_strategy,
            getattr(self.config, "prompting_strategy", "P0"),
        )

    async def _run_chunks(self, chunk_tasks: List[ChunkTask]) -> List[ChunkExtraction]:
        if not chunk_tasks:
            return []
        self._ensure_pool()
        assert self._pool is not None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._pool.process, chunk_tasks)

    async def extract_knowledge(
        self, content: str, document_type: str = "unknown", quality_threshold: Optional[float] = None
    ) -> ExtractionResult:
        final_threshold = quality_threshold if quality_threshold is not None else self.config.quality_threshold
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{document_type}_{final_threshold}"

        async with self.cache_lock:
            cached = self.cache.get(cache_key)
            if cached:
                self.performance_stats["cache_hits"] += 1
                return cached

        start_time = time.time()
        chunker = get_chunker(self.config)
        chunk_start = time.time()
        chunk_objects = chunker.chunk(content)
        chunk_duration = time.time() - chunk_start

        chunk_metrics = self._summarize_chunks(chunk_objects, chunk_duration)
        if chunk_metrics:
            logger.info(chunk_metrics["message"], extra=chunk_metrics["extra"])

        tasks = [ChunkTask(idx=i, text=chunk.text, document_type=document_type) for i, chunk in enumerate(chunk_objects)]
        chunk_results = await self._run_chunks(tasks)

        all_entities, raw_steps, raw_constraints = self._merge_chunk_results(chunk_results)
        normalized_steps, step_id_map = self._normalize_steps(raw_steps)
        normalized_constraints = self._normalize_constraints(raw_constraints, step_id_map)

        processing_time = time.time() - start_time
        confidence_score = (
            sum(entity.confidence for entity in all_entities) / len(all_entities) if all_entities else 0.0
        )

        extraction = ExtractionResult(
            entities=all_entities,
            steps=normalized_steps,
            constraints=normalized_constraints,
            confidence_score=confidence_score,
            processing_time=processing_time,
            strategy_used=self._strategy_name(),
            quality_metrics={
                "entity_count": len(all_entities),
                "avg_confidence": confidence_score,
                "chunks_processed": len(chunk_objects),
            },
        )

        if extraction.confidence_score < final_threshold:
            logger.warning(
                "Extraction confidence below threshold",
                extra={"confidence_score": extraction.confidence_score, "quality_threshold": final_threshold},
            )

        self._update_stats(extraction)

        async with self.cache_lock:
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = extraction

        logger.info(
            "Extracted %s entities via %s in %.2fs (confidence %.2f)",
            len(extraction.entities),
            extraction.strategy_used,
            extraction.processing_time,
            extraction.confidence_score,
        )
        return extraction

    def _strategy_name(self) -> str:
        backend = self._backend_name or self._determine_backend()
        prompt = getattr(self.config, "prompting_strategy", "P0")
        return f"{backend}:{prompt}"

    def _summarize_chunks(self, chunk_objects, duration: float) -> Optional[Dict[str, Any]]:
        chunk_count = len(chunk_objects)
        if chunk_count == 0:
            return None
        avg_chunk_size = sum(len(chunk.text) for chunk in chunk_objects) / chunk_count
        avg_sentences = (
            sum(max(1, chunk.sentence_span[1] - chunk.sentence_span[0]) for chunk in chunk_objects) / chunk_count
        )
        cohesion_values = [
            chunk.meta.get("cohesion")
            for chunk in chunk_objects
            if isinstance(chunk.meta, dict) and isinstance(chunk.meta.get("cohesion"), (int, float))
        ]
        avg_cohesion = sum(cohesion_values) / len(cohesion_values) if cohesion_values else 0.0
        message = (
            f"Chunked document via {self.config.chunking_method}: "
            f"{chunk_count} chunks, avg size {avg_chunk_size:.1f} chars, "
            f"avg sentences {avg_sentences:.1f}, avg cohesion {avg_cohesion:.2f}, "
            f"chunk_time {duration:.3f}s"
        )
        return {
            "message": message,
            "extra": {
                "chunking_method": self.config.chunking_method,
                "chunk_count": chunk_count,
                "avg_chunk_size": round(avg_chunk_size, 2),
                "avg_sentences_per_chunk": round(avg_sentences, 2),
                "avg_chunk_cohesion": round(avg_cohesion, 2),
                "chunking_duration": round(duration, 4),
            },
        }

    def _merge_chunk_results(
        self, chunk_results: List[ChunkExtraction]
    ) -> Tuple[List[ExtractedEntity], List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_entities: List[ExtractedEntity] = []
        raw_steps: List[Dict[str, Any]] = []
        raw_constraints: List[Dict[str, Any]] = []
        for result in chunk_results:
            if not result:
                continue
            all_entities.extend(result.entities)
            raw_steps.extend(result.steps)
            raw_constraints.extend(result.constraints)
        return all_entities, raw_steps, raw_constraints

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(self.config.confidence_threshold)

    def _normalize_steps(self, steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        normalized: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}
        for idx, step in enumerate(steps):
            content = (step.get("text") or "").strip()
            if not content:
                continue
            new_id = f"S{len(normalized) + 1}"
            original_id = str(step.get("id") or new_id)
            id_map[original_id] = new_id
            normalized.append(
                {
                    "id": new_id,
                    "text": content,
                    "order": len(normalized) + 1,
                    "confidence": self._coerce_confidence(step.get("confidence")),
                }
            )
        return normalized, id_map

    def _normalize_constraints(self, constraints: List[Dict[str, Any]], step_id_map: Dict[str, str]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, constraint in enumerate(constraints):
            text = (constraint.get("text") or "").strip()
            if not text:
                continue
            raw_refs = constraint.get("steps", [])
            attached_steps = [step_id_map[ref] for ref in raw_refs if ref in step_id_map]
            normalized.append(
                {
                    "id": f"C{len(normalized) + 1}",
                    "text": text,
                    "confidence": self._coerce_confidence(constraint.get("confidence")),
                    "steps": attached_steps,
                }
            )
        return normalized

    def _update_stats(self, result: ExtractionResult):
        stats = self.performance_stats
        total_before = stats["total_extractions"]
        stats["total_extractions"] = total_before + 1
        stats["avg_processing_time"] = (
            (stats["avg_processing_time"] * total_before + result.processing_time) / (total_before + 1)
        )
        strategy = result.strategy_used
        stats["strategy_usage"][strategy] = stats["strategy_usage"].get(strategy, 0) + 1

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_stats.copy()

    def clear_cache(self):
        self.cache.clear()
        if self._pool:
            self._pool.shutdown()
            self._pool = None
        logger.info("Cleared knowledge engine cache and reset worker pool")
