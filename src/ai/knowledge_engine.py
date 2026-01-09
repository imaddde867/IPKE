"""LLM-driven knowledge extraction engine."""
from src.ai.llm_env_setup import *  

import difflib
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from copy import deepcopy

import numpy as np

from src.ai.types import ChunkExtraction, ExtractedEntity, ExtractionResult
from src.ai.worker_pool import ChunkTask, LLMWorkerPool
from src.ai.llm_backends import normalize_backend_name
from src.core.unified_config import UnifiedConfig, get_config
from src.logging_config import get_logger
from src.processors.chunkers import Chunk, get_chunker
from src.validation import validate_extraction
from src.graph.canonicalizer import assign_canonical_ids
from src.validation.constraint_validator import validate_constraints

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
        self._chunk_dedup_model = None
        self._chunk_dedup_device: Optional[str] = None
        log_path = getattr(self.config, "validation_error_log", "logs/validation_errors.jsonl")
        self.validation_log_path = Path(log_path)

    def _determine_backend(self) -> str:
        requested = getattr(self.config, "llm_backend", None)
        if requested and str(requested).lower() not in {"auto", "none"}:
            return normalize_backend_name(str(requested).lower())
        backend = self.config.detect_gpu_backend()
        if backend == "cuda":
            return "llama_cpp"
        return "transformers"

    def _ensure_pool(self):
        if self._pool is not None:
            return
        backend_name = self._determine_backend()
        worker_setting = getattr(self.config, "llm_num_workers", 0)
        try:
            worker_count = int(worker_setting)
        except (TypeError, ValueError):
            worker_count = 0
        device_strategy = getattr(self.config, "llm_device_strategy", "auto")
        self._pool = LLMWorkerPool(
            config=self.config,
            backend_name=backend_name,
            num_workers=worker_count,
            device_strategy=device_strategy,
            timeout=self.config.processing_timeout,
        )
        self._backend_name = backend_name
        resolved_workers = getattr(self._pool, "num_workers", worker_count)
        logger.info(
            "Initialized LLM worker pool backend=%s workers=%s device_strategy=%s prompting=%s",
            backend_name,
            resolved_workers,
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
        self,
        content: str,
        document_type: str = "unknown",
        quality_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        final_threshold = quality_threshold if quality_threshold is not None else self.config.quality_threshold
        content_hash = hashlib.md5(content.encode()).hexdigest()
        doc_id = document_id or f"doc_{content_hash[:8]}"
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
        chunk_objects = self._apply_chunk_redundancy_filter(content, chunk_objects)
        chunk_duration = time.time() - chunk_start

        chunk_metrics = self._summarize_chunks(chunk_objects, chunk_duration)
        if chunk_metrics:
            logger.info(chunk_metrics["message"], extra=chunk_metrics["extra"])

        tasks = [ChunkTask(idx=i, text=chunk.text, document_type=document_type) for i, chunk in enumerate(chunk_objects)]
        chunk_results = await self._run_chunks(tasks)

        all_entities, raw_steps, raw_constraints = self._merge_chunk_results(chunk_results)
        normalized_steps, step_id_map, step_alias_map = self._normalize_steps(raw_steps)
        normalized_constraints = self._normalize_constraints(raw_constraints, step_id_map, step_alias_map)

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
            metadata={"document_id": doc_id},
        )

        self._validate_against_schema(extraction, doc_id, document_type)

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

    def _apply_chunk_redundancy_filter(self, content: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks or not bool(getattr(self.config, "enable_chunk_dedup", False)):
            return chunks
        if len(chunks) <= 1:
            return chunks
        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self._embed_chunk_texts(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chunk deduplication skipped; embedding error: %s", exc)
            return chunks
        if embeddings.size == 0:
            return chunks
        threshold = float(getattr(self.config, "chunk_dedup_threshold", 0.9))
        overlap_ratio = float(getattr(self.config, "chunk_dedup_overlap_ratio", 0.7))
        min_unique_chars = max(32, int(getattr(self.config, "chunk_dedup_min_unique_chars", 200)))
        filtered: List[Chunk] = [chunks[0]]
        kept_embeddings: List[np.ndarray] = [embeddings[0]]
        trimmed = 0
        dropped = 0
        for idx in range(1, len(chunks)):
            chunk = chunks[idx]
            current_embedding = embeddings[idx]
            similarity = float(np.dot(kept_embeddings[-1], current_embedding))
            if similarity >= threshold:
                chunk, current_embedding, action = self._resolve_redundant_chunk(
                    filtered[-1],
                    chunk,
                    current_embedding,
                    content,
                    overlap_ratio,
                    min_unique_chars,
                )
                if action == "drop" or chunk is None:
                    dropped += 1
                    continue
                if action == "trim":
                    trimmed += 1
            filtered.append(chunk)
            kept_embeddings.append(current_embedding)

        if trimmed or dropped:
            logger.info(
                "Chunk redundancy filter compacted %d -> %d chunks (trimmed=%d dropped=%d)",
                len(chunks),
                len(filtered),
                trimmed,
                dropped,
            )
        return filtered

    def _resolve_redundant_chunk(
        self,
        prev_chunk: Chunk,
        curr_chunk: Chunk,
        current_embedding: np.ndarray,
        content: str,
        overlap_threshold: float,
        min_unique_chars: int,
    ) -> Tuple[Optional[Chunk], np.ndarray, str]:
        overlap = max(0, prev_chunk.end_char - curr_chunk.start_char)
        total = max(1, curr_chunk.end_char - curr_chunk.start_char)
        if overlap <= 0:
            return curr_chunk, current_embedding, "keep"
        ratio = overlap / total
        if ratio < overlap_threshold:
            return curr_chunk, current_embedding, "keep"
        unique_start = max(prev_chunk.end_char, curr_chunk.start_char)
        unique_len = curr_chunk.end_char - unique_start
        if unique_len < max(1, min_unique_chars):
            return None, current_embedding, "drop"
        trimmed_text, new_start, new_end = self._slice_content(content, unique_start, curr_chunk.end_char)
        if not trimmed_text or (new_end - new_start) < max(1, min_unique_chars // 2):
            return None, current_embedding, "drop"
        curr_chunk.text = trimmed_text
        curr_chunk.start_char = new_start
        curr_chunk.end_char = new_end
        new_embedding = self._embed_chunk_texts([trimmed_text])
        if new_embedding.size == 0:
            return None, current_embedding, "drop"
        return curr_chunk, new_embedding[0], "trim"

    def _slice_content(self, content: str, start: int, end: int) -> Tuple[str, int, int]:
        segment = content[start:end]
        left = 0
        right = len(segment)
        while left < right and segment[left].isspace():
            left += 1
        while right > left and segment[right - 1].isspace():
            right -= 1
        trimmed = segment[left:right]
        return trimmed, start + left, start + right

    def _embed_chunk_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        model = self._get_chunk_dedup_embedder()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.atleast_2d(np.asarray(embeddings))

    def _get_chunk_dedup_embedder(self):
        if self._chunk_dedup_model is not None:
            return self._chunk_dedup_model
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        model_id = (
            getattr(self.config, "chunk_dedup_embedding_model", None)
            or getattr(self.config, "embedding_model_path", "all-mpnet-base-v2")
        )
        device = self._resolve_chunk_dedup_device()
        self._chunk_dedup_model = SentenceTransformer(model_id, device=device)
        logger.info("Loaded chunk deduplication encoder '%s' on %s", model_id, device)
        return self._chunk_dedup_model

    def _resolve_chunk_dedup_device(self) -> str:
        if self._chunk_dedup_device:
            return self._chunk_dedup_device
        backend = str(getattr(self.config, "gpu_backend", "cpu") or "cpu").lower()
        if backend in {"cuda", "auto", "gpu", "multi_gpu_parallel"}:
            try:
                import torch

                if torch.cuda.is_available():
                    self._chunk_dedup_device = "cuda"
                    return self._chunk_dedup_device
            except Exception:  # noqa: BLE001
                pass
        if backend in {"metal", "mps"}:
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._chunk_dedup_device = "mps"
                    return self._chunk_dedup_device
            except Exception:  # noqa: BLE001
                pass
        self._chunk_dedup_device = "cpu"
        return self._chunk_dedup_device

    def _merge_chunk_results(
        self, chunk_results: List[ChunkExtraction]
    ) -> Tuple[List[ExtractedEntity], List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_entities: List[ExtractedEntity] = []
        raw_steps: List[Dict[str, Any]] = []
        raw_constraints: List[Dict[str, Any]] = []
        for result in chunk_results:
            if not result:
                continue
            raw_entities.extend(result.entities)
            raw_steps.extend(result.steps)
            raw_constraints.extend(result.constraints)
        merged_entities = assign_canonical_ids(raw_entities)
        unique_entities: Dict[str, ExtractedEntity] = {}
        for entity in merged_entities:
            key = entity.canonical_id or entity.content
            if key not in unique_entities:
                unique_entities[key] = entity
        return list(unique_entities.values()), raw_steps, raw_constraints

    def _validate_against_schema(
        self,
        extraction: ExtractionResult,
        document_id: str,
        document_type: str,
    ) -> None:
        payload = self._build_validation_payload(document_id, document_type, extraction)
        autofix = getattr(self.config, "schema_autofix_enabled", True)
        is_valid, issues = validate_extraction(payload, autofix=autofix)
        if issues:
            self._log_validation_messages(document_id, issues, is_valid)
        if not is_valid:
            message = f"Schema validation failed for {document_id}"
            if getattr(self.config, "strict_schema_validation", False):
                raise ValueError(f"{message}: {'; '.join(issues)}")
            logger.warning(message, extra={"document_id": document_id, "issues": issues})

    def _build_validation_payload(
        self,
        document_id: str,
        document_type: str,
        extraction: ExtractionResult,
    ) -> Dict[str, Any]:
        step_nodes: List[Dict[str, Any]] = []
        for step in extraction.steps:
            step_nodes.append(
                {
                    "id": str(step.get("id")) if step.get("id") is not None else None,
                    "order": step.get("order"),
                    "text": step.get("text") or "",
                    "section": step.get("section"),
                    "context": step.get("context"),
                    "references": step.get("references"),
                }
            )

        condition_nodes: List[Dict[str, Any]] = []
        for constraint in extraction.constraints:
            expression = (
                constraint.get("text")
                or constraint.get("expression")
                or constraint.get("description")
                or ""
            )
            condition_nodes.append(
                {
                    "id": str(constraint.get("id")) if constraint.get("id") is not None else None,
                    "type": constraint.get("type"),
                    "expression": expression,
                    "context": constraint.get("context"),
                }
            )

        edges: List[Dict[str, Any]] = []
        ordered_ids = [str(step["id"]) for step in step_nodes if step.get("id")]
        for current, nxt in zip(ordered_ids, ordered_ids[1:]):
            edges.append({"from_id": current, "to_id": nxt, "type": "NEXT"})

        constraint_step_refs = {}
        for constraint in extraction.constraints:
            cid = constraint.get("id")
            if not cid:
                continue
            cid = str(cid)
            refs = constraint.get("steps") or []
            if not isinstance(refs, list):
                refs = [refs]
            normalized_refs = []
            for ref in refs:
                if isinstance(ref, dict):
                    ref_id = ref.get("id")
                else:
                    ref_id = ref
                if ref_id:
                    normalized_refs.append(str(ref_id))
            if normalized_refs:
                constraint_step_refs[cid] = normalized_refs

        for cid, targets in constraint_step_refs.items():
            for target in targets:
                edges.append({"from_id": cid, "to_id": target, "type": "CONDITION_ON"})

        return {
            "document_id": document_id or "unknown_document",
            "document_type": document_type,
            "title": None,
            "steps": step_nodes,
            "conditions": condition_nodes,
            "equipment": [],
            "parameters": [],
            "edges": edges,
            "metadata": {"source": "knowledge_engine"},
        }

    def _log_validation_messages(self, document_id: str, issues: List[str], is_valid: bool) -> None:
        self.validation_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_id": document_id,
            "is_valid": is_valid,
            "issues": issues,
        }
        with self.validation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(self.config.confidence_threshold)

    def _normalize_steps(
        self, steps: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
        deduped_steps, step_alias_map = self._deduplicate_steps(steps)
        normalized: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}

        for idx, step in enumerate(deduped_steps):
            content = (step.get("text") or "").strip()
            if not content:
                continue
            new_id = f"S{len(normalized) + 1}"
            raw_id = str(step.get("id") or new_id)
            id_map[raw_id] = new_id
            normalized.append(
                {
                    "id": new_id,
                    "text": content,
                    "order": len(normalized) + 1,
                    "confidence": self._coerce_confidence(step.get("confidence")),
                }
            )

        # map aliases (duplicate raw IDs) onto the canonical new IDs
        for raw_id, canonical_raw in step_alias_map.items():
            if raw_id in id_map:
                continue
            if canonical_raw in id_map:
                id_map[raw_id] = id_map[canonical_raw]

        return normalized, id_map, step_alias_map

    def _normalize_constraints(
        self,
        constraints: List[Dict[str, Any]],
        step_id_map: Dict[str, str],
        step_alias_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        deduped_constraints = self._deduplicate_constraints(constraints, step_alias_map)
        normalized: List[Dict[str, Any]] = []
        available_raw_ids = set(step_alias_map.keys()) | set(step_alias_map.values()) | set(step_id_map.keys())
        available_target_ids = available_raw_ids | set(step_id_map.values())

        def _coerce_ref(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, dict):
                for key in ("id", "step_id", "step", "target"):
                    candidate = value.get(key)
                    if candidate:
                        return str(candidate)
                return None
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    coerced = _coerce_ref(item)
                    if coerced:
                        return coerced
                return None
            text = str(value).strip()
            return text or None

        def _fuzzy_match_step(ref: str) -> Optional[str]:
            candidates = list(available_target_ids)
            match = difflib.get_close_matches(ref, candidates, n=1, cutoff=0.75)
            if not match:
                return None
            candidate = match[0]
            if candidate in step_id_map:
                return step_id_map[candidate]
            if candidate in step_alias_map and step_alias_map[candidate] in step_id_map:
                return step_id_map[step_alias_map[candidate]]
            if candidate in step_id_map.values():
                return candidate
            return None

        for constraint in deduped_constraints:
            text = (
                constraint.get("text")
                or constraint.get("expression")
                or constraint.get("description")
                or ""
            ).strip()
            if not text:
                continue

            raw_refs = constraint.get("steps") or constraint.get("attached_to") or constraint.get("targets") or []
            if not isinstance(raw_refs, list):
                raw_refs = [raw_refs]

            attached_steps: List[str] = []
            for ref in raw_refs:
                ref_id = _coerce_ref(ref)
                if not ref_id:
                    continue
                canonical_raw = step_alias_map.get(ref_id, ref_id)
                mapped = step_id_map.get(canonical_raw) or step_id_map.get(ref_id)
                if not mapped:
                    mapped = _fuzzy_match_step(ref_id)
                if mapped:
                    attached_steps.append(mapped)

            if not attached_steps:
                continue

            normalized_constraint = deepcopy(constraint)
            normalized_constraint.pop("steps", None)
            normalized_constraint.pop("attached_to", None)
            normalized_constraint["id"] = f"C{len(normalized) + 1}"
            normalized_constraint["text"] = text
            normalized_constraint["confidence"] = self._coerce_confidence(constraint.get("confidence"))
            normalized_constraint["steps"] = attached_steps
            normalized.append(normalized_constraint)

        if not normalized:
            return normalized

        report = validate_constraints(normalized)
        if report.warnings:
            logger.warning("Constraint validation warnings: %s", report.warnings)
        if report.errors:
            error_ids = {cid for cid, _ in report.errors}
            logger.warning("Constraint validation errors (filtered out): %s", report.errors)
            normalized = [c for c in normalized if c.get("id") not in error_ids]

        return normalized

    def _deduplicate_steps(
        self, steps: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not steps:
            return [], {}
        threshold = float(
            getattr(self.config, "step_dedup_threshold", getattr(self.config, "chunk_dedup_threshold", 0.9))
        )
        texts = [(step.get("text") or "").strip() for step in steps]
        embeddings = self._embed_chunk_texts(texts)
        kept_indices: List[int] = []
        kept_embeddings: List[np.ndarray] = []
        alias_map: Dict[str, str] = {}

        for idx, (step, text) in enumerate(zip(steps, texts)):
            if not text:
                continue
            raw_id = str(step.get("id") or idx)
            if not kept_embeddings:
                kept_indices.append(idx)
                kept_embeddings.append(embeddings[idx])
                alias_map[raw_id] = raw_id
                continue
            sims = np.dot(np.vstack(kept_embeddings), embeddings[idx])
            best = float(np.max(sims))
            best_idx = int(np.argmax(sims))
            if best >= threshold:
                canonical_raw = str(steps[kept_indices[best_idx]].get("id") or kept_indices[best_idx])
                alias_map[raw_id] = canonical_raw
            else:
                kept_indices.append(idx)
                kept_embeddings.append(embeddings[idx])
                alias_map[raw_id] = raw_id

        deduped = [steps[i] for i in kept_indices]
        # ensure canonical ids map to themselves
        for step in deduped:
            raw_id = str(step.get("id") or "")
            if raw_id and raw_id not in alias_map:
                alias_map[raw_id] = raw_id
        return deduped, alias_map

    def _deduplicate_constraints(
        self, constraints: List[Dict[str, Any]], step_alias_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        if not constraints:
            return []
        threshold = float(
            getattr(self.config, "constraint_dedup_threshold", getattr(self.config, "chunk_dedup_threshold", 0.9))
        )
        texts = [
            (c.get("text") or c.get("expression") or c.get("description") or "").strip()
            for c in constraints
        ]
        embeddings = self._embed_chunk_texts(texts)
        kept: List[Dict[str, Any]] = []
        kept_embeddings: List[np.ndarray] = []

        def _normalize_refs(raw_refs: Any) -> List[str]:
            refs = raw_refs or []
            if not isinstance(refs, list):
                refs = [refs]
            normalized_refs: List[str] = []
            for ref in refs:
                if isinstance(ref, dict):
                    candidate = ref.get("id") or ref.get("step_id") or ref.get("step")
                else:
                    candidate = str(ref) if ref is not None else ""
                if candidate:
                    canonical = step_alias_map.get(candidate, candidate)
                    normalized_refs.append(canonical)
            return normalized_refs

        for idx, (constraint, text) in enumerate(zip(constraints, texts)):
            if not text:
                continue
            if not kept_embeddings:
                clone = deepcopy(constraint)
                clone["attached_to"] = _normalize_refs(
                    constraint.get("attached_to") or constraint.get("steps") or constraint.get("targets")
                )
                kept.append(clone)
                kept_embeddings.append(embeddings[idx])
                continue

            sims = np.dot(np.vstack(kept_embeddings), embeddings[idx])
            best = float(np.max(sims))
            best_idx = int(np.argmax(sims))
            if best >= threshold:
                # merge references into existing constraint
                existing = kept[best_idx]
                merged_refs = set(existing.get("attached_to", []))
                merged_refs.update(
                    _normalize_refs(
                        constraint.get("attached_to") or constraint.get("steps") or constraint.get("targets")
                    )
                )
                existing["attached_to"] = list(merged_refs)
                existing["confidence"] = max(
                    self._coerce_confidence(existing.get("confidence")),
                    self._coerce_confidence(constraint.get("confidence")),
                )
            else:
                clone = deepcopy(constraint)
                clone["attached_to"] = _normalize_refs(
                    constraint.get("attached_to") or constraint.get("steps") or constraint.get("targets")
                )
                kept.append(clone)
                kept_embeddings.append(embeddings[idx])

        return kept

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
