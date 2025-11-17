"""Multiprocess worker pool for LLM chunk inference."""

from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass
from multiprocessing import Process, get_context
from typing import Dict, List, Optional, Sequence

from src.ai.llm_backends import build_backend
from src.ai.prompting import build_prompt_strategy
from src.ai.types import ChunkExtraction
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkTask:
    idx: int
    text: str
    document_type: str


def _available_cuda_devices() -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        devices = [d.strip() for d in env.split(",") if d.strip() not in {"", "-1"}]
        if devices:
            result = []
            for dev in devices:
                try:
                    result.append(int(dev))
                except ValueError:
                    continue
            if result:
                return result
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:  # pragma: no cover
        pass
    return []


def _worker_entry(task_queue, result_queue, config, backend_name: str, strategy_name: str, device_id: Optional[int]):
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    if hasattr(config, "prompting_strategy"):
        setattr(config, "prompting_strategy", strategy_name)
    backend = build_backend(config, backend_name)
    strategy = build_prompt_strategy(config)
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, payload = task
        try:
            extraction = strategy.run(backend, payload["text"], payload["document_type"])
            result_queue.put((idx, extraction, None))
        except Exception as exc:  # pragma: no cover - defensive
            result_queue.put((idx, ChunkExtraction(), str(exc)))
    result_queue.put(("__worker_exit__", None, None))


class LLMWorkerPool:
    """Fan-out worker pool that runs LLM prompts in parallel."""

    def __init__(
        self,
        config,
        backend_name: str,
        num_workers: int,
        device_strategy: str = "single",
        timeout: Optional[int] = None,
    ):
        self.config = config
        self.backend_name = backend_name
        self.num_workers = max(1, num_workers)
        self.device_strategy = device_strategy
        self.timeout = timeout or getattr(config, "processing_timeout", 300)
        self.ctx = get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=self.num_workers * 2)
        self.result_queue = self.ctx.Queue()
        self.workers: List[Process] = []
        self._start_workers()

    def _assign_devices(self) -> List[Optional[int]]:
        devices = _available_cuda_devices()
        if not devices:
            return [None] * self.num_workers
        if self.device_strategy == "single":
            return [devices[0]] * self.num_workers
        assigned = []
        for idx in range(self.num_workers):
            assigned.append(devices[idx % len(devices)])
        return assigned

    def _start_workers(self):
        device_map = self._assign_devices()
        for worker_idx in range(self.num_workers):
            device_id = device_map[worker_idx]
            proc = self.ctx.Process(
                target=_worker_entry,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.config,
                    self.backend_name,
                    getattr(self.config, "prompting_strategy", "P0"),
                    device_id,
                ),
                daemon=True,
            )
            proc.start()
            self.workers.append(proc)
            logger.info(
                "Started LLM worker %s on device %s", proc.pid, device_id if device_id is not None else "cpu"
            )

    def process(self, tasks: Sequence[ChunkTask]) -> List[ChunkExtraction]:
        if not tasks:
            return []
        in_flight = 0
        for task in tasks:
            payload = {"text": task.text, "document_type": task.document_type}
            self.task_queue.put((task.idx, payload))
            in_flight += 1

        results: Dict[int, ChunkExtraction] = {}
        errors = 0
        deadline = time.time() + self.timeout

        while len(results) < len(tasks):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("LLM inference timed out.")
            try:
                idx, extraction, error = self.result_queue.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if idx == "__worker_exit__":
                continue
            if error:
                logger.warning("Worker failed on chunk %s: %s", idx, error)
                errors += 1
                results[idx] = ChunkExtraction()
            else:
                results[idx] = extraction

        ordered = [results[task.idx] for task in tasks]
        if errors:
            logger.warning("Completed with %s worker errors.", errors)
        return ordered

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for proc in self.workers:
            proc.join(timeout=10)
        self.workers.clear()


__all__ = ["ChunkTask", "LLMWorkerPool"]
