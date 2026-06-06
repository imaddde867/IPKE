"""Set LLM environment variables before any heavy imports.

All three variables use `os.environ.setdefault`, so any value the caller
sets in the environment before launching the process takes precedence.

TOKENIZERS_PARALLELISM
    HuggingFace tokenizers fork worker processes by default. Disable forking
    when the LLM worker pool manages its own process-level parallelism, to
    avoid deadlocks and excess memory.

OMP_NUM_THREADS / MKL_NUM_THREADS
    BLAS/MKL default to using all available cores inside a single Python
    process. When multiple LLM workers run concurrently, each trying to
    use all cores, the result is oversubscription and slower overall
    throughput. Default of 1 keeps each worker single-threaded and lets
    the worker pool (not BLAS) control parallelism.

    Override before launch when appropriate:
      - High-core lab machine, 2 workers:
          OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LLM_NUM_WORKERS=2 uv run ...
      - GPU-backed run: leave at 1 (GPU backend handles parallelism).
      - Single-worker CPU run on laptop: leave at 1.

    Record the values used in experiment metadata; they affect both
    throughput and reproducibility of timing measurements.
"""
import os

# Must be set before tokenizers library loads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
