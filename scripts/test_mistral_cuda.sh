#!/usr/bin/env bash
# CUDA smoke test for the Mistral 7B GGUF on NVIDIA hardware.
# Prereqs on the CUDA box:
#   pip install --force-reinstall --no-cache-dir \
#     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
#     "llama-cpp-python==0.2.84"
# Run from repo root: bash scripts/test_mistral_cuda.sh

set -euo pipefail

export LLAMA_CUBLAS=1
# Optional: export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
import os
from llama_cpp import Llama

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES", "<not set>"))

llm = Llama(
    model_path="models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # offload all layers to CUDA
    n_threads=8,
    use_mmap=True,
    use_mlock=False,
    verbose=True,
)

print("Loaded; running a test generation...")
resp = llm("<s>[INST] Give a 1-sentence fun fact. [/INST]", max_tokens=64)
print(resp["choices"][0]["text"].strip())
PY
