#!/usr/bin/env bash
# Quick Metal smoke test for the Mistral 7B GGUF on Apple silicon.
# Run from repo root: bash scripts/test_mistral_metal.sh

set -euo pipefail

export GGML_METAL_PATH_RESOURCES="$(
python - <<'PY'
import site, pathlib
print(pathlib.Path(site.getsitepackages()[0]) / "bin")
PY
)"
export LLAMA_METAL=1

python - <<'PY'
from llama_cpp import Llama

llm = Llama(
    model_path="models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # push all layers to Metal
    n_threads=8,
    use_mmap=True,
    use_mlock=False,
    verbose=True,
)

print("Loaded; running a test generation...")
resp = llm("<s>[INST] Give a 1-sentence fun fact. [/INST]", max_tokens=64)
print(resp["choices"][0]["text"].strip())
PY
