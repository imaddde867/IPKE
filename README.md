# IPKE — Industrial Procedural Knowledge Extraction

Thesis-grade, privacy-preserving pipeline that reconstructs Procedural Knowledge Graphs (PKG) from safety-critical industrial manuals for human-in-the-loop validation and decision-support integration (Thesis Abstract).

## Validated Impact

- **Local privacy preservation** — IPKE processes sensitive SOPs entirely on-prem via quantised 7B models, avoiding external APIs while maintaining schema fidelity (Thesis Abstract, §6.6).
- **Two-Stage Decomposition (P3)** — P3 delivers Step F1 = 0.377 and Φ = 0.611 across Tier-A documents (Table 10) and Φ = 0.699 with 75% constraint coverage on the 3M SOP, outperforming Llama-3.1-70B zero-shot (Φ = 0.187, 0% coverage) and even its own P3 setup (Φ = 0.439, 50% coverage) (Table 12).
- **Constraint-focused PKGs** — Constraint coverage rises from ≈0% under baseline prompting to 0.708 with P3 (Table 10), yielding queryable PKGs where GUARD edges bind safety rules to each procedural step (Fig. 11).

![Efficiency Frontier](assets/efficiency_frontier_phi.png)

## Method Kernel

- **Dual Semantic Chunking (DSC)** aligns document headings with embedding-based cohesion to limit context fragmentation for mid-sized models (Thesis §4.1).
- **P3 — Two-Stage Decomposition** decouples ordered step extraction from constraint attachment, eliminating schema drift seen in zero-shot and chain-of-thought baselines (Thesis §4.2–5.4).

## Run IPKE

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

```ini
# .env
GPU_BACKEND=metal
CHUNKING_METHOD=dual_semantic
PROMPTING_STRATEGY=P3
# Deduplicate overlapping content and enforce clean constraints/steps
ENABLE_CHUNK_DEDUP=true
```

```bash
# Reproduce chunking experiments
python scripts/experiments/run_all_chunking_experiments.py \
  --documents datasets/archive/test_data/text/*.txt

# API surface
python main.py  # http://localhost:8000/docs
```

Research distribution for academic and regulated industrial settings. See `LICENSE`.

---

Turku University of Applied Sciences · 2025

## Local LLM (Mistral 7B, GGUF)

- Download weights (requires a Hugging Face token):  
  `python - <<'PY'\nfrom huggingface_hub import hf_hub_download\nhf_hub_download(\n  repo_id=\"TheBloke/Mistral-7B-Instruct-v0.2-GGUF\",\n  filename=\"mistral-7b-instruct-v0.2.Q4_K_M.gguf\",\n  local_dir=\"models/llm\",\n  local_dir_use_symlinks=False,\n)\nPY`

- Metal (Apple silicon, fastest locally):  
  `pip install --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal "llama-cpp-python==0.2.84"`  
  Test: `bash scripts/test_mistral_metal.sh`

- CUDA (NVIDIA, other hardware):  
  `pip install --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 "llama-cpp-python==0.2.84"`  
  Test: `bash scripts/test_mistral_cuda.sh`

The app will pick up the GGUF at `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`; set `LLM_N_GPU_LAYERS=-1` to offload all layers to Metal/CUDA.

## Hardware Compatibility

IPKE auto-detects your hardware and gracefully falls back to CPU if GPU acceleration is unavailable:

| Hardware | Configuration | Notes |
|----------|--------------|-------|
| **NVIDIA GPU** | `GPU_BACKEND=cuda` | Auto-detected if CUDA available |
| **Apple Silicon** | `GPU_BACKEND=metal` | Auto-detected on macOS with MPS |
| **CPU only** | `GPU_BACKEND=cpu` | Default fallback, no GPU required |

The system uses `torch.cuda.is_available()` and `torch.backends.mps.is_available()` with try/except guards to ensure safe operation on any platform.

