# IPKE Pipeline: Real-Hardware Validation Report

**Date:** 2026-05-27  
**Hardware:** RTX 5060 Ti 16 GB (Blackwell SM 12.0), WSL2 Ubuntu, CUDA 13.2 (Driver 596.36)  
**Software:** Python 3.12.3, llama-cpp-python 0.3.23 (cu124 wheel), PyTorch 2.6.0+cu124  
**Model:** Mistral-7B-Instruct-v0.2 (Q4_K_M, 4.1 GB)  
**Strategy:** Dual Semantic Chunking (DSC) + Progressive Prompting Paradigm (P3)  
**Temperature:** 0.1 · **Seed:** 42

---

## 1. Hardware Compatibility

The RTX 5060 Ti (Blackwell SM 12.0) under CUDA 13.2 required specific handling:

- **llama-cpp-python**: Source builds with `GGML_CUDA=1` against CUDA 13.2 time out (>15 min) due to SM 12.0 PTX compilation. The pre-built cu124 wheel from the abetlen/llama-cpp-python v0.3.23 GitHub release includes SM 12.0 cubins and works out of the box.
- **PyTorch**: CUDA 12.4 wheels (torch 2.6.0+cu124) are backward-compatible with CUDA 13.2 via the NVIDIA driver compatibility layer.
- **GPU utilisation**: With GPU layers = -1 (full offload), the 4.1 GB Q4_K_M model + ~400 MB sentence-transformer embedder consume ~4.5 GB of 16 GB VRAM, leaving comfortable headroom.

**Takeaway**: Blackwell GPUs require pre-built wheels or explicit SM 12.0 PTX targets. The cu124 compatibility layer works transparently for PyTorch.

---

## 2. Dependency Resolution

Running with `uv` revealed ABI sensitivity:

- **numpy version pin**: spaCy/thinc links against numpy C ABI. numpy ≥2.4.6 violates the expected ABI version (96 expected vs 88 provided), causing `thinc` import failure. Pinning `numpy<2` resolves this.
- **llama-cpp-python locked via URL source**: Adding `[tool.uv.sources]` in pyproject.toml pointing to the cu124 wheel URL prevents `uv sync` from overwriting the local installation with PyPI's CPU-only build.

---

## 3. Extraction Results

| Document | Steps Extracted | Constraints | Entities | Time (s) | Confidence |
|---|---|---|---|---|---|
| 3M Marine OEM SOP | 40 | 17 | 23 | 175.7 | 0.913 |
| DOA Food Processing | 152 | 92 | 129 | 328.4 | 0.894 |
| Fire Safety Guideline | 82 | 52 | 49 | 174.2 | 0.903 |

The pipeline extracted procedural knowledge across three diverse domains (manufacturing, food processing, safety) with high self-reported confidence (0.89–0.91) and reasonable runtime (avg ~3 min/doc on Mistral-7B Q4_K_M).

---

## 4. Evaluation (Tier A)

| Document | StepF1 | AdjacencyF1 | Kendall τ | Phi |
|---|---|---|---|---|
| 3M OEM SOP | 0.348 | 0.500 | 0.758 | **0.256** |
| DOA Food Proc Stor | 0.156 | 0.545 | 0.981 | **0.243** |
| Fire Safety | 0.248 | 0.462 | 0.934 | **0.261** |
| **Macro Avg** | **0.251** | **0.502** | **0.891** | **0.253** |

### Notable Deviations from Thesis Baseline

The thesis reports Phi = 0.699, StepF1 = 0.632, Kendall = 0.303 for the 3M SOP. Our run shows:

- **Lower StepF1 (0.348 vs 0.632)**: Likely reflects differences in the Mistral-7B checkpoint (v0.2 vs thesis version), prompt templates, or alignment threshold.
- **Higher Kendall τ (0.758 vs 0.303)**: The P3 strategy recovers step order more faithfully in our run.
- **ConstraintCoverage = 0**: The gold standard stores constraints as step-level fields (`steps[].constraints.*`), but the evaluator expects a top-level `constraints` list. Since gold has zero top-level constraints, ConstraintCoverage is undefined and defaults to 0 in the Phi formula. The thesis baseline likely evaluated against a gold version with top-level constraint annotations.
- **Phi composition**: With ConstraintCoverage = 0 and AdjacencyF1 excluded from the formula, Phi reduces to `0.3·StepF1 + 0.2·Kendall`, capping the maximum at 0.5. The observed Phi = 0.256 for 3M is within this constrained range.

### Comparison to Thesis

| Metric | Thesis (3M SOP) | This Run (3M SOP) |
|---|---|---|
| Phi | 0.699 | 0.256 |
| StepF1 | 0.632 | 0.348 |
| Kendall τ | 0.303 | 0.758 |

The divergence suggests sensitivity to model version, prompt templates, or gold annotation format. The Kendall improvement (+0.455) with P3 is encouraging.

---

## 5. Technical Roadblocks Encountered

| Issue | Root Cause | Resolution |
|---|---|---|
| `uv run xxx` re-downloads 1.3 GB wheel every time | uv's isolated venv recreation | Use `source .venv/bin/activate` instead |
| llama-cpp-python build timeout (>15 min) | CUDA 13.2 + SM 12.0 PTX compilation | Pre-built cu124 wheel from GitHub releases |
| numpy ABI mismatch (96 vs 88) | numpy 2.x ABI break with spaCy/thinc | Pin `numpy<2` in pyproject.toml |
| `uv sync` overwrites cu124 wheel | PyPI has CPU-only llama-cpp-python | Locked URL source in `[tool.uv.sources]` |
| CUDA not detected by llama-cpp-python | Missing `nvidia-smi` in WSL PATH | Set `CUDA_VISIBLE_DEVICES=0` in `.env` |

---

## 6. Conclusions

The IPKE pipeline (DSC + P3 + Mistral-7B) runs end-to-end on consumer Blackwell hardware (RTX 5060 Ti) under WSL2. Key findings:

1. **Blackwell compatibility is achievable** with pre-built cu124 wheels; source compilation is not required for inference workloads.
2. **Step ordering (Kendall τ) is robust** across documents (0.758–0.981), suggesting P3 preserves procedural structure well.
3. **Step recall (StepF1) varies by domain** — the DOA food processing doc (152 steps) achieves only 0.156 StepF1, indicating chunking or prompt limits for longer documents.
4. **Constraint evaluation is blocked** by a format mismatch between gold annotations (step-level) and evaluator interface (top-level). Aligning the evaluation harness to read step-level constraints would enable meaningful ConstraintCoverage and Phi comparisons.
5. **Total pipeline runtime** for 3 documents is ~11.3 minutes on a 16 GB VRAM consumer card, demonstrating practical feasibility for small-scale procedural knowledge extraction.
