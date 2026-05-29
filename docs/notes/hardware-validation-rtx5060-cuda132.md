# Hardware Validation: RTX 5060 Ti, CUDA 13.2

Date: 2026-05-27

This note records a real-hardware IPKE run on consumer Blackwell hardware. Treat it as an
engineering validation artifact, not as a paper result table. The run proves the pipeline can
execute end to end on the target local-GPU stack, but the evaluation numbers are not comparable
to the thesis baseline until the constraint-gold format issue is fixed.

## Environment

| Field | Value |
|---|---|
| Hardware | RTX 5060 Ti 16 GB, Blackwell SM 12.0 |
| OS | WSL2 Ubuntu |
| CUDA / driver | CUDA 13.2, driver 596.36 |
| Python | 3.12.3 |
| llama-cpp-python | 0.3.23, cu124 wheel |
| PyTorch | 2.6.0+cu124 |
| Model | Mistral-7B-Instruct-v0.2 Q4_K_M |
| Strategy | DSC + P3 |
| Temperature / seed | 0.1 / 42 |

## Dependency Findings

- `llama-cpp-python` source builds with `GGML_CUDA=1` against CUDA 13.2 timed out after more
  than 15 minutes, likely because Blackwell SM 12.0 compilation is expensive from source.
- The prebuilt `llama-cpp-python` 0.3.23 cu124 wheel from the upstream GitHub release worked on
  Linux x86_64 and avoided the source build path.
- The direct cu124 wheel is Linux x86_64 specific. It must be guarded with an environment marker
  so Apple Silicon development can continue using the Metal wheel or a local platform build.
- `spacy` / `thinc` / `blis` is not safe with Python 3.14 in the current dependency set. Use
  Python 3.12 for reproducible runs until the NLP stack is upgraded.
- A numpy C-ABI mismatch was observed when compiled dependencies and runtime numpy diverged.
  Keeping `numpy<2` is the correct short-term compatibility choice for this stack.

## Extraction Run

| Document | Steps | Constraints | Entities | Runtime (s) | Self-confidence |
|---|---:|---:|---:|---:|---:|
| 3M Marine OEM SOP | 40 | 17 | 23 | 175.7 | 0.913 |
| DOA Food Processing | 152 | 92 | 129 | 328.4 | 0.894 |
| Fire Safety Guideline | 82 | 52 | 49 | 174.2 | 0.903 |

Total runtime for the three-document smoke run was about 11.3 minutes. This is useful evidence
that the pipeline is operational on a 16 GB consumer GPU, but the sample is too small for claims
about method quality.

## Evaluation Output

| Document | StepF1 | AdjacencyF1 | Kendall tau | Phi |
|---|---:|---:|---:|---:|
| 3M OEM SOP | 0.348 | 0.500 | 0.758 | 0.256 |
| DOA Food Processing | 0.156 | 0.545 | 0.981 | 0.243 |
| Fire Safety | 0.248 | 0.462 | 0.934 | 0.261 |
| Macro average | 0.251 | 0.502 | 0.891 | 0.253 |

Do not compare these Phi values directly to the thesis headline result. The run surfaced a likely
evaluation-interface problem: the current gold files store constraints under each step, while the
evaluator expects top-level constraints. That makes ConstraintCoverage collapse to zero and caps
Phi at `0.3 * StepF1 + 0.2 * Kendall` for this run.

The high Kendall values are still worth noting as a diagnostic signal: P3 preserved ordering well
in this run. Step recall and constraint scoring need a corrected evaluation path before they can
support ECIR claims.

## Implications For The Paper Push

1. Use this note as reproducibility context for local GPU setup, not as an experiment result.
2. Fix constraint loading or gold normalization before running the next metric table.
3. Keep Python below 3.14 until the `spacy` / `thinc` / `blis` chain is upgraded and tested.
4. Keep platform-specific LLM wheels behind uv environment markers.
5. Record hardware, CUDA version, model, quantization, seed, temperature, backend, and library
   versions in every run artifact.

## Next Checks

Run these after dependency changes:

```bash
uv run pytest
uv run python -m src.evaluation.metrics \
  --gold-dir datasets/archive/gold_human \
  --pred-dir logs/pkg_runs \
  --out-file logs/eval_results/results.json \
  --tier A
```

The second command should only be used with committed or otherwise archived raw predictions.
