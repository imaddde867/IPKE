# Pre-registered hypothesis and model selection for ECIR paper

Status: Superseded by ADR-0005 and the method-paper design on 2026-07-10. The historical
Phi pipeline-gap hypothesis and 320-run matrix below are not the current confirmatory
design.

## Hypothesis (pre-registered before any model results are seen)

The pipeline gap — defined as Φ(P3+DSC) − Φ(P0+fixed) — is larger at smaller
model scale than at larger model scale.

If results show a flat or inverted gap, the paper publishes the finding as
"pipeline gains are uniform across model scale" — also a valid contribution.
The framing must not be decided after seeing the numbers.

## Model selection

### Primary scale axis (within-family — clean scale test)

| Model | Params | Purpose |
|---|---|---|
| Llama-3.2-3B-Instruct | 3B | Small scale anchor |
| Llama-3.1-8B-Instruct | 8B | Mid scale |
| Llama-3.1-70B-Instruct | 70B | Large scale anchor |

Same family, same instruction-tuning approach. Confounds scale with nothing.

### Cross-architecture sanity (at P3+DSC only — not on scale axis)

| Model | Params | Purpose |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7B | Shows gain not a Llama quirk |
| Qwen3-9B-Instruct | 9B | Latest-gen cross-family check |

If Qwen3-9B Q4_K_M GGUF is unavailable on HuggingFace, fall back to Qwen2.5-7B.

## Quantization

All models: Q4_K_M GGUF via llama.cpp. Held constant — quantization is a confound.

## Experiment matrix

Scale axis: 3 models × 2 pipelines (P0+fixed, P3+DSC) × 8 docs × 5 seeds = 240 runs
Cross-arch: 2 models × 1 pipeline (P3+DSC) × 8 docs × 5 seeds = 80 runs
Total: 320 runs. Estimated GPU time: ~33 hrs on 4×V100.

## Rationale for rejecting "small beats large" framing

Comparing best-pipeline-small vs worst-pipeline-large is an unfair comparison —
a reviewer will catch it. The gap framing (same pipeline, measured at each scale)
is the testable scientific version and survives methodological scrutiny.
