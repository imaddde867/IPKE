# Dual Semantic Chunking Implementation

The current implementation is in `src/processors/chunkers/dual_semantic.py`.

## Implemented Algorithm

IPKE's Dual Semantic Chunker (DSC):

1. Splits document text into sentences with spaCy.
2. Embeds sentences with SentenceTransformers.
3. Computes adjacent sentence cosine similarities (stored as a prefix-sum array).
4. Builds parent boundaries using the global DP objective from ADR-0001:
   `J(B) = Σ H(b) − λ|B|`, solved via `DP[j] = max_{i<j}{ DP[i] + H(i,j) − λ + β·𝟙[j is heading] }`
   where `H(i,j)` is the mean consecutive-pair cosine similarity of the block via prefix sums.
   Hard guardrails `[dsc_parent_min_sentences, dsc_parent_max_sentences]` prevent degenerate blocks.
5. Refines each parent block with `BreakpointSemanticChunker._compute_boundaries` (child breakpoints).
6. Enforces the configured character cap.

## Hyperparameters

| Param | Default | How set |
|-------|---------|---------|
| `dsc_lambda` | 0.05 | Calibrated on `niosh_nmam_surface_sampling_guidance` (held-out, not in 8-doc eval set). Sweep {0.05..0.50}; peak avg_parent_cohesion=0.347 at λ=0.05. See ADR-0001. |
| `dsc_beta` | 0.2 | Manually chosen. Sensitivity analysis over {0.1, 0.2, 0.3} required in ECIR paper. |

## Paper Status

The DP implementation matches the global objective claimed in the thesis (§4.1.2) and ECIR paper.
POC validation completed 2026-06-07: DP Φ=0.3473 vs heuristic Φ=0.3346 (Δ=+0.0127, PASS).
See ADR-0001 for full calibration results and kill-criterion record.

## What's next

- **Multi-seed sweep** (issue #60): blocked until this issue was closed — now unblocked.
  Run `{DP parents + breakpoint children}` vs `{heuristic parents + breakpoint children}` on the
  8-doc eval set across multiple seeds. Reports StepF1, ConstraintCoverage, Φ.
- **β sensitivity analysis**: sweep β ∈ {0.1, 0.2, 0.3}, report in ECIR paper.
- `_compute_boundaries()` (child breakpoint refinement) is unchanged and out of scope for DP work.
