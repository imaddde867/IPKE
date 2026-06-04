# Dual Semantic Chunking Implementation

The current implementation is in `src/processors/chunkers/dual_semantic.py`.

## Implemented Algorithm

IPKE's current Dual Semantic Chunker:

1. Splits document text into sentences with spaCy.
2. Embeds sentences with SentenceTransformers.
3. Computes adjacent sentence cosine distances.
4. Builds parent boundaries using local distance thresholds and optional heading detection.
5. Refines each parent block with `BreakpointSemanticChunker._compute_boundaries`.
6. Enforces the configured character cap.

## Paper Wording Constraint

The thesis describes DSC using a global objective and heading bonus. The code currently implements a practical hierarchical heuristic rather than a single global DSC dynamic program with explicit heading bonus.

For ECIR 2027, choose one:

- Update the method text to describe the implemented heuristic exactly.
- Or implement the global DSC objective and validate it against the existing chunker tests and experiment results.

Until that decision is made, paper drafts should avoid claiming the current code solves a single global DSC objective with heading bonus.
