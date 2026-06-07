# Implement DSC as the global DP claimed in the thesis

The thesis (§4.1.2) and the ECIR paper claim DSC solves a global objective
`J(B) = Σ H(b) − λ|B|` via dynamic programming, with a heading bonus
`DP[j] = max_{i<j}{ DP[i] + c(i,j) − λ + β·𝟙[j is heading] }` and β = 0.2
set by manual inspection.

The current `_parent_boundaries()` implements a greedy local-window heuristic
(sliding-window mean + k·std threshold, left-to-right scan) — not the DP.

We decide to implement the global DP to match the claim (Option B over Option A),
because the DP formulation is the novel contribution and rewriting the paper to
describe a threshold heuristic would eliminate the methodological novelty.

Before committing experimental resources, a single-seed proof-of-concept on one
document must confirm the DP outperforms or matches the heuristic on StepF1/Φ.

POC implementation scope: replace only `_parent_boundaries()` with the DP.
Keep `_compute_boundaries()` child refinement intact. Two conditions only:
{heuristic parents + breakpoint children} vs {DP parents + breakpoint children}.

POC document: `niosh_nmam_surface_sampling_guidance` (downloaded alternate,
`selected_for_gold=false`, CDC public domain). Not in the 8-doc eval set, not in
the thesis archive. NIOSH analytical method docs have clear numbered section
structure — heading bonus β fires meaningfully, reducing false-negative risk.
Must NOT be one of the 8 evaluation gold files — same held-out discipline as λ.

Pre-registered success criterion (set before running, not after):
  - DP Φ ≥ heuristic Φ → ship DP
  - DP within ±0.01 of heuristic → ship DP (paper-code alignment > noise at n=1)
  - DP < heuristic by > 0.05 Φ → stop; likely a bug, not algorithmic loss; debug first

If the POC shows the DP is strictly worse beyond the kill criterion, this decision
must be revisited and Option A (rewrite prose to match heuristic) reconsidered.

β = 0.2 is treated as manually chosen, not tuned. A sensitivity analysis over
β ∈ {0.1, 0.2, 0.3} is required in the ECIR paper to show robustness.

λ is not stated in the thesis. It must be calibrated on a held-out document
(not one of the 8 evaluation docs) by sweeping λ ∈ {0.05, 0.1, 0.2, 0.3} and
selecting the value that produces a parent-block count comparable to the current
heuristic. Both λ and β must be reported in the paper.

## Deviations from pre-registration

**λ sweep range extended**: The pre-registered sweep was λ ∈ {0.05, 0.1, 0.2, 0.3}.
The actual sweep runs λ ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}. Rationale: the
wider range provides better coverage of the penalty landscape and avoids truncating
the search if the optimum lies above 0.3. The selection criterion (maximise
avg_parent_cohesion on the held-out doc) is unchanged. This deviation is recorded
here before results are observed, preserving the pre-registration discipline.
