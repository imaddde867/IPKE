# IPKE

**Primary direction**: an IPKE method paper on skeleton-conditioned, source-grounded
constraint attachment for procedural graph extraction with local language models.

IPKE is the primary contribution. The corpus, taxonomy, validators, and metrics are
supporting evaluation infrastructure. ADR-0005 and
`docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md` control the research
direction. `docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md`
controls how annotation candidates become paper evidence. ADR-0004's ECIR Resource
Paper decision is superseded.

The central causal question is whether an explicit step skeleton improves fine-grained
constraint attachment after matching schema information, parser behavior, calls, token
budget, context, and validation policy. Do not describe generic two-stage prompting,
constraint edges, or semantic chunking as first-ever contributions.

## Language

### Core output

**Procedural Knowledge Graph (PKG)**:
A directed graph representing a procedure as a network of steps and constraints, with typed edges encoding execution order and constraint attachment.
_Avoid_: knowledge graph (too generic), structured output, extraction result

**Procedural Step**:
One ordered, actionable operation in a procedure that an agent must perform. Steps have an execution order and form the backbone of a PKG.
_Avoid_: action, task, instruction (too generic)

**Constraint**:
A condition, guard, precondition, postcondition, warning, or prohibition that governs one or more steps. Constraints must be attached to at least one step to be valid.
_Avoid_: requirement, rule, annotation

**Constraint Attachment**:
The explicit link between a constraint and the step or steps it governs. It is IPKE's
primary evaluated outcome. Unsupported predictions and attachments to unknown steps are
reported before any deterministic filter; they are not silently discarded from the
method comparison.
_Avoid_: constraint linking, constraint assignment

### Chunking

**Dual Semantic Chunker (DSC)**:
The implementation name for IPKE's hierarchy-aware segmentation path. It creates parent
blocks with a global objective and heading information, then applies semantic child
refinement. The paper uses **constraint-preserving segmentation** for the research claim
and retains that claim only if pair co-location and downstream attachment improve.
_Avoid_: dual chunker, semantic chunker (ambiguous — use DSC)

**Cohesion Score**:
The average cosine similarity between adjacent sentence embeddings within a segment. Defined as `H(b_{i:j}) = (1/(j−i)) Σ cos(e_k, e_{k+1})`. Measures semantic continuity within a chunk.
_Avoid_: similarity score, chunk quality

**Heading Bonus (β)**:
A bonus term added to the DP objective when a candidate split point aligns with a structural heading. Set to β = 0.2. Encourages splits at structural boundaries even when local cohesion is high.
_Avoid_: heading weight, structure penalty

### Prompting

**Two-Stage Decomposition (P3)**:
The implementation name for skeleton-conditioned extraction. Stage 1 extracts procedural
steps. Stage 2 receives the source plus step identifiers and texts, then extracts typed
constraints attached to those steps. Reduced instruction drift is a hypothesis, not an
established mechanism.
_Avoid_: two-stage prompting, P3 strategy (use full name on first mention)

**Prompting Strategy**:
One of four approaches for instructing the LLM: P0 (zero-shot), P1 (few-shot in-context learning), P2 (chain-of-thought), P3 (two-stage decomposition).
_Avoid_: prompt type, prompting mode

### Evaluation

**Procedural Fidelity Score (Φ)**:
A legacy composite index: `Φ = 0.5·ConstraintCoverage + 0.3·StepF1 +
0.2·Kendall`. It is exploratory and cannot determine a gate or headline claim because it
does not include the primary attachment or grounding outcomes. Component metrics are
authoritative until Phi is redesigned, justified, and sensitivity-tested.
_Avoid_: Phi score, fidelity metric

**Tier-A Evaluation**:
Step-level and constraint-level evaluation: StepF1, AdjacencyF1, Kendall τ, ConstraintCoverage, ConstraintAttachmentF1, and Φ. Uses semantic alignment via sentence embeddings at cosine threshold 0.75.
_Avoid_: standard evaluation, basic evaluation

**Tier-B Evaluation**:
Graph-structure evaluation using SMatch: GraphPrecision, GraphRecall, GraphF1, NEXT\_EdgeF1, Logic\_EdgeF1. Operates on the full PKG topology.
_Avoid_: graph evaluation, advanced evaluation

**Gold Annotation**:
A source-grounded production reference annotation. Models and agents may create and
audit immutable candidates, but cannot create human evidence. Paper eligibility requires
all of the following:

- a named independent human completed a full pass over the bounded source procedure;
- every accepted step and constraint resolves to exact committed-source offsets;
- primary-pass timing and candidate accept, edit, reject, and add decisions are logged;
- any preregistered blind pass was frozen and scored before independent adjudication;
- principal-investigator decisions are limited to unresolved taxonomy, implicit-evidence,
  or safety-critical cases and are logged;
- schema, structure, grounding, provenance, agreement, adjudication, manifest, and split
  gates pass.

`quality.review_status = "reviewed"` and a `+ human-verified:<handle>` provenance marker
are not sufficient by themselves. Agent review is never human verification.
_Avoid_: ground truth, reference annotation, gold standard (unless qualifying)

**review_status values** (locked vocabulary):
- `unreviewed` — schema-valid scaffold, no human pass. Default for fresh LLM drafts.
- `reviewed` — a review pass is declared complete. This status alone does not identify
  who performed it and does not make an annotation eligible for paper evidence.
- `llm_draft` — produced by LLM pipeline only; EXCLUDED from paper IAA per `docs/annotation/guidelines.md`. Used to mark legacy `datasets/paper/second_pass/*.json` files that are NOT independent human annotations.

**Constraint Type** (locked, 6 values):
`precondition`, `postcondition`, `guard`, `parameter`, `role_assignment`, `reference`. Defined in `docs/annotation/constraint-types.md`. No other type values are valid for paper-grade gold.

**Constraint Enforcement** (locked, 3 values):
`must`, `should`, `may`. Maps to source-text modal verbs ("shall"/"will" → must, "should"/"recommended" → should, "may"/"can" → may).

**Inter-Annotator Agreement (IAA)**:
Pre-adjudication agreement between independently produced annotations of the same frozen
source span. At least 25% of experiment-eligible procedures receive a source-only blind
second pass selected before results are inspected. Both passes are frozen and every
selected pair is reported before a third independent human adjudicates disagreements.
Attachment-edge F1 of at least 0.70 is the method design's current G0 gate; Cohen's κ and
other agreement measures are diagnostics, not reasons to discard a selected pair.
Accidental exposure to another pass invalidates that assignment and requires
reassignment. Principal-investigator escalation is limited to unresolved taxonomy,
implicit-evidence, or safety-critical decisions.
_Avoid_: annotation agreement, kappa score
