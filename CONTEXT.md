# IPKE

A local, privacy-preserving pipeline for extracting structured procedural knowledge from safety-critical industrial documents, producing validated Procedural Knowledge Graphs (PKGs) suitable for decision support and human-in-the-loop workflows.

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
The explicit link between a constraint and the step(s) it governs. The primary challenge distinguishing IPKE from generic IE: a constraint without attachment is invalid and discarded.
_Avoid_: constraint linking, constraint assignment

### Chunking

**Dual Semantic Chunker (DSC)**:
IPKE's proposed chunking algorithm. Segments documents hierarchically: first into parent blocks using a global DP objective with heading bonus, then refines each block using breakpoint-based child chunking. The global objective is `J(B) = Σ H(b) − λ|B|`, solved by DP with a heading bonus term β·𝟙[j is heading position].
_Avoid_: dual chunker, semantic chunker (ambiguous — use DSC)

**Cohesion Score**:
The average cosine similarity between adjacent sentence embeddings within a segment. Defined as `H(b_{i:j}) = (1/(j−i)) Σ cos(e_k, e_{k+1})`. Measures semantic continuity within a chunk.
_Avoid_: similarity score, chunk quality

**Heading Bonus (β)**:
A bonus term added to the DP objective when a candidate split point aligns with a structural heading. Set to β = 0.2. Encourages splits at structural boundaries even when local cohesion is high.
_Avoid_: heading weight, structure penalty

### Prompting

**Two-Stage Decomposition (P3)**:
IPKE's proposed prompting strategy. Stage 1 extracts only procedural steps. Stage 2 extracts constraints and entities, with the constraint that every constraint must reference a step ID from Stage 1. Reduces instruction drift in smaller models.
_Avoid_: two-stage prompting, P3 strategy (use full name on first mention)

**Prompting Strategy**:
One of four approaches for instructing the LLM: P0 (zero-shot), P1 (few-shot in-context learning), P2 (chain-of-thought), P3 (two-stage decomposition).
_Avoid_: prompt type, prompting mode

### Evaluation

**Procedural Fidelity Score (Φ)**:
A composite metric: `Φ = 0.5·ConstraintCoverage + 0.3·StepF1 + 0.2·Kendall`. Primary headline metric for comparing configurations. ConstraintCoverage is weighted highest because in safety-critical industrial procedures, a missed constraint (unchecked guard, omitted warning) is a more severe failure mode than imperfect step ordering. Weights are justified by domain risk, not by optimisation. Sensitivity across `{0.4:0.4:0.2, 0.5:0.3:0.2, 0.6:0.2:0.2}` must be reported to show rankings are stable.
_Avoid_: Phi score, fidelity metric

**Tier-A Evaluation**:
Step-level and constraint-level evaluation: StepF1, AdjacencyF1, Kendall τ, ConstraintCoverage, ConstraintAttachmentF1, and Φ. Uses semantic alignment via sentence embeddings at cosine threshold 0.75.
_Avoid_: standard evaluation, basic evaluation

**Tier-B Evaluation**:
Graph-structure evaluation using SMatch: GraphPrecision, GraphRecall, GraphF1, NEXT\_EdgeF1, Logic\_EdgeF1. Operates on the full PKG topology.
_Avoid_: graph evaluation, advanced evaluation

**Gold Annotation**:
A human-reviewed ground-truth JSON file for a document. Must have `quality.review_status = "reviewed"` before use in paper experiments. LLM-assisted drafts that have not been corrected are not gold annotations — they are drafts.
_Avoid_: ground truth, reference annotation, gold standard (unless qualifying)

**Inter-Annotator Agreement (IAA)**:
Agreement between two independent annotators on the same document, measured as Cohen's κ. Minimum acceptable for a paper IAA claim: κ ≥ 0.61 (substantial, Landis & Koch 1977).
_Avoid_: annotation agreement, kappa score
