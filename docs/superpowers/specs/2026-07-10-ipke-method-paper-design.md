# IPKE Method Paper Design

Status: approved for implementation on 2026-07-10

## Decision

IPKE itself is the primary research contribution. The paper will test a method for
extracting source-grounded procedural knowledge graphs from safety-critical documents
with local language models. The dataset, taxonomy, validators, and evaluation harness
are supporting evidence infrastructure, not a benchmark-paper contribution.

The primary method is **skeleton-conditioned constraint attachment**:

1. extract an explicit procedural step skeleton;
2. condition typed constraint extraction on the resulting step identifiers;
3. require every predicted constraint to be grounded in source text and attached to
   the step or steps it governs;
4. validate the resulting graph without silently discarding unsupported output.

Hierarchy-aware segmentation remains a secondary method component. It becomes a paper
contribution only if a controlled experiment shows that it preserves constraint-step
context and improves downstream attachment.

This decision supersedes ADR-0004's benchmark-first direction and the active ECIR
Resource Paper framing.

## One-sentence thesis

Under matched schemas, decoding policies, context, and inference budgets, conditioning
constraint extraction on an explicit step skeleton improves source-grounded constraint
attachment for local language models.

## Novelty boundary

The paper must not claim that IPKE is the first system to represent constraints, attach
constraints to actions, use multiple prompts, or extract procedural graphs. PAGED
represents constraint-flow edges, text2flow evaluates flow-constraint structure, and
Carriero et al. use a two-prompt procedural KG pipeline.

The narrower question is not settled by those systems:

> Does explicit step-skeleton conditioning improve fine-grained, source-grounded
> safety-constraint attachment under local-model and inference-cost constraints?

The paper may claim this contribution only after the controlled protocol below is
completed. It must not use "first" language without a separate systematic review.

## Research questions

### RQ1: Skeleton conditioning

Does conditioning constraint extraction on an explicit step skeleton improve
ConstraintAttachmentF1 and grounded constraint recall relative to schema-matched,
budget-matched alternatives?

Primary outcome: document-level ConstraintAttachmentF1 under the preregistered Tier-A
semantic alignment protocol.

Secondary outcomes:

- exact ConstraintAttachmentF1;
- grounded constraint precision, recall, and F1;
- unsupported-constraint rate;
- constraint type and enforcement F1;
- StepF1;
- results before and after any output filter;
- calls, input and output tokens, latency, peak memory, and hardware metadata.

### RQ2: Filtering contribution

How much of full IPKE's measured improvement comes from skeleton-conditioned generation,
and how much comes from deterministic rejection of invalid output?

The paper must report predictions before and after filtering. If the skeleton-conditioned
raw output does not beat call-matched self-refinement, the result is filter-driven and
cannot support a skeleton-conditioning claim.

### RQ3: Constraint-preserving segmentation

Does hierarchy-aware segmentation keep a constraint and its governed step in the same
model context more often than simpler segmentation methods, and does this improve RQ1's
primary outcome?

Primary segmentation outcome: gold constraint-step pair co-location rate.

Secondary segmentation outcomes:

- procedure-boundary precision, recall, and F1;
- downstream ConstraintAttachmentF1;
- method-by-segmentation interaction;
- chunk count, size distribution, and context utilization.

Generic embedding cohesion is diagnostic only. It is not evidence of extraction quality.

### RQ4: Model scale and quality-cost frontier

Is the skeleton-conditioning gain larger for smaller checkpoints within one same-release
dense model family, and where does full IPKE sit on the quality-cost frontier across at
least two local model families?

This is a secondary systems result. Local execution may support a reduced-data-egress
claim when runtime evidence exists. It must not be described as private, safe,
GDPR-compliant, or deployment-ready without separate evidence.

## RQ1 controlled conditions

All conditions use the same source span, taxonomy and enforcement definitions, output
schema, grammar-constrained decoding where supported, non-deleting parser and normalizer,
repair policy, model parameters, paired seed, and metric implementation. Two-call
conditions share a frozen total completion-token cap selected on development data.

1. **C0 Joint/raw**: one call extracts steps, constraints, types, enforcement, and links;
   no deleting filter.
2. **C1 Joint/filtered**: C0 followed by IPKE's deterministic type, enforcement,
   grounding, and valid-step-ID rules.
3. **C2 Self-refine/raw**: a second call receives C0 and the source, returns corrections,
   and a deterministic non-deleting merge produces the final graph.
4. **C3 Skeleton/raw**: call 1 extracts steps; call 2 receives the source and extracted
   step identifiers and texts, then returns typed constraints attached to those steps.
5. **C4 Full IPKE**: C3 followed by the same filter used in C1.

The confirmatory contrasts are C3 minus C2 for skeleton conditioning, C1 minus C0 and C4
minus C3 for filtering, and C4 minus C1 for the full method. If C4 minus C1 succeeds but
C3 minus C2 does not, the evidence supports a filter effect, not skeleton conditioning.

Prompt text and total available information must be audited so that a richer taxonomy or
extra task instruction is not unique to one condition. Report actual, not planned, token
usage.

## RQ3 controlled conditions

Compare:

1. **S0**: full annotated subprocedure with no segmentation where it fits the declared
   context;
2. **S1**: sentence-aligned fixed-token windows;
3. **S2**: heading-only recursive segmentation;
4. **S3**: semantic-breakpoint segmentation;
5. **S4**: hierarchy-aware global parents plus semantic child refinement.

The confirmatory matrix is C0-C4 across S0 and S4, plus C0 and C4 across S1-S3. Select
the strongest non-IPKE chunked baseline among S1-S3 on development documents and freeze
it before testing. Use zero overlap and a shared child-token cap unless a preregistered
development study justifies otherwise.

Tune segmentation hyperparameters on development documents only. Freeze them before
evaluating held-out source families. Do not use the test gold to tune boundary penalties,
heading bonuses, semantic thresholds, or child refinement.

## Data and evidence eligibility

The system must represent these states separately:

- **structurally valid**: schema, identifiers, relations, and enumerations are valid;
- **source grounded**: annotation text and evidence spans resolve to source material;
- **manually reviewed**: a source-to-gold pass has corrected step, constraint, type,
  enforcement, and attachment decisions;
- **human verified**: a named human has personally reviewed and signed the document;
- **agreement eligible**: an independent annotator completed a blind second pass;
- **experiment eligible**: the document satisfies all gates for the declared experiment
  and belongs to its frozen split.

No state may be inferred from `review_status = "reviewed"` alone. Agent review must never
be stamped as human verification.

Gold annotation decisions are manual. Automation is limited to schema checks, source
span checks, hashes, provenance completeness, placeholder detection, split leakage,
identifier integrity, and warnings. Scripts and models must not decide step identity,
constraint identity, type, enforcement, attachment, scope, sign-off, or adjudication.

The active eight-document corpus is not experiment eligible yet. It has no human-verified
gold, no independent blind second pass, and no locked source-family-held-out test split.
Current full-document evaluation is additionally invalid because most gold files cover a
small subprocedure span while the runner processes the complete source document.

## Evaluation protocol

- Evaluate RQ1 first on the exact annotated subprocedure spans.
- Evaluate RQ3 separately on long-document boundary and co-location annotations.
- Run the combined system only after both component protocols pass their pilot gates.
- Use documents or source families as the generalization unit.
- Treat seeds as repeated measurements nested within a document and configuration.
- Average paired seeds within each document, model, and configuration before the main
  comparison.
- Use a hierarchical bootstrap that resamples source family, document, and then seed,
  plus an exact paired sign-flip or permutation test for the primary contrast.
- Report effect sizes, confidence intervals, per-document results, and per-constraint-type
  results.
- Freeze semantic matching thresholds on development judgments.
- Preserve raw predictions and complete run metadata.

Phi is a secondary descriptive index at most. Component metrics remain primary. Phi may
not support a headline claim until attachment and grounding are represented, its weights
are justified, and a sensitivity analysis shows the conclusion is not weight-dependent.

## Stop/go gates

Before any full model sweep:

1. **G0 Measurement**: every test gold is human verified; at least 25% is independently
   double-annotated blind; attachment-edge agreement F1 is at least 0.70; disagreements
   are adjudicated before test evaluation.
2. **G1 Engineering**: schema parse success is at least 99%; truncation is below 1%; code,
   prompt, gold, configuration, and model hashes are recorded; one command reproduces
   metrics from cached predictions.
3. **G2 Skeleton method**: C3 minus C2 is at least 0.05 macro AttachmentF1, its 95%
   confidence interval lower bound is above zero, the direction holds under S0 and S4,
   and ConstraintCoverage loss is no more than 0.03.
4. **G3 Full IPKE**: G2 passes; C4 minus C1 is at least 0.05 AttachmentF1 with a positive
   confidence interval lower bound and coverage loss no more than 0.03.
5. **G4 Segmentation**: under C4, S4 beats the frozen non-IPKE chunked baseline by at
   least 0.03 AttachmentF1 and 0.05 co-location recall without losing more than 0.03
   StepF1.
6. **G5 Long context**: S4 is non-inferior to S0 within 0.01 AttachmentF1 for an
   efficiency claim. Only a positive confidence interval permits a claim that chunking
   improves extraction over full context.

Before G0 is available, a development-only two-document pilot may proceed after both
documents are manually corrected and structurally grounded, at least one is human
verified, and every C0-C4 condition produces non-empty raw predictions. The pilot is a
pipeline gate, not paper evidence. It should show C3 minus C2 of at least 0.03 without
coverage collapse before expanding the matrix.

Stop the sweep if any gate fails. Fix the protocol before spending more compute.

## Claims policy

Claims allowed after a successful controlled experiment:

- skeleton conditioning improved or did not improve attachment within the evaluated
  documents, models, and budgets;
- hierarchy-aware segmentation improved or did not improve pair co-location and
  downstream attachment;
- full IPKE occupied a measured point on a local quality-cost frontier;
- particular constraint types or document families benefited or failed.

Claims forbidden without additional evidence:

- first-ever constraint attachment or two-stage procedural extraction;
- generic novelty of Dual Semantic Chunking;
- pipeline design matters more than model size;
- 7B beats 70B as a general result;
- privacy, GDPR compliance, safety-critical readiness, executability, or digital-twin
  readiness;
- improved operator trust, expert effort, or downstream safety without a human or task
  study;
- generalization beyond the evaluated source families.

## Research architecture

The research artifact has three layers:

1. **Method kernel**: text input, segmentation, skeleton extraction, constraint extraction,
   grounding, attachment, and canonical procedural graph construction.
2. **Experiment kernel**: immutable run specification, execution, raw prediction storage,
   evidence eligibility, metrics, paired statistics, and result materialization.
3. **Optional adapters**: API, Streamlit, Neo4j, visualization, OCR, audio, Office, and
   container surfaces. These consume the method kernel but do not define paper behavior.

The first architectural implementation is one deep experiment module. Existing scripts
and Make targets become thin adapters. Conflicting multi-seed and Docker sweep paths are
retired only after the canonical path reproduces their supported behavior.

## Venue strategy

The immediate venue family is NLP or knowledge-graph research, not a resource track.
COLING 2027 is a viable October checkpoint only if the human gold, controlled pilot, and
frozen protocol are complete early enough for a credible paper. Otherwise the work moves
to a later ARR cycle or an appropriate Semantic Web research track. Submission timing may
not weaken the evidence gates.

## Implementation order

1. Supersede benchmark-first governance and issue framing.
2. Correct evidence eligibility so unsigned gold cannot enter a paper run.
3. Manually repair active golds one source at a time, without stamping human sign-off.
4. Establish a frozen development and source-family-held-out split.
5. Build the canonical experiment module with one red-green tracer bullet.
6. Run the two-document RQ1 pilot.
7. Add RQ2 annotations and segmentation controls only after RQ1 is trustworthy.
8. Run the full matrix and write only claims supported by the stored evidence.

## Superseded direction documents

The following documents remain historical evidence until they are rewritten, but they are
not active research direction after this decision:

- `docs/adr/0004-ecir-resource-paper-primary.md`;
- `docs/paper/ipke-bench-resource-prd.md`;
- `docs/paper/2026-07-04-execution-direction.md`;
- benchmark-first sections of `README.md`, `BENCHMARK.md`, `AGENTS.md`, and
  `docs/research-vision.md`;
- benchmark-first notes in the 2ndBrain IPKE project folder.

## References

- PAGED: <https://aclanthology.org/2024.acl-long.583/>
- text2flow: <https://aclanthology.org/2026.findings-eacl.158/>
- EDC: <https://aclanthology.org/2024.emnlp-main.548/>
- Carriero et al.: <https://doi.org/10.1007/978-3-031-77792-9_26>
- ARR review form: <https://aclrollingreview.org/reviewform>
- ARR reviewer guidelines: <https://aclrollingreview.org/reviewerguidelines>
