# Research Vision: IPKE Method First

This is the top-level research direction. ADR-0005 records the decision, and
`docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md` defines the controlling
experiment protocol.

## One-sentence thesis

**Under matched schemas, decoding policies, context, and inference budgets, conditioning
constraint extraction on an explicit step skeleton improves source-grounded constraint
attachment for local language models.**

IPKE itself is the contribution under test. The corpus, taxonomy, validator, metrics,
datasheet, and reproducibility tooling exist to make that method claim credible.

## Why the problem matters

Procedural extraction is incomplete when a system recovers actions but loses the guards,
thresholds, roles, and prerequisites that govern them. A constraint that is absent,
hallucinated, or attached to the wrong step can change the meaning of a procedure even
when the step list appears plausible.

Prior work already represents action-constraint links and uses multi-stage procedural
graph extraction. IPKE therefore does not claim first-ever constraint attachment,
two-stage prompting, or semantic chunking. Its narrower contribution is to isolate
whether a step skeleton helps local models produce fine-grained, grounded attachments
under controlled inference budgets.

## Contribution hierarchy

1. **Skeleton-conditioned attachment**

   Extract a step skeleton, expose its stable identifiers and text to a second call, and
   require typed constraints to attach to those steps. Compare it with call- and
   budget-matched self-refinement so any gain is attributable to conditioning rather than
   a second chance to generate.

2. **Constraint-preserving segmentation**

   Test whether hierarchy-aware segmentation keeps a constraint and its governed step in
   the same model context. This is secondary until it beats full-context and simpler
   chunking controls on pair co-location and downstream attachment.

3. **Local quality-cost frontier**

   Report quality together with actual calls, tokens, latency, peak memory, and hardware.
   Local execution supports a reduced-data-egress argument only when runtime evidence
   exists. It does not by itself establish privacy, compliance, or production safety.

4. **Supporting evidence infrastructure**

   Maintain source-grounded gold, explicit graph relations, typed constraints, honest
   provenance, and reproducible analysis. These are indispensable scientific controls,
   but they are not positioned as a standalone benchmark contribution.

## What elite means here

An elite paper is a short chain from one important claim to evidence that can survive an
adversarial review.

1. **Causal controls**: schema information, parser behavior, calls, token budget,
   decoding, and filtering are separated instead of bundled into P3 versus P0.
2. **Human evidence**: agent-reviewed annotations remain explicitly unsigned until a
   human checks them against the source. Independent agreement is blind and measured.
3. **Correct statistical units**: documents and source families support generalization;
   seeds estimate stochastic variation and are nested within documents.
4. **Grounding**: annotations and predictions can be traced to source evidence. Unsupported
   output remains visible before filtering.
5. **Reproducibility**: raw predictions, prompt and model hashes, complete configuration,
   failures, and cached re-scoring reproduce every table.
6. **Negative results remain publishable**: if skeleton conditioning does not beat
   call-matched self-refinement, report that result rather than changing the test after
   seeing it.
7. **Scoped claims**: no privacy, safety, executability, model-scale, or downstream-value
   claim exceeds the measurements performed.

## Evidence gates

No confirmatory sweep begins until:

- active test gold is human verified and structurally valid;
- blind second-pass coverage and agreement meet the frozen protocol;
- exact annotated spans are used for the attachment experiment;
- explicit relations are evaluated instead of synthesizing every graph as a linear chain;
- the same parser, schema, and repair policy are applied across conditions;
- a two-document pilot proves that every condition produces auditable, non-empty output;
- cached predictions reproduce the pilot metrics with one command.

The current eight-document corpus does not meet these gates. `review_status="reviewed"`
does not substitute for a human-verification marker.

## Experiment order

1. Compare joint, self-refined, skeleton-conditioned, and filtered extraction on exact
   annotated spans.
2. Separate raw generation effects from deterministic filter effects.
3. Evaluate segmentation independently with boundary and constraint-step co-location
   evidence.
4. Combine the components only after their individual effects are understood.
5. Test scale within one same-release dense model family and confirm direction on a
   second family.
6. Run external transfer only with a mapping frozen before evaluation.

ConstraintAttachmentF1 is the primary outcome. Grounded constraint recall, unsupported
rate, type and enforcement F1, StepF1, graph relations, and quality-cost measures are
secondary. Phi is exploratory until redesigned and validated.

## Venue strategy

The natural audience is NLP, information extraction, or knowledge-graph research, not a
resource track. COLING 2027 is an October checkpoint only if the human evidence and
controlled pilot are credible early enough. Otherwise the work moves to a later ARR
cycle or an appropriate Semantic Web research track. A calendar never relaxes the
evidence gates.

## CoRe value

IPKE supports CoRe work that turns technical procedures into auditable structured
representations or uses procedures as references for checking work. The durable value is
not a demo claim. It is a reusable method and evaluation discipline for local industrial
AI: explicit constraints, traceable evidence, measurable failure modes, and controlled
deployment costs.

Do not claim that IPKE underlies unrelated vision or time-series projects. Do not expose
partner manuals to cloud services as part of the paper workflow.

## Principles

- One falsifiable method claim defended completely beats several bundled claims.
- Honest negative evidence is more valuable than a confounded positive result.
- Human verification cannot be automated or delegated to a marker script.
- Gold corrections are manual; automation checks structure and provenance only.
- Full-context is a required control for any chunking claim.
- Component metrics outrank an unvalidated composite score.
- Preserve historical decisions by superseding them, not rewriting them silently.

## Controlling documents

- `docs/adr/0005-ipke-method-paper-primary.md`
- `docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md`
- `docs/paper/2026-07-04-execution-direction.md`
- `CONTEXT.md`
- `REPRODUCIBILITY.md`

ADR-0004 and `docs/paper/ipke-bench-resource-prd.md` remain historical records of the
superseded resource-paper direction.
