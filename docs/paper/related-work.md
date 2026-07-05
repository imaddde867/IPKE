# Related Work — positioning IPKE-Bench

Working notes for the Related Work section. Purpose: make the differentiation
from PAGED explicit and accurate, anchor the enforcement axis in prior art so it
does not read as invented, and own the scale asymmetry as a design choice rather
than a weakness. Every claim here is verifiable by a reviewer; none of it says
"first to attach constraints."

## 1. Procedural graph / procedural-IE benchmarks

- **PAGED (Du et al., ACL 2024).** The nearest competitor and the one to
  position against directly. 3,394 documents; encodes `DataConstraint` /
  `ActionConstraint` and links them to actions via constraint-flow edges. Gold
  is template-derived silver: built on a pre-existing BPMN business-process
  model collection (Dumas et al., 2018) with WikiHow-trained segmentation, and
  scored by BLEU text overlap. No enforcement dimension. Reports that LLMs reach
  SOTA on element *text* but stay < 0.5 F1 on logical structure (gateway/flow) —
  a result that supports the IPKE thesis that attachment/structure, not text, is
  the hard part. Full axis-by-axis comparison table lives in
  `docs/annotation/constraint-types.md`.
- **PAGED follow-ups.** A 2026 multi-agent procedural-graph extraction method
  (arXiv 2601.19170) already uses PAGED as its testbed — evidence the task is
  live and that PAGED is becoming the standard baseline. IPKE-Bench should cite
  it as the reason our differentiation must be sharp.
- **FABLE (arXiv 2505.24258).** Data-flow analysis over *synthetic* procedural
  text (recipes / travel / STRIPS plans) with preconditions-and-effects derived
  from PDDL schemas. Synthetic and reasoning-focused; contrast: IPKE is real
  regulated SOPs with human-verified labels.
- **Carriero & Celino 2024; KEO; CAMB.** Extract steps/actions/objects/
  equipment/temporal but carry no constraint-on-step schema. IPKE's taxonomy is
  a strict superset w.r.t. constraint coverage.
- **Procedural memory / agent benchmarks** (ProcBench-style; procedural-memory
  retrieval, arXiv 2511.21730; LCStep, arXiv 2409.01344). Adjacent but different
  target: they measure retrieval/planning over procedures, not extraction of the
  constraint structure from source documents. Useful to cite as the *downstream*
  consumer of what IPKE-Bench measures (constraint-aware retrieval is on our
  research arc).

## 2. The enforcement axis is grounded in Requirements Engineering — cite it

The must / should / may grade is the sharpest novelty relative to procedural-IE
benchmarks, but it is **not novel to NLP broadly** — and saying so is a strength,
not a concession. The requirements-engineering / NLP4RE community has long
treated modal/deontic verbs ("shall / should / may") as the carriers of
obligation strength (RFC 2119 formalises exactly this three-level scale for
normative documents; NLP4RE work extracts requirements and their modality from
specification text). NASA NPR 8715.3 in our own corpus *is* a requirements
document.

Positioning move: frame IPKE-Bench's enforcement axis as **importing the RE
deontic distinction into procedural-knowledge extraction, where it has been
absent.** This (a) grounds the axis in established prior art rather than
asserting it from nothing, (b) widens the reviewer/audience pool to
REFSQ/RE-adjacent readers, (c) pre-empts the "is must/should/may principled?"
question with "yes — it is the RFC 2119 / RE modality scale, applied to
step-bound constraints." Add 2–3 NLP4RE citations in the camera-ready.

## 3. Owning the scale asymmetry (8 docs vs PAGED's 3,394)

We will not out-scale PAGED and should not try. The defensible frame, stated
first so a reviewer cannot use it against us:

> IPKE-Bench is a **small, expert-verified diagnostic benchmark** — the
> deliberate counterpart to large, template-derived silver sets. For measuring
> whether an extractor gets *typed, graded, step-bound constraints* right, label
> fidelity dominates label volume: a multi-fold under-production of constraints
> by an LLM draft (our §1 finding; exact ratio under decision per
> `docs/paper/D1_SCOPE_DECISION.md`) is only diagnosable against gold that is
> itself correct. Eight documents, each human-verified against source, spanning five
> regulatory families, is sized for *evaluation*, not training — researchers
> tune on external corpora (PAGED, ProPara, OpenPI) and report on IPKE-Bench.

Pair it with an honest **contamination caveat**: our sources are public
US-government documents that predate current model cutoffs and are plausibly in
pretraining data. This makes IPKE a test of *extraction-under-known-source*, not
of memorisation — and argues for the forthcoming Finnish-language extension (on
the research arc) as the contamination-resistant complement.

## 4. Source-span grounding — a scoped future sub-metric (do NOT overclaim now)

A 2026 instruction-hierarchy paper (arXiv 2604.17624) and others require the
model to quote the *verbatim source span* each constraint originated from, as a
hallucination guard. IPKE already carries char-spans at the document level
(`second_pass/_source/`). A natural, low-cost extension: record a per-constraint
verbatim `source_span`, and add a **grounding** sub-metric (does the extracted
constraint's span overlap the gold span?). This would (a) harden the datasheet
against "are these constraints really in the text", (b) give a third scored
axis beside type and attachment.

Status: **backlog, not v1.** The v1 verbatim-wording rule already requires
constraint text to be drawn from source near-verbatim; a scored span metric is
an enhancement, and claiming it before it is implemented would be dishonest.
Tracked as a candidate issue; decide during camera-ready whether it lands in the
resource paper or the follow-up method paper.

## One-paragraph Related Work opener (draft)

> Procedural information extraction has moved from rule- and template-based
> systems to LLM-based extractors, with PAGED (Du et al., 2024) establishing the
> current large-scale benchmark for procedural-graph extraction, including coarse
> data/action constraints linked to actions. What no benchmark measures is
> whether an extractor recovers the *typed, deontically-graded, step-bound*
> constraints that make a regulated procedure executable and safe — the
> difference between a guard and a parameter, between a `must` and a `may`, and
> the specific step each governs. IPKE-Bench targets exactly this, importing the
> deontic modality scale from requirements engineering into procedural-knowledge
> extraction, and scoring constraint attachment by exact step-id F1 over
> human-verified gold on real safety-critical SOPs.

## Citations to secure before submission

- Du et al. 2024, PAGED (arXiv 2408.03630) — verify venue/pages.
- Dumas et al. 2018 (BPMN model collection PAGED builds on) — confirm it is the
  cited source for PAGED's gold provenance.
- RFC 2119 (Bradner 1997) — for the must/should/may deontic scale.
- 1–3 NLP4RE / requirements-modality-extraction papers — for the enforcement bridge.
- arXiv 2601.19170 (PAGED multi-agent follow-up) — as evidence PAGED is a standard testbed.
- Gebru et al. 2021 (Datasheets for Datasets) — already cited for the datasheet.
