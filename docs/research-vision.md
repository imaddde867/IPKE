# Research Vision — the North Star

*This is the top-level direction document. It sits above the PRD
(`docs/paper/ipke-bench-resource-prd.md`, the requirements) and the execution
direction (`docs/paper/2026-07-04-execution-direction.md`, the current issue
board). Those two answer "what do we build next"; this one answers "what are we
actually trying to become, and what separates an elite contribution from a
competent one." Read this when you lose the thread of why the small work
matters.*

---

## The one-sentence thesis

**Procedural knowledge is only actionable when its constraints are bound to the
steps they govern — and no existing benchmark measures whether an extraction
system gets that binding right.** IPKE-Bench makes *constraint attachment* a
first-class, separately-scored extraction target. That is the whole idea.
Everything else — the taxonomy, the validator, the baselines, the retrieval
task — exists to make that one claim rigorous and reusable.

If a reviewer remembers exactly one thing about this work, it must be this: we
were the group that noticed step extraction without constraint attachment is a
half-measured task, defined the missing half, and shipped the resource that lets
the field measure it.

## Why this is a hill worth taking

A contribution reaches the top tier when it is **necessary, not merely novel.**
Test the idea against that bar:

- **It names a real gap, not a marginal one.** PAGED, CAMB, KEO and the broader
  procedural-IE line measure step coverage and graph topology. The clause that
  makes a procedure *safe* — "*do not* open the valve *until* pressure < 5 bar",
  "the operator *must* wear a respirator *before* sampling" — is exactly the part
  they leave unscored. In safety-critical industrial procedures that omission is
  not cosmetic; it is the difference between a procedure a system can follow and
  one it can only paraphrase.
- **It is measurable.** Constraint attachment reduces to typed, enforcement-
  graded, step-anchored edges — a structure that admits precision/recall, fuzzy
  semantic alignment, and inter-annotator agreement. A gap you can define but not
  score is an essay; a gap you can score is a benchmark.
- **It generalizes past our own pipeline.** The benchmark judges any extractor.
  IPKE is one baseline on it, not its reason to exist. A resource that only
  measures its author's system is a leaderboard of one; a resource that any group
  can submit to is infrastructure.

Hold this line under pressure. When scope creep tempts (more documents, more
metrics, a bigger model), the question is always: *does this sharpen the
constraint-attachment claim, or dilute it?* Sharpen, or cut.

## What "elite" means here — the bar, concretely

The gap between a paper that is accepted and one that is *cited for a decade* is
not cleverness. It is discipline on a small number of axes:

1. **Reproducible to the digit.** A stranger clones the repo and one command
   regenerates the headline numbers, or the resource is not a resource. `make
   gold-pipeline` and `make eval` are load-bearing, not decoration. Pin every
   motivating number (the D1 constraint-blindness result) so drift fails loudly.
2. **Agreement that means something.** κ is only meaningful if the second pass
   was authored blind to the first. The anchoring control in
   `docs/annotation/independent-annotator-workflow.md` is not bureaucracy — it is
   the thing that lets us claim the golds are reproducible rather than one
   annotator's taste. Never let convenience erode it.
3. **Honest provenance.** Golds are model-assisted drafts adjudicated by an
   independent pass, stamped as exactly that until a human signs off. The moment
   we overclaim human authorship, a sharp reviewer finds it and the whole
   resource's credibility goes with it. Our credibility is the asset; protect it
   above all speed.
4. **A defensible, licensed corpus.** Every document is publicly licensed with
   recorded provenance (`datasets/paper/public_sources_manifest.csv`). Genre
   diversity beats raw count — eight procedures across eight domains that stress
   different constraint shapes is worth more than twenty near-duplicates.
5. **A datasheet, not just data.** Follow the Gebru et al. datasheet discipline:
   motivation, composition, collection, preprocessing, uses, limits. Elite
   resources tell you when *not* to use them.

Elite is not a bigger model or a longer paper. It is a smaller number of claims,
each of which survives an adversarial reviewer.

## The venue ladder

The near-term target is the **ECIR 2027 Resource Track** — the right first home
because the contribution *is* the resource. But the ambition is a multi-year arc,
and the resource paper is the foundation stone, not the building:

- **Now → ECIR 2027 (Resource).** Establish IPKE-Bench: taxonomy, validator,
  golds with IAA, baselines, the constraint-blindness motivating result.
  Abstract 12 Oct 2026, full paper 2 Nov 2026.
- **Next → a full research paper (SIGIR / EMNLP / *ACL).** Once the benchmark
  exists, the *method* paper becomes possible and far stronger: constraint-aware
  extraction architectures measured *on our own benchmark*, with the
  constraint-aware retrieval task (PKG-backed vs text-chunk RAG) as the payoff.
  A benchmark you created and then top the leaderboard of is a compounding asset.
- **Then → a journal extension (JWS, TOIS, or a Semantic-Web / data venue).**
  The expanded corpus, cross-lingual extension (Finnish SOPs via the CoRe
  pilots), JSON-LD / knowledge-graph export, and a longitudinal study of model
  progress on constraint attachment. This is where "highest-level journal" lives:
  the definitive, extended treatment of the problem you defined.
- **Landmark test.** The work becomes landmark not when it is published but when
  *other groups report constraint-attachment F1 on IPKE-Bench without being told
  to.* Design every decision so that outcome is possible: clean schema, permissive
  license, frictionless loader, a metric people trust.

Each rung must be genuinely earned. Do not skip to the journal before the
benchmark is trusted; do not chase the landmark before the method paper tops it.

## The research arc past the first paper

Directions that extend the thesis rather than dilute it — pursue in roughly this
order, and only when the current rung is solid:

- **Constraint-aware retrieval and QA.** "Under what condition may step 7 be
  skipped?" is a query a PKG can answer and a text-chunk index cannot. This is the
  clearest demonstration that constraint attachment *buys* something downstream.
- **Cross-lingual and cross-domain generalization.** Finnish industrial SOPs
  (native to the CoRe / TeoÄly / ADINO pilots) test whether the taxonomy is a
  property of procedures or an artifact of English regulatory prose.
- **Procedure-as-reference verification.** Checking *observed* work against a
  procedure — the operator-monitoring pilots — turns extraction into compliance
  checking. This is where the research meets the funded projects most directly.
- **Model progress as a measurement instrument.** A stable benchmark lets you
  chart how frontier and *local* models improve on constraint attachment over
  time — valuable precisely because it is privacy-preserving and runnable on the
  lab's own hardware.

## The CoRe link — stated narrowly, on purpose

IPKE-Bench supports the family of TeoÄly / ADINO pilots that turn unstructured
procedural or operator content into structured representations, or use a
procedure as a reference for checking observed work (Dinolift work-instruction
authoring, Konecranes inspection/assembly, JS-Group and TEHAA reporting). It does
**not** claim to underlie unrelated vision or time-series pilots. Overclaiming the
industrial link is the same failure mode as overclaiming human annotation — a
reviewer punishes the reach and discounts the real contribution. State the link
where it is true and defensible, and nowhere else.

## Principles to carry

- **One claim, defended completely, beats three claims defended partially.**
- **Reproducibility is a feature of the science, not the packaging.** If it does
  not regenerate on a clean clone, it is not done.
- **Protect credibility above speed.** Honest provenance, blind IAA, licensed
  data. These are slow; they are also the only durable moat.
- **Diversity over volume** in the corpus; **sharpness over surface** in the
  claims.
- **Build the resource others will use, not the demo that impresses once.** The
  measure of success is adoption, not acceptance.

---

*See also:* `README.md` (contributions summary),
`docs/paper/ipke-bench-resource-prd.md` (requirements),
`docs/paper/2026-07-04-execution-direction.md` (current issue board and work
order), `docs/methods/annotation-pipeline.md` (how the golds are made),
`docs/annotation/guidelines.md` (annotation decision procedure).
