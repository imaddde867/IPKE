# D1 (constraint-blindness) scope decision — RESOLVED (option 2, 2026-07-06)

**Status:** resolved · owner: Imad · decision: **option 2 — reframe §1 around corpus depth**

## Decision (2026-07-06)

**§1 leads with the corpus-depth story** (thin bounded-excerpt 43 steps / 117
constraints → full-subprocedure 256 / 231 under the locked scope rule); the
draft-vs-gold ratio is retired as the headline and kept as a clearly labelled
**cross-regime secondary illustration** (32 vs 231, 7.22×), pinned in the
non-gate `make repro-blindness` target.

Why not option 1 (regenerate the draft at the new scope), even with compute
available:

- The deep golds were themselves produced by the strong-model drafting harness
  plus adjudication. Re-running that same harness at the new scope yields a
  draft ≈ the pre-adjudication gold, so the ratio collapses toward 1× — the
  comparison stops measuring blindness and starts measuring adjudication delta.
- Substituting a weaker local model makes the model choice itself the claim
  ("you picked a weak model"), a new methodological attack surface.
- Any ratio computed against the current golds is provisional until human
  sign-off anyway (paper-valid numbers require signed-off gold).

The **apples-to-apples blindness number the paper should cite** is already in
the experiment plan: the **D2 P0 zero-shot local baseline's ConstraintCoverage
/ ConstraintAttachmentF1** on the signed-off benchmark — the benchmark itself
measures the blindness, no separate draft pipeline needed. §1 cites that once
D2 runs.

## The situation
`make eval-blindness` compares a **fixed LLM draft** (`D1_DRAFT_REF =
2379c8ef…`, 32 constraints) against the **committed golds** and reports how many
gold constraints the draft recovered (recall) and how much the human expanded on
the draft (expansion ratio).

That draft was generated at the **old thin-gold scope**. After this session's
deep re-annotation the golds moved to **full-subprocedure scope** (117 → 199
reviewed constraints, 5.9× deeper). So the comparison is now **cross-regime**:

| metric            | thin-gold era (v1) | deep golds (2026-07-05) | deep golds, verbatim-grounded (2026-07-06) |
|-------------------|--------------------|-------------------------|--------------------------------------------|
| reviewed_total    | 117                | 199                     | 231                                        |
| expansion ratio   | 3.66×              | 6.22×                   | 7.22×                                      |

(2026-07-06 column: agent verbatim-grounding + completion pass over all 8
golds — spans corrected, missed constraints added, restatements dropped; see
`docs/annotation/SIGN_OFF_ISSUE.md` and each gold's `review_notes`. Per-type
recall of the fixed draft against the current golds is in
`datasets/paper/reports/constraint_blindness_v2_sbert050.json`.)

The 7.22× is arithmetically correct but the draft and gold now come from
**different annotation rules**, so it is not yet an apples-to-apples claim.

## Options considered (for the record)
1. **Regenerate the draft at the new scope** — re-run the model on the 8 docs
   under full-subprocedure scope, recompute blindness. Needs model compute.
   Cleanest current number; apples-to-apples.
2. **Reframe §1 around corpus depth** — the thin→deep story (43/117 → 256/231)
   is already valid and needs no draft. Retire the draft-vs-gold ratio as the
   headline; keep it as a secondary illustration.
3. **Scope the ratio to v1** — report the 3.66× against the archived thin golds
   (`datasets/paper/gold_v1_bounded_excerpt_archive/`), clearly labelled "v1
   bounded-excerpt scope." Honest and cheap; smaller number.

## Follow-through (done 2026-07-06)
- ✅ Re-pinned behind the **non-gate** `make repro-blindness` target
  (32 vs 231; recall 0.0606 @0.75 / 0.3766 @0.50; expansion 7.2188).
- ✅ PRD §Problem and BENCHMARK.md updated to the corpus-depth framing with the
  cross-regime illustration labelled as such.
- ✅ `make gold-pipeline` stays gated on **structural validity only**;
  blindness regenerates + prints for inspection.
- ⏳ After D2 runs on signed-off gold: cite P0 ConstraintCoverage as the
  apples-to-apples §1 motivator; keep 7.22× as the annotation-economics note.
