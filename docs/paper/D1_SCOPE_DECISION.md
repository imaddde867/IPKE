# D1 (constraint-blindness) scope decision — OPEN

**Status:** unresolved · owner: Imad · blocks: §1 motivating number, `make repro-blindness` re-pin

## The situation
`make eval-blindness` compares a **fixed LLM draft** (`D1_DRAFT_REF =
2379c8ef…`, 32 constraints) against the **committed golds** and reports how many
gold constraints the draft recovered (recall) and how much the human expanded on
the draft (expansion ratio).

That draft was generated at the **old thin-gold scope**. After this session's
deep re-annotation the golds moved to **full-subprocedure scope** (117 → 199
reviewed constraints, 5.9× deeper). So the comparison is now **cross-regime**:

| metric            | thin-gold era (v1) | current (deep golds) |
|-------------------|--------------------|----------------------|
| reviewed_total    | 117                | 199                  |
| recovered @0.75   | 24                 | 12                   |
| recall @0.75      | 0.2051             | 0.0603               |
| expansion ratio   | 3.66×              | 6.22×                |

The 6.22× is arithmetically correct but the draft and gold now come from
**different annotation rules**, so it is not yet an apples-to-apples claim.

## Options for §1 (pick one, then re-pin)
1. **Regenerate the draft at the new scope** — re-run the model on the 8 docs
   under full-subprocedure scope, recompute blindness. Needs model compute.
   Cleanest current number; apples-to-apples.
2. **Reframe §1 around corpus depth** — the thin→deep story (43/117 → 251/199)
   is already valid and needs no draft. Retire the draft-vs-gold ratio as the
   headline; keep it as a secondary illustration.
3. **Scope the ratio to v1** — report the 3.66× against the archived thin golds
   (`datasets/paper/gold_v1_bounded_excerpt_archive/`), clearly labelled "v1
   bounded-excerpt scope." Honest and cheap; smaller number.

## After deciding
- Re-pin the chosen numbers behind a **non-gate** `repro-blindness` target
  (assertions belong on a locked experiment, not on the pre-push gate).
- Update PRD §D1 and datasheet to match the chosen framing.
- `make gold-pipeline` stays gated on **structural validity only** (always
  passable on a clean clone); blindness regenerates + prints for inspection.
