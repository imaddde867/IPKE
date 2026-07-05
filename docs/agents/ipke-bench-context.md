# IPKE-Bench Context for Agents

Tracked entry point for any AI agent (Claude Code, Codex, Cursor, ChatGPT) working in this repo. The personal `AGENTS.md` and `CLAUDE.md` at the repo root are gitignored — this file is the committed source of truth on framing and current state.

## What this repository delivers

**Primary contribution**: IPKE-Bench, a constraint-aware benchmark for procedural knowledge extraction from safety-critical industrial documents. Target venue: ECIR 2027 Resource Paper Track.

**Secondary**: IPKE, a local/private extraction pipeline that demonstrates the benchmark can be cleared. IPKE is the strong baseline, NOT the paper's contribution.

If you are reframing or summarising the project, lead with the benchmark, not the pipeline. The previous "Mistral-7B beats Llama-70B" narrative is supporting evidence at best, not the headline.

## §1 motivating result (live, regenerable)

LLM-drafted gold across 8 documents produced **3.66× fewer constraints** than the human-reviewed gold on the same source text (32 vs 117) — **thin-gold-era numbers**. After the deep re-annotation the golds hold 199 constraints, so the fixed draft is now cross-regime (current arithmetic ≈ 6.22×); the §1 framing is **under decision** per `docs/paper/D1_SCOPE_DECISION.md`. The expansion ratio (magnitude TBD) is the durable §1 claim.

Reproduce with:

```bash
make eval-blindness
```

Reports land in `datasets/paper/reports/constraint_blindness_v2_sbert{050,075}.json`. If you change the seed corpus, re-run and update `BENCHMARK.md`, `docs/dataset/datasheet.md`, and `docs/paper/ipke-bench-resource-prd.md` headline numbers.

## Mandatory workflow guards

Before editing any function, class, or method:

1. Run `gitnexus_impact({target: "symbolName", direction: "upstream"})`.
2. Report the blast radius to the user.
3. If HIGH or CRITICAL, warn explicitly before proceeding.

Before committing:

1. Run `gitnexus_detect_changes()` to confirm the changes only affect expected symbols.
2. Run `uv run python scripts/validate_paper_gold.py` if any `datasets/paper/gold/*.json` changed.
3. Run `uv run pytest -q --ignore=tests/test_api.py --ignore=tests/test_integration.py`.

Before claiming any work is complete:

1. Read the verification command output. Do not extrapolate.
2. State the result with evidence.

## Locked vocabularies (do not drift)

Defined in `CONTEXT.md` and `docs/annotation/constraint-types.md`:

- **review_status**: `unreviewed`, `reviewed`, `llm_draft`.
- **constraint.type**: `precondition`, `postcondition`, `guard`, `parameter`, `role_assignment`, `reference`.
- **constraint.enforcement**: `must`, `should`, `may`.

Any new value invalidates the paper-grade validator and the κ-grade IAA. If you encounter source content that doesn't fit, open a discussion in the corresponding issue and do NOT silently extend the vocabulary.

## Current open gates (in critical-path order)

1. **Recruit 4 independent annotators** for blind second-pass IAA. Memo drafted in the user's Obsidian vault. Lead time weeks.
2. **Corpus expansion** from 8 to 12 documents (genre-diverse: FAA AC 43.13-1B, FDA Food Code, NIST SP 800-61 Rev. 2, OEM service manual).
3. **D2 baseline sweep** once corpus and IAA close.
4. **D3 constraint-aware retrieval task** (optional).
5. **JSON-LD export** (for an ESWC fallback).

The user's manual action list is in `BENCHMARK.md` §Status. Don't recreate it.

## Don't do these without asking

- Reframe the contributions list — already singular (one artifact + ranked demonstrations). Don't re-expand.
- Re-annotate `datasets/paper/second_pass/*.json` to match gold — methodologically invalid, forbidden by `docs/annotation/guidelines.md`.
- Add ad-hoc constraint types — the taxonomy is locked.
- Touch `AGENTS.md` or `CLAUDE.md` and claim it persisted — they are gitignored.
- Edit the constraint-blindness headline numbers without regenerating the report first.

## See also

- `BENCHMARK.md` — top-level benchmark entry point.
- `CONTEXT.md` — domain glossary.
- `docs/paper/ipke-bench-resource-prd.md` — resource paper PRD.
- `docs/dataset/datasheet.md` — Gebru-format datasheet.
- `docs/annotation/constraint-types.md` — locked taxonomy.
- `docs/annotation/guidelines.md` — annotation procedure.
- `docs/annotation/independent-annotator-workflow.md` — recruited-annotator workflow.
- `docs/plans/2026-06-13-ipke-bench-taxonomy-and-review.md` — most recent sprint plan.
- `docs/adr/0004-ecir-resource-paper-primary.md` — venue decision.
- `docs/agents/domain.md` — how to consume domain docs.
- `docs/agents/issue-tracker.md` — issue conventions.
- `docs/agents/triage-labels.md` — label scheme.
