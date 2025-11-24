# Prompting Grid Deep-Dive (Tier-A)

## Setup
- Config: `configs/prompting_grid.yaml` (DSC chunking, Mistral-7B, prompts P0–P3, Tier-A threshold 0.75).
- Evaluation rerun on existing predictions using `tools/evaluate.py` with the local `models/embeddings/all-mpnet-base-v2` encoder (`results/prompting_reval.json`).

## Re-evaluated Tier-A metrics
| Prompt | Document | StepF1 | Kendall τ | ConstraintCoverage | AdjacencyF1 | A_score |
| --- | --- | --- | --- | --- | --- | --- |
| P0_zero_shot | 3M_OEM_SOP | 0.553 | 0.757 | 0.000 | 0.176 | 0.317 |
|  | DOA_Food_Proc | 0.108 | 0.644 | 0.000 | 0.000 | 0.161 |
|  | op_firesafety_guideline | 0.255 | 0.769 | 0.100 | 0.353 | 0.280 |
| **Macro P0** | | **0.305** | **0.723** | **0.033** | **0.176** | **0.253** |
| P1_few_shot | 3M_OEM_SOP | 0.603 | 0.714 | 0.000 | 0.143 | 0.324 |
|  | DOA_Food_Proc | 0.203 | 0.476 | 0.625 | 0.000 | 0.469 |
|  | op_firesafety_guideline | 0.133 | 0.933 | 0.400 | 0.800 | 0.427 |
| **Macro P1** | | **0.313** | **0.708** | **0.342** | **0.314** | **0.407** |
| P2_cot | all docs | 0.000 | – | 0.000 | 0.000 | 0.000 |
| P3_two_stage | 3M_OEM_SOP | 0.619 | 0.689 | 0.750 | 0.000 | 0.699 |
|  | DOA_Food_Proc | 0.161 | 0.639 | 0.875 | 0.000 | 0.614 |
|  | op_firesafety_guideline | 0.351 | 0.821 | 0.500 | 0.143 | 0.520 |
| **Macro P3** | | **0.377** | **0.716** | **0.708** | **0.048** | **0.611** |

## Observations
1. **P3 two-stage dominates constraints**: Constraint coverage jumps from 0.03 (P0 macro) to 0.71 (P3), validating the parser-enforced schema split. StepF1 is only slightly lower than P1 despite producing richer constraint sets.
2. **P1 few-shot wins on 3M precision**: The curated marine example reduces hallucinated step counts (43 vs. 54 in P3) and boosts StepF1 to 0.603, while the newly completed fire-safety run keeps ConstraintCoverage at 0.40 and surprisingly high adjacency (0.80) despite lower precision.
3. **P2 chain-of-thought is broken**: All documents return zero predictions, pointing to parser failures or the model staying in “reasoning” mode without emitting JSON. This is the primary regression to fix before new thesis experiments.
4. **Adjacency still shallow**: Apart from the anomalously high 0.80 from the fire-safety P1 run (likely a side-effect of short merged sequences), all strategies rely on implicit sequential order from `_normalize_steps`, so adjacency F1 generally stays <0.35. Building explicit NEXT edges is now the limiting factor for Tier-B readiness.
5. **Constraint attachment F1 remains 0**: Even when coverage is high, linked step IDs rarely survive alignment because constraints still reference hallucinated or duplicated steps. Better deduplication and canonical IDs are required before attachment metrics can move.

## Failure Analysis
- **Step explosion (P0/P3)**: Stats files show 64–176 steps per doc, far above the 15–40 gold steps. Chunk boundary overlap and lack of post-merge dedup inflate false positives, crushing precision even when recall is strong.
- **Constraint drift**: `_normalize_constraints` (`src/ai/knowledge_engine.py`) rewrites IDs but preserves whatever step references survived chunk parsing. When chunks hallucinate IDs, attachments silently point to non-existent `step_id_map` entries, yielding empty `steps` arrays in the normalized constraints.
- **CoT parser gaps**: The Chain-of-Thought template (`src/ai/prompting/chain_of_thought.py`) asks for reasoning plus a `<json>` block, yet `_parse_json` never finds valid payloads. Logging raw generations for empty chunks and tightening the stop tokens around `<json>` will expose whether the model is truncating before the tag or emitting invalid JSON.
- **Evaluation drift**: Legacy `summary.csv` values were based on an older evaluator. Re-running with the current `tools/evaluate.py` produces much higher StepF1/Kendall for P3, so thesis tables should cite the re-evaluated metrics above.

## Recommended fixes & next experiments
1. **Instrument P2 pipeline**: Capture raw generations when `_extract_json_payload` returns `None`, and enforce `<json>` tags via an explicit stop sequence (e.g., `stop=['</json>']`). Once fixed, re-run the CoT rows to populate the table instead of zeros.
2. **Post-merge dedup**: After `_merge_chunk_results`, cluster steps by cosine similarity over sentence embeddings and keep the highest-confidence representative before `_normalize_steps`. This will shrink step counts and lift StepF1/Adjacency across all prompts.
3. **Constraint attachment repair**: When normalized constraints end up with empty `steps`, fall back to fuzzy matching on step text to re-link them. This, combined with dedup, should finally move `ConstraintAttachmentF1` off zero.
4. **Tier-B / graph build**: Use `src/graph/adapter.py` to emit nodes + NEXT/condition edges, then evaluate Tier-B via `tools/evaluate.py --tier both`. Include those results plus qualitative graph visualizations (e.g., `tools/visualize_graph.py`) in the Thesis knowledge-graph chapter.
5. **Model/RAG ablations**: Extend `configs/prompting_grid.yaml` with:
   - Backend swaps (`llm.backend: llama_cpp` for offline GGUF vs. `transformers` quantized 4-bit).
   - Retrieval-augmented chunks (e.g., insert retrieved similar sentences into the prompt context) and log their effect on ConstraintCoverage.
6. **Document coverage**: Re-run P3 on the fire safety doc (P1 is now complete). Use `scripts/run_prompting_experiments.py --config configs/prompting_grid.yaml --out-root logs/prompting_grid_rerun --evaluate true` after setting `LLM_BACKEND=llama_cpp` in `.env` to avoid transformer downloads inside the restricted environment.

Document every rerun (config + metrics) under `logs/<run_name>/` so Thesis Chapter 5 can reference reproducible artifacts.
