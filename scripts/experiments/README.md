## Chunking Experiment Harness

The `scripts/experiments` directory provides reproducible sweeps for every chunking strategy exposed by the API. Each runner:

- Writes a docker-compose override with the chosen hyper-parameters.
- Restarts **only** the target container (`--service-name` / `--container-name`) so the other methods stay untouched.
- Waits for `/health`, runs HTTP extraction on the selected documents, converts the payload to Tier‑B graphs, and stores everything under `results/experiments/<method>/<config_id>/`.
- Captures Tier‑A and Tier‑B metrics with `tools/evaluate.py`, logs docker output, and records provenance metadata (git SHA, command line, env overrides, duration, document-level chunk stats).
- Appends a CSV summary with the clean headline metrics you requested: `StepF1`, `AdjacencyF1`, `Kendall`, `ConstraintCoverage`, `ConstraintAttachmentF1`, `A_score`, `GraphF1`, `NEXT_EdgeF1`, `Logic_EdgeF1`, `ConstraintAttachmentF1`, `B_score`.

> Tier-A headline: `A_score = 0.7 * StepF1 + 0.3 * AdjacencyF1`  
> Tier-B headline: `B_score = GraphF1`

### Metric details

- **Tier A (flat extraction):** StepF1, AdjacencyF1, Kendall τ_b, Constraint Coverage, Constraint Attachment F1, and the composite `A_score`. Constraint metrics follow Carriero et al. (2024) and Xu et al. (2024) to decouple detection vs. linkage accuracy.
- **Tier B (graph extraction):** GraphF1 (Smatch-style), NEXT EdgeF1, Logic EdgeF1, Constraint Attachment F1, AdjacencyF1, Kendall. Predictions are auto-converted to Tier‑B via `src.graph.adapter.flat_to_tierb`.
- All reports are written to `<run_dir>/metrics/tier_a.json` and `tier_b.json`, and macro averages feed the summary CSVs.

### Scripts

| Script | Purpose | Default grid |
| --- | --- | --- |
| `fixed_sweep.py` | Sweep `CHUNK_MAX_CHARS` (and optional `CHUNK_STRIDE_CHARS`) for the fixed chunker. | `CHUNK_MAX_CHARS = [1000, 1500, 2000, 3000]` |
| `semantic_sweep.py` | Vary breakpoint semantic parameters one axis at a time. | `SEM_LAMBDA = {0.05, 0.15, 0.25}`, `SEM_WINDOW_W = {20,30,40}`, `SEM_MIN_SENTENCES_PER_CHUNK = {1,2,3}`, `SEM_MAX_SENTENCES_PER_CHUNK = {30,40,60}` |
| `dsc_sweep.py` | Explore DSC hierarchy settings (parent span bounds, delta window, threshold, heading toggle). | `DSC_PARENT_MIN_SENTENCES = {5,10,15}`, `DSC_PARENT_MAX_SENTENCES = {80,120,160}`, `DSC_DELTA_WINDOW = {15,25,35}`, `DSC_THRESHOLD_K = {0.8,1.0,1.2}`, `DSC_USE_HEADINGS = {true,false}` |

Each script shares the same ergonomic flags:

```
python scripts/experiments/fixed_sweep.py \
  --documents datasets/archive/test_data/text/3m_marine_oem_sop.txt \
  --gold-tier-a datasets/archive/gold_human \
  --gold-tier-b datasets/archive/gold_human_tierb \
  --timeout 2000 \
  --skip-existing
```

Other common options:

- `--doc-id-map <json>`: map file stems to gold IDs (already handles `3m_marine_oem_sop -> 3M_OEM_SOP`).
- `--service-name / --container-name`: override compose identifiers if you renamed them.
- `--compose-file`: point to an alternate compose stack.
- `--host / --port`: target a remote deployment rather than `localhost`.

### General methodology baked into the scripts

1. **Reset per configuration** – `docker compose up -d --force-recreate <service>` is executed after writing the override to guarantee a clean graph-state for the chunker being tuned.
2. **Isolate externals** – other services stay untouched, satisfying the “keep all other services at baseline values” constraint.
3. **Archive provenance** – `metadata.json` contains the git SHA, CLI invocation, env overrides, wall-clock time, prediction/tier-B file paths, evaluator reports, docker log path, and per-document chunk statistics.
4. **Clean outputs** – predictions follow gold filenames so `tools/evaluate.py` can be run directly; Tier‑B conversions live under `<run_dir>/tierb/`.
5. **Evaluation coverage** – Tier A + Tier B metrics are always generated, and the summary CSV collates the requested KPIs plus the derived `A_score` and `B_score`.
6. **Docker logs** – each configuration stores the relevant container logs (starting from the moment before extraction) under `<run_dir>/logs/docker.log` for traceability.

### Recommended workflow

1. **Fixed sweep:** `python scripts/experiments/fixed_sweep.py --timeout 2000`. Inspect the CSV to pick the chunk size/stride pair balancing chunk count, confidence, and Tier-A/Tier-B accuracy.
2. **Breakpoint semantic sweep:** `python scripts/experiments/semantic_sweep.py` while noting the macro averages; plot `SEM_LAMBDA` vs `A_score` if needed.
3. **DSC sweep:** `python scripts/experiments/dsc_sweep.py` (or rerun with restricted `--documents` if you want Doc‑specific Latin-square assignments).
4. **Document the winning settings:** copy the env tuples from `metadata.json` into your lab notebook / thesis appendix.
5. **Final 3×3 grid:** Once the “best” configuration per method is fixed, restart all three services with those env values and run `python scripts/run_thesis_experiments.py --timeout 2000 --output-dir results/thesis_final_<date> --skip-existing`. The resulting `thesis_summary.json` plus the evaluation summaries from the sweep form the clean experimental record for your research chapter.

Feel free to mix and match inputs (e.g., run semantic sweeps on just one doc by passing `--documents path/to/doc.txt`) or tighten the grids to zoom into promising regions. The helpers intentionally keep everything ASCII and scriptable for lab automation / CI.*** End Patch
