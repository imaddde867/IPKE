## Project Title
IPKE - Industrial Procedural Knowledge Extraction

## Author line
Imad Eddine Elmouss - BSc Information and Communication Technology (Applied AI/ML + Data Engineering), Turku UAS (2025)
Thesis (Theseus): https://www.theseus.fi/handle/10024/907018

## Hero image
![IPKE Streamlit UI](https://raw.githubusercontent.com/imaddde867/IPKE/main/datasets/Samples/UI_Frontend_screenshot.png)

## Short Description
IPKE is an on-prem pipeline that extracts Procedural Knowledge Graphs (PKGs) from industrial SOPs using quantized 7B-class LLMs. It ships a Streamlit UI, a FastAPI API, and a metrics suite for chunking/prompting experiments.

## Full Description
This project focuses on reliable, local extraction of procedural structure from safety-critical manuals: ordered steps, entities/resources, and constraints (e.g., guards and warnings). The core idea is to reduce schema drift by combining Dual Semantic Chunking (DSC) with P3 (Two-Stage Decomposition): first extract an ordered step list, then attach constraints to the correct steps. Outputs can be visualized as interactive graphs and stored in Neo4j via a containerized deployment. An evaluation suite computes step/adjacency F1, constraint coverage, and a composite Procedural Fidelity score.

Core math snapshot (from `src/processors/chunkers/breakpoint.py`, `src/processors/chunkers/dual_semantic.py`, and `src/evaluation/metrics.py`):

```text
Sentence embeddings are L2-normalized, so cosine(e_i, e_{i+1}) = e_i dot e_{i+1}.

Breakpoint chunker cohesion(start,end) = mean_{k=start}^{end-2} (e_k dot e_{k+1})
DP objective uses: score += cohesion(start,end) - lambda*(preferred_len / chunk_len)

Dual Semantic parents use distances d_i = 1 - (e_i dot e_{i+1}) and split when:
d_{idx-1} > mean(local_window) + k*std(local_window) OR heading_regex(sentence[idx]) matches.

StepF1 uses Hungarian 1:1 alignment on cosine similarity with threshold tau=0.75.
Phi = 0.5*ConstraintCoverage + 0.3*StepF1 + 0.2*Kendall
```

## Pipeline or Architecture
1. Document ingestion (PDF/DOCX/PPTX/TXT) and text normalization.
2. Chunking (DSC / fixed / breakpoint) to keep chunks coherent under mid-sized LLM context limits.
3. LLM-based extraction using pluggable prompting strategies (P3, zero-shot, few-shot, chain-of-thought).
4. Schema + constraint validation to keep outputs queryable and consistent.
5. PKG build + canonicalization (NEXT/parallel/gateway relations, GUARD attachments for safety rules).
6. Serving + visualization via Streamlit, FastAPI (`/extract`, `/config`, `/stats`), and Neo4j (`docker-compose.yml`).

## Dataset/Training Snapshot
Gold labels shipped in `datasets/archive/gold_human/*.json`:

| Document | Steps | Constraints | NEXT edges | Gateways | Parallel groups | Sentence span coverage |
|---|---:|---:|---:|---:|---:|---:|
| 3M_OEM_SOP | 29 | 0 | 28 | 2 | 0 | 24 / 86 (0.279) |
| DOA_Food_Man_Proc_Stor | 40 | 3 | 11 | 1 | 10 | 40 / 641 (0.062) |
| op_firesafety_guideline | 31 | 8 | 12 | 1 | 3 | 40 / 156 (0.256) |
| Total | 100 | 11 | 51 | 4 | 13 | 104 / 883 (0.118) |

Inference configuration defaults (from `.env.example` and `models/llm/m4_optimization.json`):

| Setting | Value |
|---|---|
| LLM | Mistral-7B-Instruct-v0.2 (GGUF) |
| Quantization | Q4_K_M |
| Inference backend | llama.cpp (via llama-cpp-python) |
| Embeddings | SBERT (all-mpnet-base-v2) |
| Context window | 8192 (default), 4096 (Apple preset) |
| Temperature | 0.1 |
| Max output tokens | 1536 |
| GPU backend | auto (metal/cuda/cpu) |
| GPU layers | -1 (all layers) |
| Chunk size | 2000 |
| Threads (Apple preset) | 8 |
| Batch size (Apple preset) | 4 |
| Max workers | 8 |
| Random seed | 42 |

## Evaluation/Tracking Snapshot
Key thesis results summarized in `README.md`:

| Method | Scope | Step F1 | Constraint coverage | Phi |
|---|---|---:|---:|---:|
| P3 (Two-Stage Decomposition) | Tier-A documents | 0.377 | 0.708 | 0.611 |
| Llama-3.1-70B (zero-shot) | 3M SOP | - | 0.000 | 0.187 |
| P3 (Two-Stage Decomposition) | 3M SOP | - | 0.750 | 0.699 |
| P3 (earlier setup) | 3M SOP | - | 0.500 | 0.439 |

## Visual Evidence and Artifacts
![PKG schema](https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/figure_pkg_schema.png)

![Extracted PKG example graph (3M OEM SOP)](https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/graph_example.png)

![Efficiency frontier (Phi)](https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/efficiency_frontier_phi.png)

![Chunking comparison](https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/chunking_comparison_chart.png)

![Prompting comparison](https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/prompting_comparison_chart.png)

## Engineering Highlights
- On-prem, privacy-preserving inference via quantized GGUF models (no external APIs required).
- Reproducible experiment harness for chunking/prompting sweeps (`scripts/experiments/*`) plus plotting scripts that generate publication-grade figures (`scripts/plot_*.py`).
- Clean separation of concerns: chunkers, prompting strategies, graph building, validation, and evaluation metrics live in dedicated modules under `src/`.
- Dual interfaces: Streamlit for interactive exploration and FastAPI for programmatic extraction (`/extract` + OpenAPI docs).
- Neo4j integration via `docker-compose.yml` for queryable PKG storage and downstream decision-support use.

## Tech Stack
Python, FastAPI, Streamlit, Neo4j, Docker Compose, llama-cpp-python, transformers, sentence-transformers, spaCy, PyTorch, NetworkX, pandas, PyMuPDF, python-docx, python-pptx, PyVis, jsonschema, pytest

## Demo Video
<video controls src="https://raw.githubusercontent.com/imaddde867/IPKE/main/assets/demo.mp4" poster="https://raw.githubusercontent.com/imaddde867/IPKE/main/datasets/Samples/UI_Frontend_screenshot.png"></video>

## Try It
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
docker compose up -d
streamlit run streamlit_app.py
```
