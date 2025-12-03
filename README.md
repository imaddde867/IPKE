# IPKE: Industrial Procedural Knowledge Extraction

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

**IPKE** is a research framework for extracting structured procedural knowledge from unstructured industrial documentation. It transforms static PDFs and manuals into queryable **Procedural Graphs (Tier-B)** and **Structured Flows (Tier-A)**.

Developed as a research thesis, this project explores the intersection of **Semantic Chunking** (Fixed, Breakpoint, DSC) and **Multi-Stage Prompting** to optimize extraction fidelity in low-resource environments.

![Efficiency Frontier](assets/efficiency_frontier_phi.png)

## Technical Highlights

- **Resource Efficiency:** Achieves high-fidelity extraction using optimized 7B parameter local models through architectural optimization rather than raw model scale.
- **Deep Logic Extraction:** Automates the detection of complex dependencies (AND/OR/XOR gates) and conditional branching, enabling structural analysis of procedures.
- **Dual Semantic Chunking (DSC):** Implements a specialized chunking algorithm designed to preserve long-range semantic coherence in technical documentation.

## Architecture Overview

```
.
├── main.py                 # FastAPI backend entry point
├── streamlit_app.py        # Interactive Research Dashboard
├── src/                    # Core Research Implementation
│   ├── ai/                 # Inference Engines & Prompting Strategies
│   ├── api/                # API Interface
│   ├── core/               # Unified Configuration
│   ├── evaluation/         # Metrics (Tier-A/B, Smatch, Procedural Fidelity Score)
│   ├── graph/              # Procedural Graph Topology
│   ├── pipelines/          # Extraction Pipelines
│   └── utils/              # Analytical Utilities
├── scripts/                # Experimentation Harness
│   └── experiments/        # Dockerized Research Experiments
├── configs/                # Experimental Configurations
└── assets/                 # Research Figures
```

## Quick Start

### 1. Environment Setup

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configuration

Copy `.env.example` to `.env` and configure resources:

```ini
# Example .env
GPU_BACKEND=metal       # or 'cuda' for NVIDIA acceleration
LLM_MODEL_PATH=models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf
CHUNKING_METHOD=dual_semantic
PROMPTING_STRATEGY=P3
```

### 3. Usage

**Research Dashboard (Streamlit):**
```bash
streamlit run streamlit_app.py
```

**API Endpoint (FastAPI):**
```bash
python main.py
# Documentation: http://localhost:8000/docs
```

## Reproducing Experiments

To replicate the thesis findings regarding chunking and prompting efficacy:

```bash
python scripts/experiments/run_all_chunking_experiments.py \
  --documents datasets/archive/test_data/text/3m_marine_oem_sop.txt \
              datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt \
              datasets/archive/test_data/text/op_firesafety_guideline.txt
```

Key performance indicators include **StepF1** (Step Recognition), **GraphF1** (Topology Alignment), and the **Procedural Fidelity Score (Φ)**. See `scripts/experiments/README.md` for detailed instructions.

## License

Research Thesis License. This codebase is provided for academic and research purposes. See `LICENSE` for details.