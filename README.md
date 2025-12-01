# IPKE: Industrial Procedural Knowledge Extraction

**IPKE** is a research framework designed for high-fidelity extraction of procedural knowledge from unstructured industrial documentation. This system represents a significant advancement in the field of technical document understanding, transforming static PDFs and manuals into queryable **Procedural Graphs (Tier-B)** and **Structured Flows (Tier-A)**.

Developed as part of my thesis research, IPKE introduces novel methodologies at the intersection of **Semantic Chunking** (Fixed, Breakpoint, DSC) and **Multi-Stage Prompting** (Zero-shot to Schema-aware Two-stage), successfully optimizing the efficiency-fidelity frontier in low-resource environments.

![Efficiency Frontier](assets/efficiency_frontier_phi.png)

## Research Contributions

- **Optimized Efficiency Frontier:** Demonstrates that strategic prompting (P3) combined with semantic chunking can achieve performance comparable to 70B+ parameter models using efficient 7B parameter local models.
- **Dual-Backend Architecture:** A robust, asynchronous worker pool supporting `llama.cpp` (Metal/CUDA) and `transformers` (CUDA), enabling flexible deployment across diverse hardware constraints.
- **Tier-B Graph Reconstruction:** A novel approach to extracting complex logical dependencies (AND/OR/XOR), connectivity (Next/Condition), and parametric constraints, surpassing traditional flat extraction methods.
- **Adaptive Semantic Chunking:** Introduction of the **Dual Semantic Chunking (DSC)** algorithm, utilizing `all-mpnet-base-v2` embeddings to preserve semantic coherence in technical manuals.

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

## Quick Start for Researchers

### 1. Environment Setup

```bash
# Create research environment
python3 -m venv .venv
source .venv/bin/activate

# Install research dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configuration

Copy `.env.example` to `.env` and configure your computational resources:

```ini
# Example .env
GPU_BACKEND=metal       # or 'cuda' for NVIDIA acceleration
LLM_MODEL_PATH=models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf
CHUNKING_METHOD=dual_semantic
PROMPTING_STRATEGY=P3
```

### 3. Interactive Analysis

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

Key performance indicators include **StepF1** (Step Recognition), **GraphF1** (Topology Alignment), and the **Procedural Fidelity Score (Φ)** (composite Tier-A fidelity). See `scripts/experiments/README.md` for detailed sweep instructions.

## License

Research Thesis License. This codebase is provided for academic and research purposes. See `LICENSE` for details.
