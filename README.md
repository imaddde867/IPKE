# IPKE

**Industrial Procedural Knowledge Extraction**

A research framework that transforms unstructured industrial documentation into machine-actionable Procedural Knowledge Graphs (PKG). Built for resource-constrained deployment without sacrificing extraction quality.

![Efficiency Frontier](assets/efficiency_frontier_phi.png)

## Why IPKE?

Large language models struggle with structured extraction from technical documents. Even 70B parameter models fail at zero-shot procedural parsing due to schema non-compliance. IPKE solves this through task-decomposed prompting—achieving higher fidelity with a 7B model than larger models achieve with brute-force scaling.

## Features

- **Dual Semantic Chunking** — Preserves long-range coherence in technical documentation
- **Multi-Stage Prompting** — Four strategy tiers (P1–P4) for precision-speed tradeoffs
- **Interactive PKG Visualization** — Hierarchical graphs with constraint nodes and decision gateways
- **Comprehensive Evaluation** — StepF1, GraphF1, and Procedural Fidelity Score (Φ)

## Quick Start

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py
```

## Configuration

```ini
# .env
GPU_BACKEND=metal                # metal | cuda | cpu
CHUNKING_METHOD=dual_semantic    # fixed | semantic_breakpoint | dual_semantic
PROMPTING_STRATEGY=P3            # P1 | P2 | P3 | P4
```

## Project Structure

```
src/
├── ai/           # LLM inference, prompting strategies
├── evaluation/   # Extraction metrics
├── graph/        # PKG schema, Neo4j connector
├── pipelines/    # Orchestration
├── processors/   # Document chunking
└── utils/        # Visualization
```

## Reproducing Experiments

```bash
python scripts/experiments/run_all_chunking_experiments.py \
  --documents datasets/archive/test_data/text/*.txt
```

## API

```bash
python main.py
# http://localhost:8000/docs
```

## License

Research thesis. Academic use. See `LICENSE`.

---

Turku University of Applied Sciences · 2025
