# Explainium – Intelligent Document Knowledge Extraction Platform

> Phase 1 (Foundation) of the **EXPLAINIUM Central Intelligence Hub** – the smart Knowledge Extraction core.

Explainium converts unstructured technical, safety, compliance and operational documents into structured, validated knowledge. It runs fully locally (offline models) and produces database‑ready entities with confidence and quality metrics so the extracted knowledge can be searched, filtered, audited, or exported.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Offline](https://img.shields.io/badge/processing-offline-success.svg)](https://github.com)
![EXPLAINIUM Central Intelligence Hub](https://img.shields.io/badge/EXPLAINIUM-Central%20Intelligence%20Hub-ffd200?style=flat&logo=brain&logoColor=black)

## Context: Central Intelligence Hub Alignment

This repository delivers the **Knowledge Extraction Foundation** (Phase 1) of the broader **EXPLAINIUM Central Intelligence Hub** – an industrial AI platform that unifies:

- Tacit Company Knowledge (documents, media, training assets)
- Multimodal Sensing (future phase: IoT, CV, telemetry fusion)
- Agent Generated Outputs (future phase: autonomous specialized AI agents)

The current scope focuses on high‑fidelity transformation of unstructured institutional knowledge into normalized, confidence‑scored entities – seeding the knowledge layer that subsequent phases (semantic cortex, agent orchestration, sensor fusion) will leverage.

### Roadmap Positioning

| Phase | Focus | Status |
|-------|-------|--------|
| 1. Foundation | Document & media ingestion, structured knowledge extraction | IN PROGRESS (this repo) |
| 2. Agent Network | Orchestration & specialized autonomous agents | Planned |
| 3. Multimodal Fusion | Real-time sensor + telemetry integration | Planned |

Lightweight hooks (environment flags, modular engines, model registry stubs) have been designed to enable forward compatibility with later phases without refactoring core extraction logic.

## Overview

Core goals:
1. Extract high‑value knowledge (specifications, processes, safety measures, compliance requirements, roles, definitions) from heterogeneous document formats.
2. Maintain a structured schema with traceable confidence scores and validation flags.
3. Provide a predictable processing pipeline with graceful fallback when advanced semantic extraction is unavailable.

## Processing Pipeline

Priority order:
1. Primary semantic engine (large local instruction model + multi‑prompt strategy)
2. Enhanced pattern / NLP extraction (specialised patterns & embeddings)
3. Lightweight legacy pattern matching (minimal emergency fallback)

Quality gates (examples):
- Minimum semantic extraction confidence: 0.75
- Entity validation threshold: 0.70
- Production readiness threshold: 0.85 aggregate confidence

## Knowledge Categories

| Category | Typical Content | Target Confidence |
|----------|-----------------|-------------------|
| Technical Specifications | Parameters, measurements, operating ranges | 0.95 |
| Safety / Risk Requirements | Hazards, mitigation measures, PPE | 0.90 |
| Process Intelligence | Steps, workflows, procedures | 0.85 |
| Compliance & Governance | Regulations, standards, mandatory items | 0.80 |
| Organizational Data | Roles, responsibilities, qualifications | 0.75 |
| Definitions / Terminology | Terms and explanations | 0.70 |

## Key Features

Extraction & Semantics:
- Multi‑prompt semantic analysis (role/targeted prompts per category)
- Relationship and context capture between extracted entities
- Confidence scoring + validation pass flags per entity

Quality & Governance:
- Hierarchical fallback with explicit method attribution
- Configurable thresholds for acceptance and production use
- Structured, normalized output ready for persistence / export

Operational:
- Local model execution (no external calls required once models are present)
- Multi‑format ingestion: PDF, DOCX, TXT, images (OCR), audio (transcription), video (extracted audio)
- Batch processing support with metadata tracking

Interface & Access:
- Streamlit dashboard for interactive review and filtering
- FastAPI backend with OpenAPI documentation
- Export utilities for downstream integration

## Architecture (Simplified Directory View)

```
explainium-2.0/
├── src/
│   ├── ai/                       # Semantic & extraction engines
│   │   ├── llm_processing_engine.py  # Primary semantic engine
│   │   ├── enhanced_extraction_engine.py
│   │   ├── knowledge_categorization_engine.py
│   │   ├── advanced_knowledge_engine.py
│   │   └── document_intelligence_analyzer.py
│   ├── processors/               # Document processing pipeline
│   │   └── processor.py
│   ├── api/                      # FastAPI backend
│   │   ├── app.py               # Main API application
│   │   └── celery_worker.py     # Background tasks
│   ├── frontend/                 # Streamlit interface
│   │   └── knowledge_table.py
│   ├── database/                 # Data models and persistence
│   ├── core/                     # Configuration and optimization
│   └── export/                   # Export utilities
├── models/                       # Local AI model assets
├── documents_samples/            # Sample documents for testing
├── docker/                       # Docker configuration
└── scripts/                      # Utility scripts
```

## Quick Start

Prerequisites:
- Python 3.12+
- 16 GB RAM recommended for larger model variants (smaller models also supported)
- macOS (Metal) or Linux with suitable CPU/GPU acceleration

Installation:
```bash
git clone https://github.com/imaddde867/explainium-2.0.git
cd explainium-2.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For research and development:
```bash
# Start the frontend directly
streamlit run src/frontend/knowledge_table.py

# Or start the API backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Optional: Run with Docker
docker-compose up
```

Access:
- Dashboard: http://localhost:8501 (Streamlit)
- API: http://localhost:8000 (FastAPI)
- API Docs: http://localhost:8000/docs

Health check:
```bash
./scripts/health_check.sh
```

## Environment Variables (Tuning)

| Variable | Purpose | Default |
|----------|---------|---------|
| EXPLAINIUM_LLM_CTX | LLM context window | 512 |
| EXPLAINIUM_LLM_BATCH | LLM batch size | 32 |
| EXPLAINIUM_LLM_THREADS | Inference threads | 4 |
| EXPLAINIUM_LLM_CHUNK_TIMEOUT | Per-chunk timeout (s) | 25 |
| EXPLAINIUM_LLM_CHUNK_RETRIES | LLM retries on timeout | 2 |
| EXPLAINIUM_LLM_CHUNK_BACKOFF | Initial backoff (s) | 3 |
| EXPLAINIUM_LLM_MAX_CHARS | Max chars per chunk (truncate) | (internal) |
| EXPLAINIUM_DISABLE_LLM / EXPLAINIUM_LLM_DISABLE | Disable LLM layer | unset |

Example:
```bash
export EXPLAINIUM_LLM_CTX=768
export EXPLAINIUM_LLM_CHUNK_TIMEOUT=35
./start.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| scripts/health_check.sh | Run readiness diagnostics |
| scripts/model_manager.py | Manage AI models and optimization |

## Configuration Snippets

Model configuration example:
```json
{
  "hardware_profile": "m4_16gb",
  "models": {
    "llm": {
      "path": "models/llm/Mistral-7B-Instruct-v0.2-GGUF",
      "quantization": "Q4_K_M",
      "context_length": 4096,
      "threads": 8
    }
  }
}
```

Threshold constants:
```python
LLM_MINIMUM = 0.75
ENHANCED_MINIMUM = 0.60
COMBINED_MINIMUM = 0.80
ENTITY_VALIDATION = 0.70
PRODUCTION_READY = 0.85
```

## API Usage Examples

Process a document via the orchestration layer:
```python
from src.processors.processor import OptimizedDocumentProcessor

processor = OptimizedDocumentProcessor()
result = processor.process_document_sync("/path/to/document.pdf")

print(result.entities_extracted, result.confidence_score)
```

Direct semantic engine invocation (async):
```python
from src.ai.llm_processing_engine import LLMProcessingEngine
import asyncio

async def run():
    engine = LLMProcessingEngine()
    await engine.initialize()
    out = await engine.process_document(
        content="Document text...",
        document_type="technical_manual",
        metadata={"filename": "manual.pdf"}
    )
    print(out.entities)

asyncio.run(run())
```

## Development & Research

Basic readiness test:
```bash
python -c "from src.ai.llm_processing_engine import LLMProcessingEngine;import asyncio;async def t():
    e=LLMProcessingEngine();await e.initialize();print('Engine ready')
asyncio.run(t())"
```

Model management:
```bash
# Setup models for your hardware
python scripts/model_manager.py --action setup --hardware-profile m4_16gb

# List available models
python scripts/model_manager.py --action list

# Validate model integrity
python scripts/model_manager.py --action validate
```

Quality / statistics probe:
```bash
python -c "from src.processors.processor import OptimizedDocumentProcessor; p=OptimizedDocumentProcessor(); print('Processor ready')"
```

Clean development environment:
```bash
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
rm -f logs/*.log
```

## License

MIT License – see [LICENSE](LICENSE).
