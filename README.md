# Explainium 2.0 – Intelligent Document Knowledge Extraction Platform

Explainium converts unstructured technical, safety, compliance and operational documents into structured, validated knowledge. It runs fully locally (offline models) and produces database‑ready entities with confidence and quality metrics so the extracted knowledge can be searched, filtered, audited, or exported.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Offline](https://img.shields.io/badge/processing-offline-success.svg)](https://github.com)

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
├── src/ai/                       # Semantic & extraction engines
│   ├── llm_processing_engine.py  # Primary semantic engine
│   ├── enhanced_extraction_engine.py
│   ├── knowledge_categorization_engine.py
│   ├── advanced_knowledge_engine.py
│   └── document_intelligence_analyzer.py
├── src/processors/               # Orchestration / pipeline
│   └── processor.py
├── src/database/                 # Models and CRUD operations
├── src/frontend/                 # Streamlit interface
├── models/                       # Local model assets
└── documents_samples/            # Sample input documents
```

## Quick Start

Prerequisites:
- Python 3.12+
- 16 GB RAM recommended for larger model variants (smaller models also supported)
- macOS (Metal) or Linux with suitable CPU/GPU acceleration

Installation:
```bash
git clone https://github.com/your-org/explainium-2.0.git
cd explainium-2.0
chmod +x setup.sh
./setup.sh
./start.sh
```

Access:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

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
from src.processors.optimized_processor import OptimizedDocumentProcessor

processor = OptimizedDocumentProcessor()
processor.optimize_for_m4()
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

## Development & Verification

Basic readiness test:
```bash
python -c "from src.ai.llm_processing_engine import LLMProcessingEngine;import asyncio;async def t():
    e=LLMProcessingEngine();await e.initialize();print('Engine ready')
asyncio.run(t())"
```

Quality / statistics probe:
```bash
python -c "from src.processors.optimized_processor import OptimizedDocumentProcessor; p=OptimizedDocumentProcessor(); p.optimize_for_m4(); print('ok')"
```

## License

MIT License – see [LICENSE](LICENSE).
