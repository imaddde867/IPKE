# Explainium 2.0 - Intelligent Document Knowledge Extraction

Explainium converts unstructured technical, safety, compliance and operational documents into structured, validated knowledge. Runs fully locally with offline AI models and produces database-ready entities with confidence scoring.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-00a86b.svg)](https://fastapi.tiangolo.com)
[![Offline](https://img.shields.io/badge/processing-offline-success.svg)](https://github.com)


#  **_FIX CONFLICT BETWEEN THE ENGINE AND CENTRAL CONFIG SETTS._** 

## Overview

**Explainium 2.0** transforms complex documents into structured knowledge using a modern, streamlined architecture:

- **LLM-Only Engine**: Mistral-7B (via llama.cpp) powers every extraction step
- **Async Processing**: High-performance document handling with concurrent operations
- **Document Coverage**: PDF, DOCX, TXT plus OCR for common images and Whisper-backed audio transcription
- **Quality Assurance**: Confidence scoring with configurable quality thresholds
- **Operational Readiness**: Environment-based configuration with health monitoring

## Architecture

### Current Clean Structure
```
src/
├── ai/
│   └── unified_knowledge_engine.py    # Strategy pattern AI engine
├── api/
│   └── simplified_app.py              # Clean FastAPI application
├── core/
│   └── unified_config.py              # Environment-based configuration
├── processors/
│   └── streamlined_processor.py       # Async document processing
├── middleware.py                      # Request logging & tracking
├── logging_config.py                  # Structured logging
└── exceptions.py                      # Custom exceptions
```

### Extraction Flow
1. **Content Extraction**: Format-specific loaders turn PDFs, Office docs, spreadsheets, images, and audio into text
2. **LLM Analysis**: The Mistral-7B model generates structured entities from extracted text
3. **Validation & Scoring**: Results are filtered using configurable confidence thresholds

## Quick Start

### Installation
```bash
git clone https://github.com/imaddde867/explainium-2.0.git
cd explainium-2.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Start the API Server
```bash
python -m uvicorn src.api.simplified_app:app --host 127.0.0.1 --port 8000 --reload
```

### Access Points
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Basic Usage
```python
import requests

# Upload and process document
files = {"file": open("document.pdf", "rb")}
response = requests.post("http://localhost:8000/extract", files=files)

result = response.json()
print(f"Extracted {len(result['entities'])} entities")
print(f"Confidence: {result['confidence_score']:.2f}")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and system info |
| `/health` | GET | System health status |
| `/extract` | POST | Process document and extract knowledge |
| `/docs` | GET | Interactive API documentation |

### Extract Endpoint Example
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "document_id": "8f1cba23b4d1e5ab",
  "document_type": "manual",
  "entities": [
    {
      "content": "Safety inspection protocol",
      "entity_type": "procedure",
      "category": "process",
      "confidence": 0.92,
      "context": "Page 3 paragraph discussing inspection cadence..."
    }
  ],
  "confidence_score": 0.87,
  "processing_time": 2.34,
  "strategy_used": "llm_extraction",
  "metadata": {
    "file_format": ".pdf",
    "file_name": "document.pdf",
    "entities_extracted": 1
  }
}
```

## Configuration

### Environment Variables
```bash
# Optional - override defaults
export EXPLAINIUM_ENV=production
export EXPLAINIUM_LOG_LEVEL=INFO
export EXPLAINIUM_MAX_FILE_SIZE_MB=50
export EXPLAINIUM_DATABASE_URL="postgresql://postgres:password@localhost:5432/explainium"
```

### Supported Formats
- **Documents**: PDF, DOC, DOCX, TXT, RTF
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF (OCR required)
- **Spreadsheets**: CSV, XLS, XLSX
- **Presentations**: PPT, PPTX
- **Audio**: WAV, MP3, FLAC, AAC (Whisper transcription)

## Knowledge Categories

| Category | Description | Target Confidence |
|----------|-------------|-------------------|
| **Technical Specifications** | Parameters, measurements, equipment specs | 0.95 |
| **Risk & Safety** | Hazards, safety measures, PPE requirements | 0.90 |
| **Process Intelligence** | Workflows, procedures, step-by-step guides | 0.85 |
| **Compliance** | Regulations, standards, requirements | 0.80 |
| **Organizational** | Roles, responsibilities, personnel info | 0.75 |
| **Definitions** | Terms, explanations, knowledge base | 0.70 |

## Development

### Local Development
```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Check code quality
python -c "from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine; print('Engine OK')"
```

### Model Management
The system automatically downloads required models on first use:
- **Whisper**: Speech-to-text (audio processing)
- **BGE Embeddings**: Semantic similarity 
- **spaCy**: NLP processing
- **Mistral-7B**: Large language model 

### Processing Performance
- **Small Documents** (< 20 pages): ~2-5 seconds
- **Medium Documents** (20-100 pages): ~10-60 seconds  
- **Large Documents** (100+ pages): ~1-3 minutes
- **Confidence Scores**: Typically 0.70-0.95 depending on content

## System Requirements

### Minimum
- **Python**: 3.12+
- **RAM**: 4GB (basic processing)
- **Storage**: 2GB (for models)

### Recommended  
- **RAM**: 8GB+ (for LLM processing)
- **CPU**: Multi-core for parallel processing
- **GPU**: Optional (Metal/CUDA acceleration)
