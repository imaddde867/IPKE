# Explainium 2.0 - Intelligent Document Knowledge Extraction

Explainium converts unstructured technical, safety, compliance and operational documents into structured, validated knowledge. Runs fully locally with offline AI models and produces database-ready entities with confidence scoring.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-00a86b.svg)](https://fastapi.tiangolo.com)
[![Offline](https://img.shields.io/badge/processing-offline-success.svg)](https://github.com)

## 🎯 Overview

**Explainium 2.0** transforms complex documents into structured knowledge using a modern, streamlined architecture:

- **Unified AI Engine**: Single engine with pluggable extraction strategies (Pattern, NLP, LLM)
- **Async Processing**: High-performance document processing with concurrent operations
- **Multi-Format Support**: PDF, DOCX, TXT, images (OCR), audio, video (23+ formats)
- **Quality Assurance**: Confidence scoring, validation gates, and fallback mechanisms
- **Production Ready**: Environment-based configuration with health monitoring

## 🏗️ Architecture

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
├── database/                          # SQLAlchemy models & operations
├── middleware.py                      # Request logging & tracking
├── logging_config.py                  # Structured logging
└── exceptions.py                      # Custom exceptions
```

### Processing Pipeline
1. **Pattern Extraction**: Fast regex-based entity detection
2. **NLP Enhancement**: spaCy + embeddings for context understanding  
3. **LLM Analysis**: Mistral-7B for complex semantic extraction
4. **Auto-Selection**: Optimal strategy chosen based on document complexity

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/imaddde867/explainium-2.0.git
cd explainium-2.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Start the API Server
```bash
python -m uvicorn src.api.simplified_app:app --host 0.0.0.0 --port 8000 --reload
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

## 📋 API Endpoints

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
  "status": "success",
  "entities": [
    {
      "type": "PROCEDURE",
      "content": "Safety inspection protocol",
      "confidence": 0.92,
      "category": "process"
    }
  ],
  "confidence_score": 0.87,
  "processing_time": 2.34,
  "extraction_method": "nlp_extraction"
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional - override defaults
export EXPLAINIUM_ENVIRONMENT=production
export EXPLAINIUM_LOG_LEVEL=INFO
export EXPLAINIUM_MAX_FILE_SIZE=52428800  # 50MB
```

### Supported Formats
- **Documents**: PDF, DOCX, TXT, RTF, ODT
- **Images**: PNG, JPG, TIFF (with OCR)  
- **Audio**: WAV, MP3, M4A (with transcription)
- **Video**: MP4, AVI, MOV (audio extraction)
- **Archives**: ZIP (extract and process contents)

## 🎛️ Knowledge Categories

| Category | Description | Target Confidence |
|----------|-------------|-------------------|
| **Technical Specifications** | Parameters, measurements, equipment specs | 0.95 |
| **Risk & Safety** | Hazards, safety measures, PPE requirements | 0.90 |
| **Process Intelligence** | Workflows, procedures, step-by-step guides | 0.85 |
| **Compliance** | Regulations, standards, requirements | 0.80 |
| **Organizational** | Roles, responsibilities, personnel info | 0.75 |
| **Definitions** | Terms, explanations, knowledge base | 0.70 |

## 🔬 Development

### Local Development
```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Check code quality
python -c "from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine; print('✅ Engine OK')"
```

### Model Management
The system automatically downloads required models on first use:
- **Whisper**: Speech-to-text (audio processing)
- **BGE Embeddings**: Semantic similarity 
- **spaCy**: NLP processing
- **Mistral-7B**: Large language model (optional)

### Docker (Optional)
```bash
docker-compose up
```

## 📊 Performance Metrics

### Before vs After Architecture Cleanup

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Lines of Code** | 13,584 | ~7,500 | -44% |
| **AI Engines** | 6 separate | 1 unified | -83% |
| **API Complexity** | 312 lines | ~150 lines | -52% |
| **Processor** | 1,508 lines | ~400 lines | -73% |
| **Dependencies** | 23 packages | 18 packages | -22% |

### Processing Performance
- **Small Documents** (< 20 pages): ~2-5 seconds
- **Medium Documents** (20-100 pages): ~10-60 seconds  
- **Large Documents** (100+ pages): ~1-3 minutes
- **Confidence Scores**: Typically 0.70-0.95 depending on content

## 🛠️ System Requirements

### Minimum
- **Python**: 3.12+
- **RAM**: 4GB (basic processing)
- **Storage**: 2GB (for models)

### Recommended  
- **RAM**: 8GB+ (for LLM processing)
- **CPU**: Multi-core for parallel processing
- **GPU**: Optional (Metal/CUDA acceleration)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/imaddde867/explainium-2.0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imaddde867/explainium-2.0/discussions)
- **Documentation**: Built-in API docs at `/docs` endpoint

---

**Explainium 2.0** - Clean, fast, and intelligent document knowledge extraction 🚀
