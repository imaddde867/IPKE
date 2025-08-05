# EXPLAINIUM Documentation

## Quick Links

- **[API Documentation](api.md)** - REST API endpoints and usage
- **[Database Schema](database.md)** - Database structure and relationships
- **[Deployment Guide](deployment.md)** - Setup and production deployment
- **[Processing Pipeline](processing.md)** - Document processing and AI analysis

## Getting Started

### 1. Setup and Installation
Follow the [Deployment Guide](deployment.md) for:
- Local development setup
- Production deployment
- Docker configuration
- Environment variables

### 2. API Usage
See [API Documentation](api.md) for:
- Document upload and processing
- Knowledge extraction endpoints
- Graph generation and querying
- Search functionality

### 3. Understanding the System
Review [Processing Pipeline](processing.md) for:
- Multi-modal content extraction
- AI-powered knowledge analysis
- Relationship mapping algorithms
- Graph generation process

### 4. Database Structure
Check [Database Schema](database.md) for:
- Table relationships
- Data models
- Indexing strategy
- Migration management

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   AI Processing  │───▶│  Knowledge DB   │
│ (PDF, DOCX,     │    │ (NLP, ML Models) │    │ (PostgreSQL)    │
│  Videos, etc.)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │◀───│ Knowledge Graph  │◀───│   Search Index  │
│   (React UI)    │    │   (NetworkX)     │    │ (Elasticsearch) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Features

### 🧠 Knowledge Extraction
- Tacit knowledge detection from enterprise documents
- Decision pattern recognition and optimization identification
- Workflow dependency mapping with confidence scoring

### 🔗 Relationship Mapping
- Process dependencies (prerequisite, parallel, downstream, conditional)
- Equipment-maintenance correlations (preventive, corrective, predictive)
- Skill-function links with proficiency assessment
- Compliance-procedure connections for regulatory requirements

### 📊 Graph Analytics
- Interactive knowledge graph visualization
- Critical path analysis and bottleneck detection
- Complex relationship queries and subgraph analysis

## Technology Stack

**Backend:** Python, FastAPI, PostgreSQL, Redis, Celery  
**AI/ML:** Hugging Face Transformers, spaCy, NetworkX, Apache Tika  
**Frontend:** React, D3.js  
**Infrastructure:** Docker, Elasticsearch

## Support and Contributing

- **Issues:** [GitHub Issues](https://github.com/imaddde867/explainium-2.0/issues)
- **Discussions:** [GitHub Discussions](https://github.com/imaddde867/explainium-2.0/discussions)
- **Contributing:** See main README for contribution guidelines

## Version Information

**Current Version:** 2.0 - Relationship Mapping System  
**Test Coverage:** 54+ test cases with 95%+ code coverage  
**Documentation Status:** Complete and up-to-date