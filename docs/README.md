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

## Clean Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   AI Processing  │───▶│  Knowledge DB   │
│ (PDF, DOCX,     │    │ (Consolidated    │    │ (PostgreSQL)    │
│  Videos, etc.)  │    │  Extractor)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │◀───│ FastAPI Backend  │◀───│   Redis Cache   │
│   (React UI)    │    │ (Single app.py)  │    │ (Task Queue)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Current File Structure

```
src/
├── ai/
│   └── knowledge_extractor.py    # Consolidated AI engine
├── api/
│   ├── app.py                    # Main FastAPI application
│   └── celery_worker.py          # Task queue worker
├── core/
│   └── config.py                 # Centralized configuration
├── database/
│   ├── crud.py                   # Database operations
│   ├── database.py               # Database management
│   └── models.py                 # Database models
├── processors/
│   └── processor.py              # Document processing engine
├── frontend/                     # React frontend (optional)
├── exceptions.py                 # Custom exceptions
├── logging_config.py             # Logging configuration
└── middleware.py                 # FastAPI middleware
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
**AI/ML:** Hugging Face Transformers, spaCy, Whisper, Apache Tika  
**Frontend:** React (optional)  
**Infrastructure:** Docker, Professional deployment automation

## Support and Contributing

- **Issues:** [GitHub Issues](https://github.com/imaddde867/explainium-2.0/issues)
- **Discussions:** [GitHub Discussions](https://github.com/imaddde867/explainium-2.0/discussions)
- **Contributing:** See main README for contribution guidelines

## Version Information

**Current Version:** 2.0 - Clean Architecture (Cleaner v1)  
**Architecture:** Consolidated and optimized codebase  
**Dependencies:** Reduced from 44+ to 20 core packages  
**Code Reduction:** 50% reduction in complexity  
**Documentation Status:** Updated for clean architecture