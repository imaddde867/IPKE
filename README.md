# EXPLAINIUM: Enterprise Knowledge Extraction System

## Overview

EXPLAINIUM is an AI-powered system that extracts, analyzes, and structures tacit knowledge from enterprise documents. It transforms unstructured organizational knowledge into a comprehensive, queryable database with interactive knowledge graphs and relationship mapping.

**Key Capabilities:**
- Multi-modal content processing (PDFs, Office docs, videos, diagrams)
- Tacit knowledge extraction with NLP and pattern recognition
- Process dependency mapping and relationship analysis
- Interactive knowledge graphs with advanced analytics
- Enterprise-grade database with confidence scoring

## Technology Stack

**Backend:** Python, FastAPI, PostgreSQL, Redis, Celery  
**AI/ML:** Hugging Face Transformers, spaCy, NetworkX, Apache Tika  
**Frontend:** React, D3.js for visualization  
**Infrastructure:** Docker, Elasticsearch

## Project Structure

```
explainium/
├── src/
│   ├── ai/                        # AI and ML components
│   │   ├── knowledge_extraction_engine.py  # Tacit knowledge extraction
│   │   ├── relationship_mapper.py          # Process dependency mapping
│   │   └── knowledge_graph.py              # Graph generation and analysis
│   ├── database/                  # Database models and operations
│   │   ├── models.py              # Data models with relationships
│   │   └── connection.py          # Database connection management
│   ├── api/                       # FastAPI endpoints
│   │   ├── main.py                # Main application
│   │   ├── document_routes.py     # Document processing
│   │   └── knowledge_routes.py    # Knowledge extraction
│   └── processors/                # Content processing pipeline
├── tests/                         # Comprehensive test suite (54+ tests)
└── docs/                          # Project documentation
```

## Quick Start

```bash
# Clone and start
git clone https://github.com/imaddde867/explainium-2.0
cd explainium-2.0
docker-compose up --build -d

# Initialize database
docker-compose exec app alembic upgrade head

# Start frontend (optional)
cd src/frontend && npm install && npm start
```

Access the API at `http://localhost:8000` and frontend at `http://localhost:3000`.

## Core Features

### 1- Knowledge Extraction
- **Tacit Knowledge Detection:** Identifies implicit decision patterns and optimization opportunities
- **Workflow Dependencies:** Maps process relationships with confidence scoring
- **Decision Trees:** Extracts conditional logic and decision-making patterns
- **Resource Optimization:** Detects efficiency opportunities across multiple dimensions

### 2- Relationship Mapping
- **Process Dependencies:** Prerequisite, parallel, downstream, and conditional relationships
- **Equipment-Maintenance:** Links equipment to maintenance patterns (preventive, corrective, predictive)
- **Skill-Function:** Maps personnel skills to job functions with proficiency assessment
- **Compliance-Procedure:** Connects regulatory requirements (OSHA, EPA, ISO, FDA) to procedures

### 3- Knowledge Graph Analytics
- **Graph Construction:** Multi-modal graphs from extracted relationships
- **Critical Path Analysis:** Bottleneck detection and circular dependency resolution
- **Interactive Visualization:** Multiple layout algorithms with filtering capabilities
- **Complex Queries:** Path finding, neighbor analysis, and subgraph examination

## API Endpoints

**Documents:** `POST /documents/upload`, `GET /documents/{id}`  
**Knowledge:** `POST /knowledge/extract`, `GET /knowledge/items`, `GET /knowledge/relationships`  
**Graph:** `GET /graph/build`, `POST /graph/query`, `GET /graph/visualization`  
**Health:** `GET /health`, `GET /health/detailed`

## Testing

```bash
# Run all tests (54+ test cases, 95%+ coverage)
pytest

# Run specific modules
pytest tests/test_knowledge_extraction.py -v
pytest tests/test_relationship_mapping.py -v
pytest tests/test_knowledge_graph.py -v
```

**Test Coverage:** Knowledge extraction (19), Relationship mapping (19), Knowledge graph (35), Integration tests

## Recent Updates

**v2.0 - Relationship Mapping System:**
- Process dependency mapping with confidence scoring
- Equipment-maintenance correlation analysis
- Skill-function linking with proficiency assessment
- Compliance-procedure connections for regulatory requirements
- Interactive knowledge graph generation and analytics
- 54+ test cases with 95%+ code coverage

**Roadmap:**
- v2.1: Predictive gap analysis, automated updates, advanced search
- v3.0: AI agent integration, autonomous discovery, cross-enterprise sharing

## Contributing

**Areas for Contribution:**
- Knowledge extraction algorithm improvements
- Relationship mapping accuracy enhancements
- Graph analytics and visualization techniques
- Performance optimization for large-scale deployments
- Enterprise system integrations (SAP, Oracle, etc.)

## License

This project is open-source and available under the [MIT License](LICENSE).

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/imaddde867/explainium-2.0/issues)
- **Discussions:** [GitHub Discussions](https://github.com/imaddde867/explainium-2.0/discussions)

---

**EXPLAINIUM: Transforming Enterprise Knowledge into Intelligent Action**
