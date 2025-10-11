# Explainium 2.0 - Development Notes

## What was cleaned up

This repository has been cleaned and optimized for research and experimentation:

### Removed Components
- **PUHTI deployment guides** (`PUHTI_DEPLOYMENT_GUIDE.md`, `PUHTI_PERFORMANCE_ANALYSIS.md`) - CSC supercomputing specific documentation
- **Kubernetes deployment** (`k8s/` directory) - Production deployment configs  
- **Puhti deployment script** (`deploy-puhti.sh`) - HPC-specific deployment
- **Bulk sample documents** - Reduced from 20+ files to 3 representative examples
- **Empty modules** - Removed unused `src/ai/extraction/` directory
- **Cache files** - Cleaned up `__pycache__` directories and old logs

### Simplified Structure
The codebase now focuses on core functionality:

```
src/
├── ai/                    # Core AI engines
├── api/                   # FastAPI backend  
├── frontend/              # Streamlit interface
├── processors/            # Document processing
├── database/              # Data persistence
├── core/                  # Configuration
└── export/                # Export utilities
```

### For Research & Development

**Quick start:**
```bash
python setup.py              # One-time setup
streamlit run src/frontend/knowledge_table.py  # Start frontend
```

**Key features for experimentation:**
- Local AI model execution (no external APIs)
- Multi-format document processing (PDF, DOCX, images, audio, video)
- Configurable AI engines with fallback hierarchy
- Interactive Streamlit dashboard for results review
- REST API for programmatic access

**Model management:**
```bash
python scripts/model_manager.py --action setup    # Download models
python scripts/model_manager.py --action list     # List available models
```

**Sample documents:**
- `documents_samples/iso-9001-2015-guidance-document-eng.pdf` - ISO standard
- `documents_samples/osha3132.pdf` - OSHA safety document  
- `documents_samples/safe-use-of-MEWP.png` - Safety image with text

This streamlined version is ideal for:
- Research into knowledge extraction techniques
- Experimenting with different AI models
- Developing new processing pipelines
- Testing document intelligence approaches