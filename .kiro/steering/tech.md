# Technology Stack & Build System

## Backend Stack

- **Framework**: FastAPI (Python 3.10+)
- **Task Queue**: Celery with Redis broker
- **Database**: PostgreSQL 13+ with SQLAlchemy ORM
- **Search**: Elasticsearch 8.14+
- **Document Processing**: Apache Tika server
- **AI/ML Libraries**:
  - Hugging Face Transformers (NER, classification)
  - OpenAI Whisper (speech-to-text)
  - KeyBERT (keyphrase extraction)
- **Media Processing**: FFmpeg, OpenCV, PyMuPDF

## Frontend Stack

- **Framework**: React 19.1+
- **Visualization**: D3.js 7.9+
- **Build Tool**: Create React App (react-scripts 5.0.1)
- **Testing**: Jest, React Testing Library

## Infrastructure

- **Containerization**: Docker + Docker Compose
- **Services**: Multi-container setup with service dependencies
- **Development**: Hot reload enabled for both backend and frontend

## Common Commands

### Development Setup
```bash
# Start all services
docker-compose up --build -d

# Start frontend development server
cd src/frontend && npm install && npm start

# Run backend tests
docker-compose exec app pytest

# View logs
docker-compose logs -f [service_name]
```

### Database Operations
```bash
# Access PostgreSQL
docker-compose exec db psql -U user -d knowledge_db

# Check Elasticsearch
curl http://localhost:9200/_cluster/health
```

### Celery Management
```bash
# Monitor Celery workers
docker-compose exec celery_worker celery -A src.api.celery_worker inspect active

# Purge task queue
docker-compose exec celery_worker celery -A src.api.celery_worker purge
```

## Environment Configuration

- **PYTHONPATH**: Set to `/app` in containers
- **Database**: `knowledge_db` with user/password credentials
- **Ports**: 8000 (API), 3000 (React), 5432 (PostgreSQL), 9200 (Elasticsearch), 6379 (Redis), 9998 (Tika)
- **CORS**: Configured for localhost development