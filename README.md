# 🧠 EXPLAINIUM: Enterprise Knowledge Extraction System

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/imaddde867/explainium-2.0)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)

EXPLAINIUM is an AI-powered system that extracts, analyzes, and structures knowledge from enterprise documents. Transform unstructured organizational knowledge into actionable insights with real-time processing and professional web interface.

## ✨ Key Features

- 🚀 **Fast Processing**: Documents processed in ~3 seconds
- 📄 **Multi-format Support**: PDF, DOCX, TXT, and more
- 🔍 **Knowledge Extraction**: Equipment, procedures, safety info, technical specs
- 📊 **Real-time Interface**: Drag-and-drop upload with live progress
- 🛡️ **Robust Architecture**: Graceful fallbacks and error handling
- 🐳 **Docker Ready**: One-command deployment

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/imaddde867/explainium-2.0
cd explainium-2.0

# Start all services
docker-compose up --build -d

# Initialize database (first time only)
docker-compose exec app alembic upgrade head
```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 Usage

1. **Upload Documents**: Drag and drop files or click to select
2. **Real-time Processing**: Watch progress as documents are processed
3. **View Results**: Extracted knowledge displayed with confidence scores
4. **Explore Data**: Browse equipment, procedures, safety information, and more

### Supported File Types
- PDF documents
- Microsoft Word (.docx)
- Text files (.txt)
- PowerPoint (.pptx)
- Excel (.xlsx)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI App   │    │   PostgreSQL    │
│   (HTML/JS)     │◄──►│   (Python)      │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Celery Worker  │    │   Apache Tika   │
                       │  (Processing)   │◄──►│  (Text Extract) │
                       └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │ Elasticsearch   │
                       │   (Queue)       │    │   (Search)      │
                       └─────────────────┘    └─────────────────┘
```

## 📊 Extracted Knowledge Types

### Equipment Information
- Motor specifications (HP, voltage, RPM)
- Pump details (flow rate, pressure)
- Sensor data and calibration
- Maintenance schedules

### Safety Documentation
- Hazard identification
- PPE requirements
- Emergency procedures
- Compliance standards (OSHA, ISO)

### Technical Specifications
- Operating parameters
- Tolerance ranges
- Performance metrics
- Quality standards

### Personnel Information
- Roles and responsibilities
- Certifications
- Training requirements
- Contact information

## 🔧 Configuration

### Environment Variables

Key configuration options in `docker-compose.yml`:

```yaml
# Database
DB_HOST=db
DB_NAME=knowledge_db
DB_USER=user
DB_PASSWORD=password

# Processing
MAX_FILE_SIZE_MB=100
TIKA_TIMEOUT_SECONDS=300

# API
API_PORT=8000
CORS_ORIGINS=http://localhost,http://localhost:3000
```

### Custom Configuration

Create a `.env` file to override defaults:

```bash
# Custom settings
MAX_FILE_SIZE_MB=200
API_PORT=8080
DEBUG=false
```

## 🛠️ Development

### Local Development Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start services individually
docker-compose up db redis elasticsearch tika -d

# Run the application
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker
celery -A src.api.celery_worker worker --loglevel=info
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_document_processor.py -v
```

## 📈 Performance

- **Processing Speed**: ~3 seconds per document
- **Throughput**: 20+ documents/minute
- **Memory Usage**: ~2GB for full stack
- **Storage**: ~100MB per 1000 documents

## 🔍 API Endpoints

### Document Management
- `POST /uploadfile/` - Upload documents
- `GET /documents/` - List all documents
- `GET /documents/{id}` - Get specific document

### Knowledge Extraction
- `GET /documents/{id}/equipment/` - Equipment data
- `GET /documents/{id}/procedures/` - Procedures
- `GET /documents/{id}/safety_info/` - Safety information
- `GET /documents/{id}/technical_specs/` - Technical specs

### System Health
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed service status

## 🐛 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs app
docker-compose logs celery_worker
```

**Database connection issues:**
```bash
# Restart database
docker-compose restart db

# Check database logs
docker-compose logs db
```

**Processing stuck:**
```bash
# Restart worker
docker-compose restart celery_worker

# Check worker logs
docker-compose logs celery_worker --tail=50
```

### Performance Optimization

- Increase worker concurrency: `CELERY_CONCURRENCY=4`
- Adjust database pool: `DB_POOL_SIZE=20`
- Optimize file size limits: `MAX_FILE_SIZE_MB=50`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.