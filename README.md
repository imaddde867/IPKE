# 🧠 EXPLAINIUM: Enterprise Knowledge Extraction System

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/imaddde867/explainium-2.0)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)

EXPLAINIUM is an AI-powered system that extracts, analyzes, and structures knowledge from enterprise documents. Transform unstructured organizational knowledge into actionable insights with real-time processing and professional web interface.

## ✨ Key Features

- 🚀 **Fast Processing**: Documents processed in ~3 seconds
- 📄 **Multi-format Support**: PDF, DOCX, TXT, videos (MP4, AVI, MOV), images with OCR
- 🎥 **Video Processing**: Audio extraction with FFmpeg + AI transcription using OpenAI Whisper
- 🔍 **Advanced Knowledge Extraction**: Equipment specs, procedures, safety info, technical data, personnel details
- 🧠 **AI-Powered Analysis**: Named Entity Recognition, document classification, keyphrase extraction
- 📊 **Real-time Interface**: Drag-and-drop upload with live progress tracking
- 🛡️ **Robust Architecture**: Graceful fallbacks, error handling, and failsafe processing
- 🐳 **Docker Ready**: One-command deployment with full containerization

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

## ⚡ Quick Commands (Make)

For convenience, use these Makefile commands to manage the application:

```bash
# Quick start/stop
make start    # Start all services quickly
make stop     # Stop all services
make dev      # Start development environment

# Individual services
make backend  # Start backend services only
make frontend # Start frontend only  
make services # Start supporting services only (DB, Redis, etc.)

# Utilities
make logs     # View all service logs
make clean    # Clean up containers and volumes
make install  # Install frontend dependencies
make help     # Show all available commands
```

### Alternative Manual Commands

You can also use Docker Compose directly:

```bash
# Start all services
docker-compose up --build -d

# Stop all services  
docker-compose down

# View logs
docker-compose logs -f
```

## 📋 Usage

1. **Upload Documents**: Drag and drop files or click to select
2. **Real-time Processing**: Watch progress as documents are processed
3. **View Results**: Extracted knowledge displayed with confidence scores
4. **Explore Data**: Browse equipment, procedures, safety information, and more

### Supported File Types
- **Documents**: PDF, Microsoft Word (.docx), Text files (.txt), PowerPoint (.pptx), Excel (.xlsx)
- **Videos**: MP4, AVI, MOV, MKV (with audio transcription using Whisper AI)
- **Images**: JPG, PNG, GIF, BMP, TIFF (with OCR text extraction)
- **Audio**: Extracted from videos and transcribed to text

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
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   FFmpeg +      │
                       │   (Queue)       │    │   Whisper AI    │
                       └─────────────────┘    │ (Video/Audio)   │
                                │              └─────────────────┘
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Elasticsearch   │    │   AI Models     │
                       │   (Search)      │    │  (NER, Classify)│
                       └─────────────────┘    └─────────────────┘
```

## 📊 Extracted Knowledge Types

### Equipment Information
- Motor specifications (HP, voltage, RPM)
- Pump details (flow rate, pressure)
- Sensor data and calibration
- Maintenance schedules
- Enhanced pattern recognition with confidence scoring

### Safety Documentation
- Hazard identification with severity levels
- PPE requirements and compliance
- Emergency procedures
- Safety standards (OSHA, ISO)
- Risk assessment data

### Technical Specifications
- Operating parameters with tolerances
- Performance metrics and measurements
- Quality standards and compliance
- Calibration requirements

### Personnel Information
- Roles and responsibilities
- Certifications and training records
- Contact information
- Skill assessments

### Video Content Analysis
- Training video transcription
- Safety demonstration procedures  
- Equipment operation instructions
- Audio-extracted knowledge from multimedia content

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

## � Enhanced Processing Pipeline

### Multi-Modal Content Extraction
- **Documents**: Apache Tika with fallback to PyPDF2 for reliability
- **Videos**: FFmpeg audio extraction + OpenAI Whisper transcription
- **Images**: OCR text extraction for visual content
- **Fallback Systems**: Graceful degradation when services are unavailable

### AI-Powered Analysis
- **Named Entity Recognition**: Person, organization, and equipment identification
- **Document Classification**: Automatic categorization (Operational, Safety, Training, etc.)
- **Keyphrase Extraction**: Important terms and concepts identification
- **Confidence Scoring**: Multi-factor reliability assessment

### Structured Knowledge Extraction
- **Equipment Data**: Specifications, types, and technical parameters
- **Procedure Data**: Step-by-step processes with categorization
- **Safety Information**: Hazard levels, PPE requirements, emergency procedures
- **Technical Specifications**: Measurements, tolerances, and standards
- **Personnel Data**: Roles, certifications, and contact details

### Error Handling & Reliability
- Graceful fallback mechanisms for all processing stages
- Comprehensive logging and error reporting
- Service availability detection and adaptation
- Automatic cleanup of temporary files

## �📈 Performance

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

**Video processing issues:**
```bash
# Check FFmpeg availability
docker-compose exec app ffmpeg -version

# Check Whisper model loading
docker-compose logs celery_worker | grep "whisper"

# Verify video file format support
docker-compose exec app ffprobe <video_file>
```

**AI model failures:**
```bash
# Check model loading and availability
docker-compose logs celery_worker | grep -i "model\|ai\|ner"

# Restart with fallback processing
docker-compose restart celery_worker
```

### Performance Optimization

- Increase worker concurrency: `CELERY_CONCURRENCY=4`
- Adjust database pool: `DB_POOL_SIZE=20`
- Optimize file size limits: `MAX_FILE_SIZE_MB=50`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.