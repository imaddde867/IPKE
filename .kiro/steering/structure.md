# Project Structure & Organization

## Directory Layout

```
project/
├── src/                     # Core application source code
│   ├── ai/                  # AI/ML processing modules
│   │   ├── classifier.py    # Document classification
│   │   ├── ner_extractor.py # Named entity recognition
│   │   └── keyphrase_extractor.py # Keyphrase extraction
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # API endpoints and routes
│   │   └── celery_worker.py # Background task definitions
│   ├── database/            # Data layer
│   │   ├── models.py        # SQLAlchemy ORM models
│   │   ├── database.py      # Database connection/session
│   │   └── crud.py          # Database operations
│   ├── processors/          # Document processing pipeline
│   │   └── document_processor.py # Multi-modal content extraction
│   ├── search/              # Search functionality
│   │   └── elasticsearch_client.py # Elasticsearch integration
│   └── frontend/            # React web application
│       ├── src/             # React source code
│       ├── public/          # Static assets
│       └── package.json     # Frontend dependencies
├── tests/                   # Test suites
├── docs/                    # Project documentation
├── docker/                  # Docker configurations
├── uploaded_files/          # File storage directory
└── scripts/                 # Utility scripts
```

## Code Organization Patterns

### Database Models
- **Base Model**: All models inherit from SQLAlchemy `Base`
- **Relationships**: Proper foreign key relationships with back_populates
- **JSON Fields**: Used for flexible metadata and structured data storage
- **Confidence Scores**: Added to extracted entities for quality assessment

### API Structure
- **RESTful Design**: Standard HTTP methods and status codes
- **Dependency Injection**: Database sessions via FastAPI Depends
- **CORS Configuration**: Explicit origins for frontend integration
- **Error Handling**: Consistent HTTPException usage

### Processing Pipeline
- **Modular Design**: Separate processors for different content types
- **Structured Extraction**: Dedicated functions for each data type (equipment, procedures, etc.)
- **Confidence Scoring**: All extracted data includes confidence metrics
- **Section Parsing**: Documents automatically segmented into logical sections

### AI Integration
- **Model Abstraction**: Each AI capability in separate modules
- **Configurable Labels**: Classification categories defined as constants
- **Entity Mapping**: NER results mapped to domain-specific entities
- **Batch Processing**: Optimized for handling multiple documents

## File Naming Conventions

- **Python Files**: snake_case (e.g., `document_processor.py`)
- **Classes**: PascalCase (e.g., `DocumentProcessor`)
- **Functions/Variables**: snake_case (e.g., `extract_entities`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CANDIDATE_LABELS`)
- **React Components**: PascalCase (e.g., `App.js`)

## Import Organization

- **Standard Library**: First
- **Third-party**: Second  
- **Local Imports**: Last, using relative imports within src/

## Configuration Management

- **Environment Variables**: Used for service connections
- **Docker Compose**: Service configuration and dependencies
- **Package Files**: requirements.txt (Python), package.json (React)
- **Database**: Connection strings and credentials in compose file