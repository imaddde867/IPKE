# Industrial Knowledge Extraction System

## Overview

This project develops a comprehensive AI-powered system designed to extract structured knowledge from diverse industrial documentation, including documents, images, and videos. Built entirely with free and open-source technologies, it transforms unstructured data into actionable insights, stored in a searchable database.

## Key Features

-   **Multi-Modal Ingestion:** Processes PDF, DOCX, PPT, images (with OCR), and videos (with Speech-to-Text).
-   **AI-Powered Understanding:** Leverages Named Entity Recognition (NER), zero-shot text classification, and keyphrase extraction.
-   **Structured Data Extraction:** Identifies and extracts specific industrial entities such as Equipment, Procedures, Safety Information, Technical Specifications, and Personnel.
-   **Robust Data Storage:** Utilizes PostgreSQL for structured data persistence and Elasticsearch for efficient full-text and semantic search.
-   **Asynchronous Processing:** Employs Celery and Redis for scalable, non-blocking background processing of large files.
-   **RESTful API:** Provides a FastAPI backend for seamless data interaction and integration.
-   **Basic Web Interface:** A React frontend for intuitive document management, viewing, and search capabilities.

## Technology Stack

-   **Backend:** Python (FastAPI, Celery, Redis)
-   **AI/ML:** Hugging Face Transformers (NER, Classification), OpenAI Whisper (Speech-to-Text), KeyBERT (Keyphrase Extraction)
-   **Document Processing:** Apache Tika, FFmpeg, OpenCV
-   **Database:** PostgreSQL
-   **Search:** Elasticsearch
-   **Containerization:** Docker, Docker Compose
-   **Frontend:** React, D3.js

## Project Structure

```
project/
├── src/                 # Core application source code
│   ├── processors/      # Document parsing and content extraction
│   ├── ai/              # AI models and logic
│   ├── database/        # Database models and CRUD operations
│   ├── api/             # FastAPI endpoints and Celery tasks
│   ├── search/          # Elasticsearch integration
│   └── frontend/        # React web interface
├── tests/               # Unit and integration tests
├── docs/                # Comprehensive project documentation
└── docker/              # Docker configurations
```

## Setup and Running

### Quick Start (Local Development)

1.  **Clone the repository:**
    ```bash
    git clone 
    cd explainium-2.0
    ```

2.  **Build and start Docker services:**
    Ensure Docker Desktop is running, then execute:
    ```bash
    docker-compose up --build -d
    ```
    This will build all necessary Docker images and start the backend services (FastAPI, Celery, PostgreSQL, Elasticsearch, Tika, Redis).

3.  **Start the React frontend:**
    Navigate to the frontend directory and start the development server:
    ```bash
    cd src/frontend
    npm install # Install frontend dependencies (only needed once)
    npm start
    ```

    The frontend application will typically open in your browser at `http://localhost:3000`.

For more detailed setup, deployment, and troubleshooting information, please refer to: 
[docs/deployment.md](docs/deployment.md)

## API Documentation

Explore the available API endpoints and their functionalities: 
[docs/api.md](docs/api.md)

## Database Schema

Understand the underlying database structure and relationships: 
[docs/database.md](docs/database.md)

## Document Processing Pipeline

Learn about the multi-modal processing workflow: 
[docs/processing.md](docs/processing.md)

## Testing

To run the backend tests, ensure your Docker containers are running and execute:

```bash
docker-compose exec app pytest
```

For more details on testing, refer to the `tests/` directory.

## License

This project is open-source and available under the [MIT License](LICENSE). (Note: A `LICENSE` file would need to be created if not already present.)
