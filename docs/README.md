# Industrial Knowledge Extraction System

## Overview

This project aims to build a comprehensive AI-powered knowledge extraction system that processes industrial documentation (documents, images, videos) and extracts business processes into a structured database. The system is built entirely using free and open-source tools.

## Core Features

- **Multi-Modal Input Processing:** Handles PDF, DOCX, PPT, images (with OCR), and videos (with speech-to-text).
- **AI-Powered Content Understanding:** Utilizes Named Entity Recognition (NER), zero-shot text classification, and keyphrase extraction.
- **Structured Data Extraction:** Extracts specific entities like Equipment, Procedures, Safety Information, Technical Specifications, and Personnel into a structured format.
- **PostgreSQL Database:** Stores all extracted data in a normalized relational database.
- **Elasticsearch Integration:** Provides full-text and semantic search capabilities.
- **REST API:** FastAPI backend for document upload, data retrieval, and search.
- **Web Interface (Basic):** A React frontend for basic interaction, document viewing, and search.

## Technology Stack

- **Backend:** Python (FastAPI, Celery, Redis)
- **AI/ML:** Hugging Face Transformers (for NER and Classification), OpenAI Whisper (for Speech-to-Text), KeyBERT (for Keyphrase Extraction)
- **Document Processing:** Apache Tika, FFmpeg, OpenCV
- **Database:** PostgreSQL
- **Search:** Elasticsearch
- **Containerization:** Docker, Docker Compose
- **Frontend:** React, D3.js

## Project Structure

```
project/
├── src/
│   ├── processors/        # Document processing modules (Tika, FFmpeg, Whisper integration, structured extraction)
│   ├── ai/                # AI models and extraction logic (NER, Classification, Keyphrase)
│   ├── database/          # Database models (SQLAlchemy) and operations (CRUD)
│   ├── api/               # FastAPI endpoints and Celery tasks
│   ├── search/            # Elasticsearch integration
│   └── frontend/          # React web interface
├── tests/                 # Comprehensive test suite
├── docs/                  # Project documentation
├── docker/                # Dockerfiles and Docker Compose configurations
└── scripts/               # Deployment and maintenance scripts (future)
```

## Setup and Running

See `docs/deployment.md` for detailed instructions.

## API Documentation

See `docs/api.md` for detailed API endpoint documentation.

## Database Schema

See `docs/database.md` for database schema and relationships.

## Document Processing

See `docs/processing.md` for details on the document processing pipeline.
