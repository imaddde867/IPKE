# Document Processing Pipeline

This document details the multi-modal document processing pipeline within the Industrial Knowledge Extraction System. The pipeline is designed to ingest various document types, extract raw content, and enrich it with AI-powered insights.

## Data Flow Architecture

```
Input Documents → Multi-Modal Processing → AI Understanding → 
Classification → Metadata Extraction → Database Storage → 
Search Indexing → Web Interface → User Access
```

## Components and Technologies

The processing pipeline is orchestrated by Celery tasks and leverages several open-source tools for different stages:

### 1. File Ingestion and Type Detection

- **Location:** `src/api/main.py` (upload endpoint), `src/processors/document_processor.py` (`get_file_type`)
- **Description:** Documents are uploaded via the FastAPI endpoint and saved to a local `uploaded_files` directory. The `get_file_type` function performs basic file type detection based on file extensions (e.g., PDF, DOCX, images, videos).

### 2. Multi-Modal Content Extraction

- **Location:** `src/processors/document_processor.py` (`process_document`, `process_video`)
- **Technologies:**
    - **Apache Tika:** Used for extracting text content and metadata from various document formats (PDF, DOCX, PPT, etc.) and images (with OCR capabilities).
        - Tika is run as a separate Docker service (`apache/tika:latest`).
        - The system interacts with Tika's `/rmeta` endpoint to get rich JSON output.
        - OCR is enabled for scanned documents and images.
    - **FFmpeg:** Used for extracting audio streams from video files.
    - **OpenAI Whisper:** Used for performing Speech-to-Text (STT) transcription on extracted audio from videos.
- **Output:** Raw extracted text content and comprehensive metadata.

### 3. AI-Powered Content Understanding

- **Location:** `src/ai/ner_extractor.py`, `src/ai/classifier.py`, `src/ai/keyphrase_extractor.py`, `src/processors/document_processor.py`
- **Technologies:** Hugging Face `transformers` library.
    - **Named Entity Recognition (NER):**
        - **Module:** `src/ai/ner_extractor.py`
        - **Model:** `dslim/bert-base-NER` (pre-trained BERT model for NER).
        - **Function:** Identifies and extracts entities like persons, organizations, locations, etc., from the extracted text.
    - **Content Classification:**
        - **Module:** `src/ai/classifier.py`
        - **Model:** `facebook/bart-large-mnli` (zero-shot classification model).
        - **Function:** Categorizes documents into predefined industrial categories (e.g., Operational Procedures, Safety Documentation) with a confidence score.
    - **Keyphrase Extraction:**
        - **Module:** `src/ai/keyphrase_extractor.py`
        - **Library:** `KeyBERT` (uses Sentence-BERT internally).
        - **Function:** Extracts relevant keywords and key phrases from the document content.

### 4. Advanced Metadata and Structured Data Extraction

- **Location:** `src/processors/document_processor.py` (`_extract_equipment_data`, `_extract_procedure_data`, `_extract_safety_information`, `_extract_technical_specifications`, `_extract_personnel_data`, `extract_sections`)
- **Description:** This stage focuses on transforming unstructured text into structured, actionable data based on predefined categories. It uses a combination of:
    - **Regular Expressions (Regex):** For pattern matching specific data points (e.g., measurements, dates, specific phrases).
    - **Keyword Matching:** Identifying relevant sections or data based on predefined lists of keywords.
    - **Contextual Analysis:** Looking for information in the vicinity of identified entities or keywords.
    - **Section Extraction:** Identifying logical sections within the document to provide context for extracted data.
- **Output:** Populated lists of structured data for Equipment, Procedures, Safety Information, Technical Specifications, and Personnel, each with an associated confidence score.

### 5. Database Storage

- **Location:** `src/database/models.py`, `src/database/crud.py`, `src/api/celery_worker.py`
- **Technology:** PostgreSQL, SQLAlchemy ORM.
- **Function:** All extracted raw content, metadata, AI insights (NER, classification, keyphrases), and structured data are persisted into a normalized PostgreSQL database schema.

### 6. Search Indexing

- **Location:** `src/search/elasticsearch_client.py`, `src/api/celery_worker.py`
- **Technology:** Elasticsearch.
- **Function:** After successful database storage, relevant document information (extracted text, classification, entities, keyphrases, sections) is indexed into Elasticsearch for fast full-text and faceted search capabilities.

## Asynchronous Processing

- **Technology:** Celery, Redis.
- **Function:** All heavy-lifting tasks, such as content extraction, AI processing, and database/Elasticsearch indexing, are performed asynchronously by Celery workers. This ensures the FastAPI application remains responsive and can handle multiple concurrent file uploads without blocking.
