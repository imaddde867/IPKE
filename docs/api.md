# API Documentation

This document describes the RESTful API endpoints provided by the Industrial Knowledge Extraction System backend, built with FastAPI.

## Base URL

`http://localhost:8000` (when running locally via Docker Compose)

## Endpoints

### 1. Root Endpoint

`GET /`

Returns a welcome message.

**Response:**
```json
{
  "message": "Welcome to the Industrial Knowledge Extraction System!"
}
```

### 2. Upload Document

`POST /uploadfile/`

Uploads a document (PDF, DOCX, PPT, image, or video) for processing. The processing is asynchronous, handled by Celery.

**Request:**
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: The file to upload.

**Response:**
```json
{
  "info": "file '<filename>' saved at '<file_location>'. Task ID: <celery_task_id>"
}
```

### 3. Get All Documents

`GET /documents/`

Retrieves a list of all processed documents stored in the database.

**Query Parameters:**
- `skip` (integer, optional): Number of documents to skip (for pagination). Default is `0`.
- `limit` (integer, optional): Maximum number of documents to return. Default is `100`.

**Response:**
Returns a JSON array of `Document` objects.

### 4. Get Document by ID

`GET /documents/{document_id}`

Retrieves a single processed document by its ID.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a `Document` object.

**Error Responses:**
- `404 Not Found`: If the document with the given ID does not exist.

### 5. Get Extracted Entities for a Document

`GET /documents/{document_id}/entities/`

Retrieves all Named Entities extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `ExtractedEntity` objects.

**Error Responses:**
- `404 Not Found`: If no entities are found for the document.

### 6. Get Key Phrases for a Document

`GET /documents/{document_id}/keyphrases/`

Retrieves all key phrases extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `KeyPhrase` objects.

**Error Responses:**
- `404 Not Found`: If no key phrases are found for the document.

### 7. Get Equipment Data for a Document

`GET /documents/{document_id}/equipment/`

Retrieves structured equipment data extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `Equipment` objects.

**Error Responses:**
- `404 Not Found`: If no equipment data is found for the document.

### 8. Get Procedure Data for a Document

`GET /documents/{document_id}/procedures/`

Retrieves structured procedure data extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `Procedure` objects.

**Error Responses:**
- `404 Not Found`: If no procedure data is found for the document.

### 9. Get Safety Information for a Document

`GET /documents/{document_id}/safety_info/`

Retrieves structured safety information extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `SafetyInformation` objects.

**Error Responses:**
- `404 Not Found`: If no safety information is found for the document.

### 10. Get Technical Specifications for a Document

`GET /documents/{document_id}/technical_specs/`

Retrieves structured technical specifications extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `TechnicalSpecification` objects.

**Error Responses:**
- `404 Not Found`: If no technical specifications are found for the document.

### 11. Get Personnel Data for a Document

`GET /documents/{document_id}/personnel/`

Retrieves structured personnel data extracted from a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON array of `Personnel` objects.

**Error Responses:**
- `404 Not Found`: If no personnel data is found for the document.

### 12. Get Document Sections

`GET /documents/{document_id}/sections/`

Retrieves the extracted sections of a specific document.

**Path Parameters:**
- `document_id` (integer, required): The ID of the document.

**Response:**
Returns a JSON object where keys are section titles and values are section content.

**Error Responses:**
- `404 Not Found`: If the document or its sections are not found.

### 13. Search Documents

`GET /search/`

Performs a full-text search across processed documents using Elasticsearch.

**Query Parameters:**
- `query` (string, required): The search query string.
- `field` (string, optional): The field to search within. Defaults to `extracted_text`. 
  Allowed values: `extracted_text`, `filename`, `classification_category`, `extracted_entities.text`, `key_phrases`, `document_sections`.
- `size` (integer, optional): The maximum number of search results to return. Default is `10`.

**Response:**
Returns a JSON array of matching document objects from Elasticsearch.
