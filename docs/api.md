# API Documentation

## Base URL
`http://localhost:8000` (local development)

## Document Processing

### Upload Document
```http
POST /documents/upload
Content-Type: multipart/form-data

Body: file (PDF, DOCX, PPT, image, video)
```

**Response:**
```json
{
  "info": "file 'example.pdf' saved. Task ID: abc123"
}
```

### Get Documents
```http
GET /documents/
GET /documents/{document_id}
```

**Query Parameters:**
- `skip` (int): Pagination offset (default: 0)
- `limit` (int): Max results (default: 100)

## Knowledge Extraction

### Extract Knowledge
```http
POST /knowledge/extract
Body: {"document_id": 1}
```

### Get Knowledge Items
```http
GET /knowledge/items
GET /knowledge/items?domain=safety&confidence_min=0.8
```

### Get Relationships
```http
GET /knowledge/relationships
GET /knowledge/relationships?type=process_dependency
```

## Knowledge Graph

### Build Graph
```http
GET /graph/build
Body: {"knowledge_items": [...], "relationships": [...]}
```

### Query Graph
```http
POST /graph/query
Body: {
  "type": "find_paths",
  "source": "PROC_001",
  "target": "PROC_002"
}
```

### Get Visualization Data
```http
GET /graph/visualization
GET /graph/visualization?layout=hierarchical&filter=process
```

## Extracted Data Endpoints

### Get Extracted Entities
```http
GET /documents/{document_id}/entities/
```

### Get Equipment Data
```http
GET /documents/{document_id}/equipment/
```

### Get Procedures
```http
GET /documents/{document_id}/procedures/
```

### Get Safety Information
```http
GET /documents/{document_id}/safety_info/
```

### Get Personnel Data
```http
GET /documents/{document_id}/personnel/
```

## Search

### Search Documents
```http
GET /search/?query=safety&field=extracted_text&size=10
```

**Fields:** `extracted_text`, `filename`, `classification_category`, `extracted_entities.text`, `key_phrases`, `document_sections`

## System Health

### Health Check
```http
GET /health
GET /health/detailed
```

## Response Formats

All endpoints return JSON with standard HTTP status codes:
- `200`: Success
- `404`: Resource not found
- `422`: Validation error
- `500`: Server error

## Authentication

Currently no authentication required for local development. Production deployments should implement appropriate security measures.