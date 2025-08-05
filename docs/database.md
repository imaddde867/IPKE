# Database Schema

## Overview

PostgreSQL database with normalized schema supporting enterprise knowledge extraction, relationship mapping, and graph analytics.

## Core Tables

### Documents
```sql
documents (
  id SERIAL PRIMARY KEY,
  filename VARCHAR INDEX,
  file_type VARCHAR,
  extracted_text TEXT,
  metadata_json JSON,
  classification_category VARCHAR,
  classification_score FLOAT,
  status VARCHAR INDEX,
  processing_timestamp TIMESTAMP DEFAULT NOW(),
  document_sections JSON
)
```

### Knowledge Items
```sql
knowledge_items (
  id SERIAL PRIMARY KEY,
  process_id VARCHAR UNIQUE INDEX,
  name VARCHAR INDEX,
  description TEXT,
  knowledge_type VARCHAR INDEX,
  domain VARCHAR INDEX,
  hierarchy_level INTEGER INDEX,
  confidence_score FLOAT,
  completeness_index FLOAT,
  criticality_level VARCHAR,
  source_document_id INTEGER REFERENCES documents(id)
)
```

### Workflow Dependencies
```sql
workflow_dependencies (
  id SERIAL PRIMARY KEY,
  source_process_id VARCHAR REFERENCES knowledge_items(process_id),
  target_process_id VARCHAR REFERENCES knowledge_items(process_id),
  dependency_type VARCHAR INDEX,
  strength FLOAT,
  conditions JSON,
  confidence FLOAT
)
```

## Extracted Data Tables

### Equipment
```sql
equipment (
  id SERIAL PRIMARY KEY,
  document_id INTEGER REFERENCES documents(id),
  name VARCHAR INDEX,
  type VARCHAR,
  specifications JSON,
  location VARCHAR,
  confidence FLOAT
)
```

### Procedures
```sql
procedures (
  id SERIAL PRIMARY KEY,
  document_id INTEGER REFERENCES documents(id),
  title VARCHAR INDEX,
  category VARCHAR,
  confidence FLOAT
)

steps (
  id SERIAL PRIMARY KEY,
  procedure_id INTEGER REFERENCES procedures(id),
  step_number INTEGER,
  description TEXT,
  expected_result TEXT,
  confidence FLOAT
)
```

### Personnel
```sql
personnel (
  id SERIAL PRIMARY KEY,
  document_id INTEGER REFERENCES documents(id),
  name VARCHAR INDEX,
  role VARCHAR,
  responsibilities TEXT,
  certifications JSON,
  confidence FLOAT
)
```

## Relationship Tables

### Process Dependencies
```sql
procedure_equipment (
  id SERIAL PRIMARY KEY,
  procedure_id INTEGER REFERENCES procedures(id),
  equipment_id INTEGER REFERENCES equipment(id)
)

procedure_personnel (
  id SERIAL PRIMARY KEY,
  procedure_id INTEGER REFERENCES procedures(id),
  personnel_id INTEGER REFERENCES personnel(id)
)
```

## Advanced Knowledge Tables

### Decision Trees
```sql
decision_trees (
  id SERIAL PRIMARY KEY,
  process_id VARCHAR REFERENCES knowledge_items(process_id),
  decision_point VARCHAR,
  decision_type VARCHAR INDEX,
  conditions JSON,
  outcomes JSON,
  confidence FLOAT,
  priority VARCHAR
)
```

### Optimization Patterns
```sql
optimization_patterns (
  id SERIAL PRIMARY KEY,
  pattern_type VARCHAR INDEX,
  name VARCHAR,
  description TEXT,
  domain VARCHAR INDEX,
  conditions JSON,
  improvements JSON,
  success_metrics JSON,
  confidence FLOAT,
  impact_level VARCHAR
)
```

### Knowledge Gaps
```sql
knowledge_gaps (
  id SERIAL PRIMARY KEY,
  gap_type VARCHAR INDEX,
  title VARCHAR,
  description TEXT,
  domain VARCHAR INDEX,
  affected_processes JSON,
  impact_assessment JSON,
  priority VARCHAR INDEX,
  status VARCHAR INDEX,
  identified_at TIMESTAMP DEFAULT NOW()
)
```

## Indexes and Performance

**Key Indexes:**
- `documents(filename, status, processing_timestamp)`
- `knowledge_items(process_id, domain, hierarchy_level)`
- `workflow_dependencies(source_process_id, target_process_id, dependency_type)`
- `equipment(name, type)`, `procedures(title, category)`, `personnel(name, role)`

**Performance Features:**
- JSON columns for flexible metadata storage
- Confidence scoring on all extracted data
- Hierarchical process organization
- Full-text search integration with Elasticsearch

## Migration Management

Database migrations managed with Alembic:
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```