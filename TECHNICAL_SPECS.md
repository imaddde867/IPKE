# Explainium 2.0 – Technical Specifications

## Processing Architecture Overview

### **Core Processing Rules & Values**

| Rule Name | Condition | Action | Priority | Threshold | Status |
|-----------|-----------|--------|----------|-----------|--------|
| Primary Semantic Rule | `document_complexity >= 0.5 AND llm_available == True` | `use_llm_primary_processing` | Critical | 0.75 | Active |
| Quality Threshold Rule | `extraction_confidence < 0.70` | `escalate_to_higher_priority_method` | High | 0.70 | Active |
| Validation Gate Rule | `entity_count > 0` | `validate_all_entities` | Critical | 0.70 | Active |
| Structured Output Rule | `processing_complete == True` | `ensure_structured_categorized_output` | Critical | 0.80 | Active |
| Fallback Hierarchy Rule | `primary_method_failed == True` | `cascade_through_processing_hierarchy` | High | 0.60 | Active |

### **Processing Hierarchy & Confidence Targets**

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING HIERARCHY                     │
├─────────────────────────────────────────────────────────────┤
│ Primary Semantic Engine                                  │
│    ├── Confidence Range: 0.75 - 0.95                     │
│    ├── Primary LLM: Mistral-7B-Instruct-v0.2             │
│    ├── Multi-Prompt Strategy: 5 specialized prompts      │
│    └── Validation: Entity + Relationship extraction      │
├─────────────────────────────────────────────────────────────┤
│ Enhanced Pattern Recognition                              │
│    ├── Confidence Range: 0.60 - 0.80                     │
│    ├── 10+ Specialized extraction patterns               │
│    ├── NLP Enhancement: spaCy + embeddings               │
│    └── Context-aware tagging                             │
├─────────────────────────────────────────────────────────────┤
│ Legacy Pattern Matching                                   │
│    ├── Confidence Range: 0.50 - 0.65                     │
│    ├── Basic regex patterns                              │
│    ├── Emergency fallback only                           │
│    └── Minimal entity extraction                         │
└─────────────────────────────────────────────────────────────┘
```

### **Quality Thresholds & Validation Gates**

```python
class QualityThreshold(Enum):
    LLM_MINIMUM = 0.75          # Minimum LLM confidence
    ENHANCED_MINIMUM = 0.60     # Enhanced extraction minimum
    COMBINED_MINIMUM = 0.80     # Combined analysis threshold  
    ENTITY_VALIDATION = 0.70    # Entity validation score
    PRODUCTION_READY = 0.85     # Production deployment threshold
```

## Knowledge Categories & Entity Types

### **Primary Knowledge Categories**

| Category | Description | Confidence Target | LLM Enhanced |
|----------|-------------|-------------------|--------------|
| **Technical Specifications** | Equipment specs, parameters, measurements | 0.95 | ✅ Yes |
| **Risk Mitigation Intelligence** | Safety requirements, hazards, protective measures | 0.90 | ✅ Yes |
| **Process Intelligence** | Workflows, procedures, step-by-step processes | 0.85 | ✅ Yes |
| **Compliance Governance** | Regulations, standards, requirements | 0.80 | ✅ Yes |
| **Organizational Intelligence** | Roles, responsibilities, personnel data | 0.75 | ✅ Yes |
| **Knowledge Definitions** | Terms, definitions, explanations | 0.70 | ✅ Yes |

### **Entity Type Mapping**

```python
EntityType = {
    "SPECIFICATION": "Technical parameters and measurements",
    "PROCEDURE": "Step-by-step processes and workflows", 
    "PROCESS": "High-level business processes",
    "RISK": "Safety hazards and risk assessments",
    "DEFINITION": "Terms and knowledge definitions",
    "ROLE": "Personnel roles and responsibilities",
    "COMPLIANCE_ITEM": "Regulatory requirements",
    "QUANTITATIVE_DATA": "Numerical data and measurements"
}
```

## Semantic Prompt Templates

### **5 Specialized Prompt Types**

#### **1. Key Processes Prompt**
```
Analyze this {document_type} document and extract the key processes, procedures, and workflows described.
For each process, identify:
1. The main steps or phases
2. Required inputs or prerequisites  
3. Expected outputs or results
4. Responsible parties or roles
5. Critical decision points

Target Confidence: 0.85+
```

#### **2. Safety Requirements Prompt**
```
Analyze this {document_type} document and extract all safety requirements, hazards, and risk mitigation measures.
For each safety item, identify:
1. The specific requirement or hazard
2. Severity level (critical, high, medium, low)
3. Required protective equipment or measures
4. Consequences of non-compliance
5. Applicable scenarios or conditions

Target Confidence: 0.90+
```

#### **3. Technical Specifications Prompt**
```
Analyze this {document_type} document and extract all technical specifications, parameters, and measurements.
For each specification, identify:
1. The component or system being specified
2. Numerical values with units
3. Operating ranges or limits
4. Performance criteria
5. Testing or verification methods

Target Confidence: 0.95+
```

#### **4. Compliance Requirements Prompt**
```
Analyze this {document_type} document and extract all compliance requirements, standards, and regulatory information.
For each requirement, identify:
1. The specific standard or regulation
2. Mandatory vs. recommended requirements
3. Compliance verification methods
4. Documentation requirements
5. Responsible parties

Target Confidence: 0.80+
```

#### **5. Organizational Info Prompt**
```
Analyze this {document_type} document and extract organizational information including roles, responsibilities, and personnel.
For each organizational element, identify:
1. Specific roles or positions
2. Responsibilities and authorities
3. Required qualifications or certifications
4. Reporting relationships
5. Contact information if available

Target Confidence: 0.75+
```

## Model Configuration

### **Primary LLM Configuration**

```json
{
  "model_name": "Mistral-7B-Instruct-v0.2",
  "quantization": "Q4_K_M",
  "context_length": 4096,
  "temperature": 0.3,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "max_tokens": 1000,
  "stop_tokens": ["\n\n", "Document content:", "---"],
  "hardware_optimization": {
    "metal_layers": -1,
    "threads": 8,
    "batch_size": 4
  }
}
```

### **Processing Engine Settings**

```python
PROCESSING_CONFIG = {
    "llm_primary_enabled": True,
    "enhanced_pattern_enabled": True,
    "legacy_fallback_enabled": True,
    "validation_required": True,
    "quality_gates_enabled": True,
    "relationship_extraction": True,
    "context_preservation": True,
    "structured_output_required": True
}
```

## Performance Benchmarks

### **Processing Speed Benchmarks**

| Document Type | Size | LLM Processing Time | Entities Extracted | Confidence |
|---------------|------|-------------------|-------------------|------------|
| **Technical Manual** | 20 lines | 2.3s | 24 entities | 0.87 |
| **Safety Document** | 50 pages | 45s | 157 entities | 0.89 |
| **Procedure Guide** | 100 pages | 89s | 324 entities | 0.85 |
| **Compliance Doc** | 200 pages | 156s | 489 entities | 0.83 |

### **Quality Improvement Metrics**

| Metric | Baseline | Current | Improvement |
|--------|-------------------|-----------------|-------------|
| **Entity Extraction** | 4 avg | 24 avg | **6x increase** |
| **Confidence Score** | 0.55 avg | 0.87 avg | **58% increase** |
| **Processing Methods** | 1 basic | 5 intelligent | **5x more** |
| **Quality Rating** | 100% POOR | 85% EXCELLENT | **Massive** |
| **Validation Pass Rate** | 45% | 92% | **104% increase** |

## Database Schema

### **Knowledge Entities Table**
```sql
CREATE TABLE knowledge_entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    key_identifier VARCHAR(200) NOT NULL,
    core_content TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    extraction_method VARCHAR(50) NOT NULL,
    processing_method VARCHAR(50) DEFAULT 'llm_first',
    llm_enhanced BOOLEAN DEFAULT FALSE,
    validation_passed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    document_id INTEGER REFERENCES documents(id)
);
```

### **Processing Metadata Table**
```sql
CREATE TABLE processing_metadata (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    processing_method VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    quality_metrics JSONB NOT NULL,
    processing_time DECIMAL(8,3) NOT NULL,
    llm_enhanced BOOLEAN DEFAULT FALSE,
    validation_passed BOOLEAN DEFAULT FALSE,
    rules_applied TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

## API Endpoints

### **Core Processing Endpoints**

#### **POST /api/process-document**
```json
{
  "file_path": "/path/to/document.pdf",
  "document_id": 123,
  "processing_options": {
    "force_llm": true,
    "quality_threshold": 0.75,
    "enable_relationships": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "processing_method": "llm_first_processing",
  "confidence_score": 0.87,
  "entities_extracted": 24,
  "processing_time": 2.34,
  "validation_passed": true,
  "quality_metrics": {
    "overall_confidence": 0.87,
    "business_value": 0.82,
    "completeness": 0.89
  }
}
```

#### **GET /api/processing-stats**
```json
{
  "llm_primary_used": 156,
  "enhanced_pattern_used": 23,
  "fallback_used": 4,
  "total_processed": 183,
  "average_confidence": 0.84,
  "validation_pass_rate": 0.91
}
```

## Development Guidelines

### **Code Quality Standards**

1. **LLM-First Compliance**: All new processing must use LLM-first hierarchy
2. **Quality Thresholds**: Minimum 0.75 confidence for LLM processing
3. **Validation Required**: All entities must pass validation gates
4. **Structured Output**: Results must be clean and categorized
5. **Error Handling**: Graceful fallback through processing hierarchy

### **Testing Requirements**

```python
# Minimum test coverage for new features
def test_llm_processing_confidence():
    """LLM processing must achieve ≥0.75 confidence"""
    assert processing_result.confidence_score >= 0.75

def test_validation_gate():
    """All entities must pass validation"""
    assert all(entity.validation_passed for entity in entities)

def test_structured_output():
    """Output must be properly categorized"""
    assert all(entity.category in VALID_CATEGORIES for entity in entities)
```

### **Performance Requirements**

- **LLM Processing Time**: < 60s for documents ≤100 pages
- **Memory Usage**: < 8GB during processing
- **Confidence Target**: ≥0.75 for production deployment
- **Validation Pass Rate**: ≥90% for all processing methods

## Monitoring & Metrics

### **Key Performance Indicators (KPIs)**

```python
MONITORING_METRICS = {
    "processing_method_distribution": {
        "llm_first": "target: >80%",
        "enhanced_pattern": "target: <15%", 
        "legacy_fallback": "target: <5%"
    },
    "quality_metrics": {
        "average_confidence": "target: >0.80",
        "validation_pass_rate": "target: >0.90",
        "business_value_score": "target: >0.75"
    },
    "performance_metrics": {
        "processing_time_p95": "target: <120s",
        "memory_usage_max": "target: <8GB",
        "error_rate": "target: <2%"
    }
}
```

### **Quality Assurance Checks**

1. **Confidence Validation**: All LLM extractions ≥0.75 confidence
2. **Entity Completeness**: Required fields populated for all entities  
3. **Relationship Integrity**: Extracted relationships are valid
4. **Category Compliance**: All entities properly categorized
5. **Processing Method Tracking**: Method used recorded for analytics

---

Technical execution focuses on consistent quality gates, transparent fallback attribution, and reproducible structured outputs.
