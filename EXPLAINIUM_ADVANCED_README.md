# EXPLAINIUM Advanced AI Knowledge Processing System

## Overview

EXPLAINIUM has been completely transformed from a basic entity extraction system into a sophisticated, AI-powered knowledge processing platform optimized for Apple M4 Mac with 16GB RAM. The new system extracts deep, meaningful insights from company documents and tacit knowledge using advanced LLM models and multi-modal processing.

## 🚀 Key Improvements

### What Was Replaced
- ❌ Basic BERT/BART models (`facebook/bart-large-mnli`, `dslim/bert-base-NER`)
- ❌ Superficial spaCy-based entity extraction
- ❌ Simple zero-shot classification
- ❌ Surface-level text processing without context understanding

### What's New
- ✅ **Consolidated Processor**: Single robust `DocumentProcessor` for all formats
- ✅ **Multi-Modal Processing**: PDF, DOCX, images, audio, video, spreadsheets, presentations
- ✅ **Video Robustness**: Whisper transcription + frame OCR fallback
- ✅ **Cleaner Dependencies**: Removed unused heavy components
- ✅ **Apple Silicon Optimization**: Metal acceleration, memory management, and M4-specific optimizations
- ✅ **Interactive Dashboard**: Clean table-based frontend with filtering and visualization
- ✅ **Advanced Export**: Multiple formats including interactive visualizations

## 🏗️ Architecture

### Core Components

1. **AdvancedKnowledgeEngine** (`src/ai/advanced_knowledge_engine.py`)
   - Multi-pass knowledge extraction
   - LLM-powered deep understanding
   - Knowledge graph building
   - Tacit knowledge identification

2. **IntelligentDocumentProcessor** (`src/processors/intelligent_document_processor.py`)
   - Multi-modal content processing
   - Enhanced OCR with EasyOCR + Tesseract
   - Audio transcription with speaker diarization
   - Video processing with audio extraction

3. **M4 Optimization Module** (`src/core/optimization.py`)
   - Memory monitoring and management
   - Model caching and loading optimization
   - Batch processing optimization
   - Streaming for large documents

4. **Knowledge Export System** (`src/export/knowledge_export.py`)
   - Interactive HTML visualizations
   - Knowledge graphs (GEXF, GraphML, JSON)
   - Automated documentation generation
   - Training material creation

5. **Interactive Dashboard** (`src/frontend/knowledge_dashboard.py`)
   - Clean table-based display
   - Real-time filtering and search
   - Statistical visualizations
   - Export capabilities

## 🛠️ Installation & Setup

### Prerequisites
- Apple M4 Mac with 16GB RAM
- Python 3.9+
- ffmpeg (for video processing)
- Tesseract OCR

### Installation

```bash
# Install system dependencies (macOS)
brew install ffmpeg tesseract

# Install Python dependencies
pip install -r requirements.txt

# Download and setup models (optional - models will be downloaded automatically)
mkdir -p models
# Place your quantized LLM models in the models directory
```

### Model Setup

The system supports multiple LLM configurations based on available RAM:

1. **Primary**: Mistral-7B-Instruct (4-bit quantized, ~4GB RAM)
2. **Alternative**: Microsoft Phi-2 (2.7B params, ~3GB RAM)
3. **Fallback**: TinyLlama-1.1B (1.1B params, ~2GB RAM)

Models are automatically downloaded and cached on first use.

## 🚀 Usage

### Basic Document Processing

```python
from src.processors.intelligent_document_processor import IntelligentDocumentProcessor
import asyncio

async def process_document():
    processor = IntelligentDocumentProcessor()
    
    # Process any supported document type
    result = await processor.process_document(
        file_path="path/to/document.pdf",
        document_id=1,
        company_context={"industry": "manufacturing", "size": "large"}
    )
    
    print(f"Extracted {len(result['knowledge_extraction']['entities'])} entities")
    print(f"Identified {len(result['knowledge_extraction']['processes'])} processes")
    
asyncio.run(process_document())
```

### Advanced Knowledge Extraction

```python
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
import asyncio

async def extract_knowledge():
    engine = AdvancedKnowledgeEngine()
    
    document = {
        'id': 'doc_001',
        'content': 'Your document content here...',
        'metadata': {'source': 'policy_manual.pdf'}
    }
    
    # Multi-pass extraction
    results = await engine.extract_deep_knowledge(document)
    
    # Build knowledge graph
    knowledge_graph = await engine.build_knowledge_graph(results)
    
    # Extract operational intelligence
    ops_intel = await engine.extract_operational_intelligence(document['content'])
    
    return results, knowledge_graph, ops_intel

asyncio.run(extract_knowledge())
```

### Batch Processing

```python
from src.processors.intelligent_document_processor import IntelligentDocumentProcessor
import asyncio

async def batch_process():
    processor = IntelligentDocumentProcessor()
    
    file_paths = [
        "documents/policy1.pdf",
        "documents/manual.docx",
        "documents/presentation.pptx"
    ]
    
    results = await processor.process_batch_documents(
        file_paths,
        company_context={"department": "operations"}
    )
    
    for result in results:
        print(f"Processed: {result['filename']}")

asyncio.run(batch_process())
```

### Knowledge Export

```python
from src.export.knowledge_export import KnowledgeExporter
import asyncio

async def export_knowledge():
    exporter = KnowledgeExporter(output_dir="./exports")
    
    # Export knowledge graph in multiple formats
    graph_files = await exporter.export_knowledge_graph(knowledge_graph, format="all")
    
    # Generate documentation
    doc_path = await exporter.generate_documentation(knowledge_results)
    
    # Create training materials
    training_materials = await exporter.create_training_materials(knowledge_results)
    
    # Export as data table
    table_path = await exporter.export_data_table(knowledge_results)
    
    print(f"Exported to: {exporter.output_dir}")

asyncio.run(export_knowledge())
```

### Interactive Dashboard

```python
from src.frontend.knowledge_dashboard import create_knowledge_dashboard

# Create dashboard with extracted knowledge
dashboard = create_knowledge_dashboard(
    knowledge_results,
    port=8050,
    debug=False
)

# Run the dashboard server
dashboard.run(host="0.0.0.0", port=8050)
```

## 📊 Dashboard Features

The new interactive dashboard provides:

- **Large Data Table**: Sortable, filterable display of all extracted knowledge
- **Real-time Filtering**: By type, category, confidence threshold
- **Search Functionality**: Full-text search across names and descriptions
- **Statistics Cards**: Key metrics and confidence scores
- **Visualizations**: 
  - Entity type distribution (pie chart)
  - Confidence score distribution (histogram)
  - Category breakdown (bar chart)
- **Export Capabilities**: CSV export with display headers
- **Color Coding**: Different colors for entities, processes, relationships, tacit knowledge

## 🔧 Configuration

### AI Configuration (`src/core/config.py`)

```python
@dataclass
class AIConfig:
    # Core LLM configuration
    llm_model: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    llm_path: str = "./models"
    fallback_model: str = "microsoft/phi-2"
    
    # Model optimization
    max_tokens: int = 2048
    temperature: float = 0.1
    quantization: str = "4bit"
    use_gpu: bool = True  # Metal acceleration
    
    # Embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # Performance settings
    batch_size: int = 4  # Optimized for 16GB RAM
    max_memory_mb: int = 4096
    
    # Advanced features
    enable_knowledge_graph: bool = True
    enable_tacit_extraction: bool = True
    enable_relationship_extraction: bool = True
```

## 📁 Supported File Formats

### Document Types
- **Text**: PDF, DOC, DOCX, TXT, RTF, Markdown
- **Images**: JPG, PNG, GIF, BMP, TIFF, WEBP (with OCR)
- **Spreadsheets**: XLS, XLSX, CSV
- **Presentations**: PPT, PPTX
- **Audio**: MP3, WAV, FLAC, AAC, M4A (with transcription)
- **Video**: MP4, AVI, MOV, MKV, WEBM (audio extraction)

### Processing Capabilities
- **OCR**: Tesseract OCR with preprocessing
- **Audio Processing**: Whisper transcription
- **Vision**: Frame sampling OCR for videos

## 🎯 Knowledge Extraction Features

### Entity Types
- **People**: Roles, responsibilities, contact information
- **Processes**: Business workflows, procedures, decision points
- **Systems**: Technology, tools, equipment
- **Concepts**: Business terminology, domain knowledge
- **Requirements**: Compliance, regulatory, operational
- **Risks**: Factors, mitigation strategies, assessments

### Relationship Types
- **Functional**: depends_on, requires, implements
- **Causal**: leads_to, causes, affects
- **Structural**: contains, part_of, member_of
- **Comparative**: similar_to, replaces, conflicts_with

### Tacit Knowledge Discovery
- **Unstated Assumptions**: Implicit business rules
- **Best Practices**: Informal procedures and habits
- **Tribal Knowledge**: Experience-based insights
- **Patterns**: Recurring themes and workflows

## 📈 Performance Optimization

### Apple M4 Specific
- **Metal Acceleration**: GPU processing for compatible models
- **Memory Management**: Real-time monitoring with automatic cleanup
- **Quantized Models**: 4-bit quantization for memory efficiency
- **Batch Processing**: Adaptive batch sizes based on available RAM
- **Streaming**: Memory-efficient processing for large documents

### Caching Strategy
- **Model Caching**: 4GB disk cache for loaded models
- **LRU Management**: Automatic unloading of least recently used models
- **Vector Database**: ChromaDB for semantic similarity search
- **Result Caching**: Processed document results

## 📊 Export Formats

### Knowledge Graphs
- **GEXF**: For Gephi visualization
- **GraphML**: For yEd and other graph tools
- **JSON**: Structured data format
- **Cytoscape.js**: Web-based visualization
- **Interactive HTML**: Pyvis-powered visualizations

### Documentation
- **Word Documents**: Professional reports with formatting
- **Markdown**: Clean, readable documentation
- **Training Guides**: Process-specific training materials
- **Quick Reference**: Key insights and statistics

### Data Formats
- **CSV**: Comprehensive data tables
- **Interactive Dashboards**: Plotly-powered visualizations
- **Knowledge Tables**: Sortable, filterable data exports

## 🔄 Backward Compatibility

The new system maintains full backward compatibility with existing code:

- **KnowledgeExtractor**: Compatible wrapper using advanced engine
- **DocumentProcessor**: Enhanced capabilities with legacy API support
- **Configuration**: Extended settings with sensible defaults
- **Output Formats**: Enhanced results with legacy field mapping

## 🚦 System Requirements

### Minimum Requirements
- Apple M4 Mac (or M1/M2/M3 with reduced performance)
- 16GB RAM (12GB minimum)
- 50GB free disk space (for models and cache)
- macOS 12.0 or later

### Recommended Requirements
- Apple M4 Pro/Max
- 32GB RAM (for optimal performance)
- SSD storage for model cache
- External storage for large document collections

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in configuration
   - Enable aggressive memory cleanup
   - Use smaller model variants

2. **Model Loading Failures**
   - Check internet connection for model downloads
   - Verify disk space for model cache
   - Use fallback models if primary models fail

3. **Processing Errors**
   - Check file permissions and accessibility
   - Verify supported file formats
   - Review system logs for detailed error information

### Performance Tuning

1. **For Large Documents**
   - Enable streaming processing
   - Increase chunk overlap for better context
   - Use batch processing for multiple files

2. **For Limited Memory**
   - Use TinyLlama instead of Mistral
   - Reduce context window size
   - Disable knowledge graph features if needed

## 📚 API Reference

### Main Classes

- `AdvancedKnowledgeEngine`: Core AI processing engine
- `IntelligentDocumentProcessor`: Multi-modal document processing
- `KnowledgeExporter`: Export and visualization tools
- `KnowledgeDashboard`: Interactive web interface
- `ModelManager`: M4-optimized model management

### Key Methods

- `extract_deep_knowledge()`: Multi-pass knowledge extraction
- `process_document()`: Intelligent document processing
- `build_knowledge_graph()`: Knowledge graph construction
- `export_knowledge_graph()`: Graph export in multiple formats
- `create_knowledge_dashboard()`: Dashboard initialization

## 🤝 Contributing

The new system is designed for extensibility:

1. **Custom Extractors**: Add domain-specific knowledge extractors
2. **New Formats**: Extend document format support
3. **Model Integration**: Add new LLM or embedding models
4. **Export Formats**: Create additional visualization formats
5. **Dashboard Components**: Enhance the web interface

## 📄 License

EXPLAINIUM Advanced AI Knowledge Processing System
Licensed under [Your License Here]

## 📞 Support

For technical support and questions:
- Review this documentation
- Check system logs for detailed error information
- Verify system requirements and configuration
- Test with smaller documents first

---

*EXPLAINIUM v2.0 - Transforming documents into actionable knowledge through advanced AI*