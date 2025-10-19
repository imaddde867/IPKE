"""
EXPLAINIUM - Streamlined FastAPI app
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Internal imports
from src.core.unified_config import get_config, UnifiedConfig
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from src.ai.unified_knowledge_engine import ExtractionResult
from src.logging_config import get_logger
from src.exceptions import ProcessingError
from src.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware

logger = get_logger(__name__)

# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str


class ProcessingResponse(BaseModel):
    document_id: str
    status: str
    entities_extracted: int
    confidence_score: float
    processing_time: float
    message: str


class EntityResponse(BaseModel):
    content: str
    entity_type: str
    category: str
    confidence: float
    context: str


class ExtractionResponse(BaseModel):
    document_id: str
    document_type: str
    entities: List[EntityResponse]
    confidence_score: float
    processing_time: float
    strategy_used: str
    metadata: Dict[str, Any]


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    config = get_config()
    
    app = FastAPI(
        title="Explainium Knowledge Extraction API",
        description="Simplified, high-performance knowledge extraction from documents",
        version="2.0",
        debug=config.is_development()
    )
    
    # Add middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()
config = get_config()

# Initialize processor
processor = StreamlinedDocumentProcessor()

# Ensure upload directory exists
upload_dir = Path(config.upload_directory)
upload_dir.mkdir(exist_ok=True)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic health information"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0",
        environment=config.environment.value
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0",
        environment=config.environment.value
    )


@app.get("/config")
async def get_config_info():
    """Get system configuration information"""
    config = get_config()
    llm_config = config.get_llm_config()
    
    return {
        "gpu_enabled": llm_config['enable_gpu'],
        "gpu_backend": llm_config['gpu_backend'],
        "detected_backend": config.detect_gpu_backend(),
        "gpu_layers": llm_config['n_gpu_layers'],
        "model_path": llm_config['model_path'],
        "max_chunks": llm_config['max_chunks'],
        "max_tokens": llm_config['max_tokens'],
        "confidence_threshold": llm_config['confidence_threshold']
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_knowledge(
    file: UploadFile = File(...)
):
    """
    Extract knowledge from an uploaded document
    
    Args:
        file: Document file to process
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > config.get_max_file_size():
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {config.max_file_size_mb}MB"
        )
    
    # Check file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.supported_formats:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: {file_ext}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Process document
        result = await processor.process_document(
            file_path=tmp_file_path
        )
        
        # Convert entities to response format
        entities = [
            EntityResponse(
                content=entity.content,
                entity_type=entity.entity_type,
                category=entity.category,
                confidence=entity.confidence,
                context=entity.context
            )
            for entity in result.extraction_result.entities
        ]
        
        response = ExtractionResponse(
            document_id=result.document_id,
            document_type=result.document_type,
            entities=entities,
            confidence_score=result.extraction_result.confidence_score,
            processing_time=result.processing_time,
            strategy_used=result.extraction_result.strategy_used,
            metadata=result.metadata
        )
        
        logger.info(f"Successfully extracted {len(entities)} entities from {file.filename} "
                   f"in {result.processing_time:.2f}s")
        
        return response
        
    except ProcessingError as e:
        logger.error(f"Processing failed for {file.filename}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except OSError:
            pass


@app.get("/stats")
async def get_statistics():
    """Get processing statistics"""
    stats = processor.get_processing_stats()
    return {
        "processing_stats": stats,
        "api_info": {
            "version": "2.0",
            "environment": config.environment.value,
            "supported_formats": config.supported_formats,
            "max_file_size_mb": config.max_file_size_mb
        }
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear all processing caches"""
    processor.clear_cache()
    return {"message": "Caches cleared successfully"}


# Error handlers
@app.exception_handler(ProcessingError)
async def processing_error_handler(request, exc: ProcessingError):
    return JSONResponse(
        status_code=422,
        content={"error": "Processing Error", "detail": str(exc)}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP Error", "detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Explainium API starting up in {config.environment.value} environment")
    logger.info(f"Upload directory: {config.upload_directory}")
    logger.info(f"Max file size: {config.max_file_size_mb}MB")
    logger.info(f"Supported formats: {len(config.supported_formats)} types")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Explainium API shutting down")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.is_development(),
        log_level=config.log_level.lower()
    )
