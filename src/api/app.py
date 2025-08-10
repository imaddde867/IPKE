"""
EXPLAINIUM - Consolidated Knowledge Extraction API

A clean, professional implementation consolidating all functionality
from the previous main.py and enhanced_main.py files.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Internal imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.database import get_db, init_db
    from database import models, crud
    from processors.processor import DocumentProcessor
    from api.celery_worker import process_document_task, get_task_status
    from middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
    from logging_config import get_logger
    from core.config import config as config_manager
    from exceptions import ProcessingError, ValidationError
except ImportError as e:
    print(f"Import error: {e}")
    # Create minimal fallbacks
    def get_db(): pass
    def init_db(): pass
    models = None
    crud = None
    DocumentProcessor = None
    process_document_task = None
    get_task_status = None
    ErrorHandlingMiddleware = None
    RequestLoggingMiddleware = None
    def get_logger(name): return logging.getLogger(name)
    config_manager = None
    ProcessingError = Exception
    ValidationError = Exception

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EXPLAINIUM - Knowledge Extraction System",
    description="Professional AI-powered system for extracting structured knowledge from documents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware, log_request_body=False, log_response_body=False)
origins = getattr(config_manager, 'get_cors_origins', lambda: ['*'])()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if False:
    # Static mounting removed; Streamlit is the only frontend
    app.mount("/static", StaticFiles(directory="src/frontend/public"), name="static")

# Configuration
UPLOAD_DIRECTORY = config_manager.get_upload_directory()
MAX_FILE_SIZE = config_manager.get_max_file_size()
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize document processor
document_processor = DocumentProcessor()


# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    task_id: str
    message: str
    filename: str


class ProcessSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    hierarchy_level: Optional[str] = None
    confidence_threshold: float = 0.7
    max_results: int = 50


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database: str
    services: Dict[str, str]


# API Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "EXPLAINIUM - Advanced AI-Powered Knowledge Extraction System",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Deep Knowledge Extraction",
            "AI-Powered Analysis", 
            "Knowledge Graph Building",
            "Multi-format Export",
            "Real-time Processing"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    services = {
        "database": db_status,
        "document_processor": "healthy" if document_processor else "unhealthy",
        "celery": "healthy"  # TODO: Add actual celery health check
    }
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        version="2.0.0",
        database=db_status,
        services=services
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Save uploaded file
        file_path = Path(UPLOAD_DIRECTORY) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database record
        document = crud.create_document(
            db=db,
            filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Queue processing task
        task = process_document_task.delay(str(file_path), document.id)
        
        logger.info(f"Document uploaded and queued for processing: {file.filename}")
        
        return DocumentUploadResponse(
            task_id=task.id,
            message="Document uploaded successfully and queued for processing",
            filename=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")


@app.get("/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all documents"""
    documents = crud.get_documents(db, skip=skip, limit=limit)
    return {"documents": documents, "total": len(documents)}


@app.get("/documents/{document_id}")
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document details"""
    document = crud.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/processes")
async def list_processes(
    domain: Optional[str] = Query(None),
    hierarchy_level: Optional[str] = Query(None),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List processes with filtering"""
    processes = crud.get_processes(
        db=db,
        domain=domain,
        hierarchy_level=hierarchy_level,
        confidence_threshold=confidence_threshold,
        skip=skip,
        limit=limit
    )
    return {"processes": processes, "total": len(processes)}


@app.post("/knowledge/search")
async def search_knowledge(
    request: ProcessSearchRequest,
    db: Session = Depends(get_db)
):
    """Advanced knowledge search"""
    try:
        results = crud.search_knowledge(
            db=db,
            query=request.query,
            domain=request.domain,
            hierarchy_level=request.hierarchy_level,
            confidence_threshold=request.confidence_threshold,
            max_results=request.max_results
        )
        return {"results": results, "query": request.query}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """Get processing task status"""
    try:
        status = get_task_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting EXPLAINIUM application")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)