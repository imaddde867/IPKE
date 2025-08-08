from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.celery_worker import process_document_task, get_task_status, retry_failed_task
from src.database.database import get_db, init_db, get_connection_pool_status
from src.database.migrations import run_migrations, get_migration_status
from src.database import models, crud
from src.search.elasticsearch_client import es_client
from src.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from src.logging_config import get_logger, set_correlation_id, log_processing_step
from src.exceptions import DatabaseError, ValidationError, SearchError
from src.config import config_manager, config
from sqlalchemy.orm import Session
import os
import time

# Initialize logging
logger = get_logger(__name__)

# Print configuration summary on startup
config_manager.print_config_summary()

app = FastAPI(
    title="Industrial Knowledge Extraction System",
    description="AI-powered system for extracting structured knowledge from industrial documentation",
    version="1.0.0"
)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="src/frontend/public"), name="static")

# Add middleware (order matters - error handling should be first)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware, log_request_body=False, log_response_body=False)

# CORS configuration from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from config manager
UPLOAD_DIRECTORY = config.processing.upload_directory
TIKA_SERVER_URL = config_manager.get_tika_url()
MAX_FILE_SIZE = config.processing.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize database tables on startup
@app.on_event("startup")
def on_startup():
    correlation_id = set_correlation_id()
    logger.info("Application startup initiated", extra={'startup_correlation_id': correlation_id})
    
    try:
        log_processing_step(logger, "database_initialization", "started")
        init_db()
        log_processing_step(logger, "database_initialization", "completed")
        
        log_processing_step(logger, "database_migrations", "started")
        migration_count = run_migrations()
        log_processing_step(logger, "database_migrations", "completed")
        logger.info(f"Database migrations completed: {migration_count} migrations applied")
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        log_processing_step(logger, "database_initialization", "failed")
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)
        raise

@app.get("/")
def read_root():
    """Serve the main frontend application."""
    return FileResponse("src/frontend/public/simple.html")

@app.get("/api")
def api_root():
    logger.info("API root endpoint accessed")
    return {
        "message": "Welcome to the Industrial Knowledge Extraction System API!",
        "correlation_id": set_correlation_id()
    }

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "correlation_id": set_correlation_id()
    }

@app.get("/health/detailed")
def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with service dependency verification."""
    correlation_id = set_correlation_id()
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "correlation_id": correlation_id,
        "services": {}
    }
    
    # Check database connection
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        health_status["services"]["database"] = {"status": "healthy", "type": "PostgreSQL"}
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy", 
            "type": "PostgreSQL",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Elasticsearch
    try:
        if es_client.es is None:
            es_client._ensure_connection()
        es_health = es_client.es.cluster.health()
        health_status["services"]["elasticsearch"] = {
            "status": "healthy" if es_health["status"] in ["green", "yellow"] else "unhealthy",
            "type": "Elasticsearch",
            "cluster_status": es_health["status"]
        }
    except Exception as e:
        health_status["services"]["elasticsearch"] = {
            "status": "unhealthy",
            "type": "Elasticsearch", 
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Tika server
    try:
        import requests
        response = requests.get(f"{TIKA_SERVER_URL}/version", timeout=config.processing.service_connection_timeout_seconds)
        if response.status_code == 200:
            health_status["services"]["tika"] = {"status": "healthy", "type": "Apache Tika"}
        else:
            health_status["services"]["tika"] = {
                "status": "unhealthy",
                "type": "Apache Tika",
                "error": f"HTTP {response.status_code}"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["tika"] = {
            "status": "unhealthy",
            "type": "Apache Tika",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis (Celery broker)
    try:
        import redis
        r = redis.Redis(
            host=config.redis.host, 
            port=config.redis.port, 
            db=config.redis.db
        )
        r.ping()
        health_status["services"]["redis"] = {"status": "healthy", "type": "Redis"}
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "type": "Redis",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    logger.info(
        f"Health check completed: {health_status['status']}",
        extra={'health_status': health_status['status'], 'services_checked': len(health_status['services'])}
    )
    
    return health_status

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    correlation_id = set_correlation_id()
    
    # Validate file
    if not file.filename:
        raise ValidationError("Filename is required", field="filename")
    
    # Check file size using configured limit
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise ValidationError(
            f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes ({config.processing.max_file_size_mb}MB)",
            field="file_size",
            value=len(file_content)
        )
    
    logger.info(
        f"File upload started: {file.filename}",
        extra={
            'uploaded_filename': file.filename,
            'file_size': len(file_content),
            'content_type': file.content_type
        }
    )
    
    try:
        # Save file
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        # Queue processing task
        task = process_document_task.delay(file_location, correlation_id)
        
        logger.info(
            f"File uploaded and queued for processing: {file.filename}",
            extra={
                'uploaded_filename': file.filename,
                'file_location': file_location,
                'task_id': task.id
            }
        )
        
        return {
            "info": f"File '{file.filename}' uploaded successfully",
            "task_id": task.id,
            "correlation_id": correlation_id,
            "file_location": file_location
        }
        
    except Exception as e:
        logger.error(f"File upload failed for {file.filename}: {str(e)}", exc_info=True)
        raise

@app.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    try:
        logger.info(f"Retrieving document: {document_id}")
        db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if db_document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Document retrieved successfully: {document_id}")
        return db_document
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to retrieve document {document_id}",
            operation="select",
            table="documents",
            details={'document_id': document_id}
        ) from e

@app.get("/documents/")
def get_all_documents(
    page: int = 1, 
    per_page: int = 20, 
    status: str = None, 
    file_type: str = None, 
    db: Session = Depends(get_db)
):
    """Get documents with pagination and optional filtering."""
    try:
        from src.database.crud import get_documents
        
        logger.info(f"Documents list request: page={page}, per_page={per_page}, status={status}, file_type={file_type}")
        
        result = get_documents(db, page=page, per_page=per_page, status=status, file_type=file_type)
        
        logger.info(f"Documents retrieved: {len(result['items'])} of {result['total']} total")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        raise DatabaseError(
            "Failed to retrieve documents",
            operation="select",
            table="documents"
        ) from e

@app.get("/documents/{document_id}/entities/")
def get_document_entities(document_id: int, db: Session = Depends(get_db)):
    entities = db.query(models.ExtractedEntity).filter(models.ExtractedEntity.document_id == document_id).all()
    if not entities:
        raise HTTPException(status_code=404, detail="Entities not found for this document")
    return entities

@app.get("/documents/{document_id}/keyphrases/")
def get_document_keyphrases(document_id: int, db: Session = Depends(get_db)):
    key_phrases = db.query(models.KeyPhrase).filter(models.KeyPhrase.document_id == document_id).all()
    if not key_phrases:
        raise HTTPException(status_code=404, detail="Key phrases not found for this document")
    return key_phrases

@app.get("/documents/{document_id}/equipment/")
def get_document_equipment(document_id: int, db: Session = Depends(get_db)):
    equipment = db.query(models.Equipment).filter(models.Equipment.document_id == document_id).all()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment data not found for this document")
    return equipment

@app.get("/documents/{document_id}/procedures/")
def get_document_procedures(document_id: int, db: Session = Depends(get_db)):
    procedures = db.query(models.Procedure).filter(models.Procedure.document_id == document_id).all()
    if not procedures:
        raise HTTPException(status_code=404, detail="Procedure data not found for this document")
    return procedures

@app.get("/documents/{document_id}/safety_info/")
def get_document_safety_info(document_id: int, db: Session = Depends(get_db)):
    safety_info = db.query(models.SafetyInformation).filter(models.SafetyInformation.document_id == document_id).all()
    if not safety_info:
        raise HTTPException(status_code=404, detail="Safety information not found for this document")
    return safety_info

@app.get("/documents/{document_id}/technical_specs/")
def get_document_technical_specs(document_id: int, db: Session = Depends(get_db)):
    technical_specs = db.query(models.TechnicalSpecification).filter(models.TechnicalSpecification.document_id == document_id).all()
    if not technical_specs:
        raise HTTPException(status_code=404, detail="Technical specifications not found for this document")
    return technical_specs

@app.get("/documents/{document_id}/personnel/")
def get_document_personnel(document_id: int, db: Session = Depends(get_db)):
    personnel = db.query(models.Personnel).filter(models.Personnel.document_id == document_id).all()
    if not personnel:
        raise HTTPException(status_code=404, detail="Personnel data not found for this document")
    return personnel

@app.get("/documents/{document_id}/sections/")
def get_document_sections(document_id: int, db: Session = Depends(get_db)):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document.document_sections

@app.get("/search/")
def search_documents(query: str, field: str = "extracted_text", size: int = 10):
    # Validate inputs
    if not query or not query.strip():
        raise ValidationError("Search query cannot be empty", field="query", value=query)
    
    valid_fields = ["extracted_text", "filename", "classification_category", "extracted_entities.text", "key_phrases", "document_sections"]
    if field not in valid_fields:
        raise ValidationError(
            f"Invalid search field. Must be one of: {', '.join(valid_fields)}",
            field="field",
            value=field,
            validation_rule="must_be_in_allowed_list"
        )
    
    if size < 1 or size > 100:
        raise ValidationError(
            "Size must be between 1 and 100",
            field="size",
            value=size,
            validation_rule="range_1_to_100"
        )
    
    try:
        logger.info(
            f"Search request: query='{query}', field='{field}', size={size}",
            extra={'search_query': query, 'search_field': field, 'search_size': size}
        )
        
        results = es_client.search_documents(query=query, field=field, size=size)
        
        logger.info(
            f"Search completed: found {len(results.get('hits', {}).get('hits', []))} results",
            extra={'search_query': query, 'results_count': len(results.get('hits', {}).get('hits', []))}
        )
        
        return results
        
    except Exception as e:
        raise SearchError(
            f"Search failed for query: {query}",
            query=query,
            operation="search",
            details={'field': field, 'size': size}
        ) from e
# Task status tracking endpoints

@app.get("/tasks/{task_id}/status")
def get_task_status_endpoint(task_id: str):
    """Get detailed status information for a processing task."""
    correlation_id = set_correlation_id()
    
    try:
        logger.info(f"Task status request: {task_id}", extra={
            'task_id': task_id,
            'correlation_id': correlation_id
        })
        
        # Get task status from Celery
        status_info = get_task_status.delay(task_id).get(timeout=10)
        
        logger.info(f"Task status retrieved: {task_id}", extra={
            'task_id': task_id,
            'task_state': status_info.get('state'),
            'correlation_id': correlation_id
        })
        
        return {
            **status_info,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get task status: {task_id} - {str(e)}", extra={
            'task_id': task_id,
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task status: {str(e)}"
        )

@app.post("/tasks/{task_id}/retry")
def retry_task_endpoint(task_id: str, db: Session = Depends(get_db)):
    """Manually retry a failed task from the dead letter queue."""
    correlation_id = set_correlation_id()
    
    try:
        logger.info(f"Task retry request: {task_id}", extra={
            'task_id': task_id,
            'correlation_id': correlation_id
        })
        
        # This would typically look up the original task details from a dead letter table
        # For now, we'll return an error indicating manual intervention is needed
        logger.warning(f"Manual task retry requested but not implemented: {task_id}")
        
        return {
            'status': 'not_implemented',
            'message': 'Manual task retry requires implementation of dead letter queue storage',
            'task_id': task_id,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to retry task: {task_id} - {str(e)}", extra={
            'task_id': task_id,
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry task: {str(e)}"
        )

@app.get("/tasks/stats")
def get_task_statistics():
    """Get overall task processing statistics."""
    correlation_id = set_correlation_id()
    
    try:
        # This would typically query task statistics from the database or Celery
        # For now, return basic information
        stats = {
            'active_tasks': 'Not implemented',
            'completed_tasks': 'Not implemented', 
            'failed_tasks': 'Not implemented',
            'dead_letter_tasks': 'Not implemented',
            'correlation_id': correlation_id
        }
        
        logger.info("Task statistics requested", extra={
            'correlation_id': correlation_id
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get task statistics: {str(e)}", extra={
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task statistics: {str(e)}"
        )

# Database management endpoints

@app.get("/admin/database/pool-status")
def get_database_pool_status():
    """Get current database connection pool status."""
    correlation_id = set_correlation_id()
    
    try:
        pool_status = get_connection_pool_status()
        logger.info("Database pool status retrieved", extra={
            'pool_status': pool_status,
            'correlation_id': correlation_id
        })
        
        return {
            **pool_status,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get database pool status: {str(e)}", extra={
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database pool status: {str(e)}"
        )

@app.get("/admin/database/migration-status")
def get_database_migration_status():
    """Get current database migration status."""
    correlation_id = set_correlation_id()
    
    try:
        migration_status = get_migration_status()
        logger.info("Database migration status retrieved", extra={
            'migration_status': migration_status,
            'correlation_id': correlation_id
        })
        
        return {
            **migration_status,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get migration status: {str(e)}", extra={
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve migration status: {str(e)}"
        )

@app.post("/admin/database/run-migrations")
def run_database_migrations():
    """Manually run pending database migrations."""
    correlation_id = set_correlation_id()
    
    try:
        logger.info("Manual migration run requested", extra={
            'correlation_id': correlation_id
        })
        
        migration_count = run_migrations()
        
        logger.info(f"Manual migrations completed: {migration_count} applied", extra={
            'migrations_applied': migration_count,
            'correlation_id': correlation_id
        })
        
        return {
            'status': 'completed',
            'migrations_applied': migration_count,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {str(e)}", extra={
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run migrations: {str(e)}"
        )

@app.get("/admin/database/statistics")
def get_database_statistics(db: Session = Depends(get_db)):
    """Get database processing statistics."""
    correlation_id = set_correlation_id()
    
    try:
        from src.database.crud import get_processing_statistics
        
        stats = get_processing_statistics(db)
        logger.info("Database statistics retrieved", extra={
            'correlation_id': correlation_id
        })
        
        return {
            **stats,
            'correlation_id': correlation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get database statistics: {str(e)}", extra={
            'error': str(e),
            'correlation_id': correlation_id
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database statistics: {str(e)}"
        )