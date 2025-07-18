from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.celery_worker import process_document_task
from src.database.database import get_db, init_db, engine
from src.database import models, crud
from src.search.elasticsearch_client import es_client
from sqlalchemy.orm import Session
from sqlalchemy import text
import os
import asyncio
import time
import logging
from typing import Dict, Any
import redis
import requests
from elasticsearch import Elasticsearch

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Allow requests from your React frontend
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configuration
SERVICES_CONFIG = {
    "postgresql": {
        "host": "db",
        "port": 5432,
        "timeout": 5
    },
    "elasticsearch": {
        "url": "http://elasticsearch:9200",
        "timeout": 5
    },
    "redis": {
        "host": "redis",
        "port": 6379,
        "timeout": 5
    },
    "tika": {
        "url": "http://tika:9998",
        "timeout": 5
    }
}

async def check_postgresql_health() -> Dict[str, Any]:
    """Check PostgreSQL database connectivity"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        return {
            "status": "healthy",
            "message": "PostgreSQL connection successful",
            "response_time": None
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"PostgreSQL connection failed: {str(e)}",
            "response_time": None
        }

async def check_elasticsearch_health() -> Dict[str, Any]:
    """Check Elasticsearch connectivity"""
    start_time = time.time()
    try:
        # Use requests for a simple health check to avoid version compatibility issues
        response = requests.get(
            f"{SERVICES_CONFIG['elasticsearch']['url']}/_cluster/health",
            timeout=SERVICES_CONFIG["elasticsearch"]["timeout"]
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            health_data = response.json()
            return {
                "status": "healthy" if health_data["status"] in ["green", "yellow"] else "unhealthy",
                "message": f"Elasticsearch cluster status: {health_data['status']}",
                "response_time": round(response_time * 1000, 2),
                "cluster_name": health_data.get("cluster_name"),
                "number_of_nodes": health_data.get("number_of_nodes")
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Elasticsearch returned status code: {response.status_code}",
                "response_time": round(response_time * 1000, 2)
            }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Elasticsearch health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Elasticsearch connection failed: {str(e)}",
            "response_time": round(response_time * 1000, 2)
        }

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity"""
    start_time = time.time()
    try:
        r = redis.Redis(
            host=SERVICES_CONFIG["redis"]["host"],
            port=SERVICES_CONFIG["redis"]["port"],
            socket_timeout=SERVICES_CONFIG["redis"]["timeout"]
        )
        r.ping()
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "message": "Redis connection successful",
            "response_time": round(response_time * 1000, 2)
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}",
            "response_time": round(response_time * 1000, 2)
        }

async def check_tika_health() -> Dict[str, Any]:
    """Check Apache Tika server connectivity"""
    start_time = time.time()
    try:
        response = requests.get(
            f"{SERVICES_CONFIG['tika']['url']}/version",
            timeout=SERVICES_CONFIG["tika"]["timeout"]
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "message": "Tika server is responding",
                "response_time": round(response_time * 1000, 2),
                "version": response.text.strip()
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Tika server returned status code: {response.status_code}",
                "response_time": round(response_time * 1000, 2)
            }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Tika health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Tika server connection failed: {str(e)}",
            "response_time": round(response_time * 1000, 2)
        }

async def verify_service_dependencies(max_retries: int = 5, retry_delay: int = 5) -> bool:
    """Verify all service dependencies are available with retry logic"""
    services = ["postgresql", "elasticsearch", "redis", "tika"]
    
    for service in services:
        logger.info(f"Verifying {service} dependency...")
        
        for attempt in range(max_retries):
            try:
                if service == "postgresql":
                    result = await check_postgresql_health()
                elif service == "elasticsearch":
                    result = await check_elasticsearch_health()
                elif service == "redis":
                    result = await check_redis_health()
                elif service == "tika":
                    result = await check_tika_health()
                
                if result["status"] == "healthy":
                    logger.info(f"{service} dependency verified successfully")
                    break
                else:
                    logger.warning(f"{service} dependency check failed (attempt {attempt + 1}/{max_retries}): {result['message']}")
                    
            except Exception as e:
                logger.error(f"Error checking {service} dependency (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying {service} dependency check in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to verify {service} dependency after {max_retries} attempts")
                return False
    
    logger.info("All service dependencies verified successfully")
    return True

# Initialize database tables and verify dependencies on startup
@app.on_event("startup")
async def on_startup():
    logger.info("Starting application initialization...")
    
    # Initialize database tables
    try:
        init_db()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise
    
    # Verify service dependencies
    try:
        dependencies_ok = await verify_service_dependencies()
        if not dependencies_ok:
            logger.warning("Some service dependencies are not available - application may have limited functionality")
        else:
            logger.info("All service dependencies are available")
    except Exception as e:
        logger.error(f"Error during dependency verification: {e}")
        logger.warning("Continuing startup despite dependency verification errors")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Industrial Knowledge Extraction System!"}

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "message": "Industrial Knowledge Extraction System is running",
        "timestamp": time.time()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service dependency status"""
    start_time = time.time()
    
    # Check all service dependencies
    services_health = {}
    overall_status = "healthy"
    
    # Check PostgreSQL
    pg_health = await check_postgresql_health()
    services_health["postgresql"] = pg_health
    if pg_health["status"] != "healthy":
        overall_status = "degraded"
    
    # Check Elasticsearch
    es_health = await check_elasticsearch_health()
    services_health["elasticsearch"] = es_health
    if es_health["status"] != "healthy":
        overall_status = "degraded"
    
    # Check Redis
    redis_health = await check_redis_health()
    services_health["redis"] = redis_health
    if redis_health["status"] != "healthy":
        overall_status = "degraded"
    
    # Check Tika
    tika_health = await check_tika_health()
    services_health["tika"] = tika_health
    if tika_health["status"] != "healthy":
        overall_status = "degraded"
    
    total_response_time = time.time() - start_time
    
    # Count healthy vs unhealthy services
    healthy_services = sum(1 for service in services_health.values() if service["status"] == "healthy")
    total_services = len(services_health)
    
    response = {
        "status": overall_status,
        "message": f"{healthy_services}/{total_services} services are healthy",
        "timestamp": time.time(),
        "total_response_time_ms": round(total_response_time * 1000, 2),
        "services": services_health,
        "summary": {
            "healthy_services": healthy_services,
            "total_services": total_services,
            "critical_services_down": total_services - healthy_services
        }
    }
    
    # Return appropriate HTTP status code
    if overall_status == "healthy":
        return response
    else:
        # Return 503 Service Unavailable if any critical service is down
        raise HTTPException(status_code=503, detail=response)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    task = process_document_task.delay(file_location)
    return {"info": f"file '{file.filename}' saved at '{file_location}'. Task ID: {task.id}"}

@app.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.get("/documents/")
def get_all_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    documents = db.query(models.Document).offset(skip).limit(limit).all()
    return documents

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
    if field not in ["extracted_text", "filename", "classification_category", "extracted_entities.text", "key_phrases", "document_sections"]:
        raise HTTPException(status_code=400, detail="Invalid search field")
    
    results = es_client.search_documents(query=query, field=field, size=size)
    return results
