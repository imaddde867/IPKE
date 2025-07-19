from celery import Celery
from src.processors.document_processor import process_document, process_video, get_file_type
from src.database.database import SessionLocal, init_db
from src.database import crud
from src.search.elasticsearch_client import es_client
from src.logging_config import get_logger, set_correlation_id, log_processing_step, log_error
from src.exceptions import DatabaseError, ProcessingError, SearchError
import time

celery_app = Celery(
    'knowledge_extraction',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

# Initialize logging
logger = get_logger(__name__)

# Initialize database tables when Celery worker starts
@celery_app.on_after_configure.connect
def setup_database(sender, **kwargs):
    correlation_id = set_correlation_id()
    logger.info("Celery worker startup initiated", extra={'startup_correlation_id': correlation_id})
    
    try:
        log_processing_step(logger, "celery_database_initialization", "started")
        init_db()
        log_processing_step(logger, "celery_database_initialization", "completed")
        logger.info("Celery worker startup completed successfully")
    except Exception as e:
        log_processing_step(logger, "celery_database_initialization", "failed")
        logger.error(f"Celery worker startup failed: {str(e)}", exc_info=True)
        raise

@celery_app.task
def process_document_task(file_path: str, correlation_id: str = None):
    # Set correlation ID for tracking
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        correlation_id = set_correlation_id()
    
    start_time = time.time()
    logger.info(
        f"Document processing task started: {file_path}",
        extra={'file_path': file_path, 'task_correlation_id': correlation_id}
    )
    
    try:
        file_type = get_file_type(file_path)
        log_processing_step(logger, "file_type_detection", "completed", extra_data={'file_type': file_type})
    except Exception as e:
        raise ProcessingError(
            f"Failed to determine file type for {file_path}",
            file_path=file_path,
            processing_stage="file_type_detection"
        ) from e
    
    db = SessionLocal()
    try:
        # Process document based on type
        log_processing_step(logger, "document_processing", "started", extra_data={'file_type': file_type})
        processing_start = time.time()
        
        if file_type == "video":
            result = process_video(file_path)
        else:
            result = process_document(file_path)
        
        processing_duration = time.time() - processing_start
        log_processing_step(
            logger, 
            "document_processing", 
            "completed", 
            duration=processing_duration,
            extra_data={'file_type': file_type, 'status': result.get('status')}
        )

        # Save processed data to database
        log_processing_step(logger, "database_save", "started")
        db_save_start = time.time()
        
        try:
            db_document = crud.create_document(
                db=db,
                filename=result["filename"],
                file_type=file_type,
                extracted_text=result["extracted_text"],
                metadata_json=result["metadata"],
                classification_category=result["classification"]["category"],
                classification_score=result["classification"]["score"],
                status=result["status"],
                document_sections=result["document_sections"]
            )
            
            db_save_duration = time.time() - db_save_start
            log_processing_step(
                logger, 
                "database_save", 
                "completed", 
                duration=db_save_duration,
                extra_data={'document_id': db_document.id}
            )
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to save document to database: {file_path}",
                operation="create_document",
                table="documents",
                details={'file_path': file_path, 'filename': result.get("filename")}
            ) from e

        # Save extracted entities
        es_entities = []
        for entity in result["extracted_entities"]:
            crud.create_extracted_entity(
                db=db,
                document_id=db_document.id,
                text=entity["word"],
                entity_type=entity["entity_group"],
                score=entity["score"],
                start_char=entity["start"],
                end_char=entity["end"]
            )
            es_entities.append({"text": entity["word"], "entity_type": entity["entity_group"]})
        
        # Save key phrases
        es_key_phrases = []
        for phrase in result["key_phrases"]:
            crud.create_key_phrase(
                db=db,
                document_id=db_document.id,
                phrase=phrase
            )
            es_key_phrases.append(phrase)

        # Save structured data
        for equipment in result["equipment_data"]:
            crud.create_equipment(
                db=db,
                document_id=db_document.id,
                name=equipment["name"],
                type=equipment["type"],
                specifications=equipment["specifications"],
                location=equipment["location"],
                confidence=equipment["confidence"]
            )
        
        for procedure in result["procedure_data"]:
            crud.create_procedure(
                db=db,
                document_id=db_document.id,
                title=procedure["title"],
                steps=procedure["steps"],
                category=procedure["category"],
                confidence=procedure["confidence"]
            )

        for safety_info in result["safety_info_data"]:
            crud.create_safety_information(
                db=db,
                document_id=db_document.id,
                hazard=safety_info["hazard"],
                precaution=safety_info["precaution"],
                ppe_required=safety_info["ppe_required"],
                severity=safety_info["severity"],
                confidence=safety_info["confidence"]
            )

        for tech_spec in result["technical_spec_data"]:
            crud.create_technical_specification(
                db=db,
                document_id=db_document.id,
                parameter=tech_spec["parameter"],
                value=tech_spec["value"],
                unit=tech_spec["unit"],
                tolerance=tech_spec["tolerance"],
                confidence=tech_spec["confidence"]
            )

        for personnel in result["personnel_data"]:
            crud.create_personnel(
                db=db,
                document_id=db_document.id,
                name=personnel["name"],
                role=personnel["role"],
                responsibilities=personnel["responsibilities"],
                certifications=personnel["certifications"],
                confidence=personnel["confidence"]
            )

        # Index document in Elasticsearch
        log_processing_step(logger, "elasticsearch_indexing", "started")
        es_index_start = time.time()
        
        try:
            es_document_data = {
                "document_id": db_document.id,
                "filename": db_document.filename,
                "file_type": db_document.file_type,
                "extracted_text": db_document.extracted_text,
                "classification_category": db_document.classification_category,
                "classification_score": db_document.classification_score,
                "extracted_entities": es_entities,
                "key_phrases": es_key_phrases,
                "processing_timestamp": db_document.processing_timestamp.isoformat(),
                "document_sections": db_document.document_sections
            }
            es_client.index_document(es_document_data)
            
            es_index_duration = time.time() - es_index_start
            log_processing_step(
                logger, 
                "elasticsearch_indexing", 
                "completed", 
                duration=es_index_duration,
                extra_data={'document_id': db_document.id}
            )
            
        except Exception as e:
            # Log error but don't fail the entire task
            log_error(
                logger,
                SearchError(
                    f"Failed to index document in Elasticsearch: {db_document.id}",
                    operation="index_document",
                    details={'document_id': db_document.id, 'filename': db_document.filename}
                ),
                "Elasticsearch indexing failed - document saved to database but not searchable"
            )

        total_duration = time.time() - start_time
        logger.info(
            f"Document processing completed successfully: {file_path}",
            extra={
                'file_path': file_path,
                'document_id': db_document.id,
                'total_duration_seconds': total_duration,
                'status': 'success'
            }
        )

        return {
            "status": "success", 
            "document_id": db_document.id,
            "correlation_id": correlation_id,
            "processing_duration": total_duration
        }

    except Exception as e:
        db.rollback()
        total_duration = time.time() - start_time
        
        # Log the error with full context
        log_error(
            logger,
            e,
            f"Document processing failed: {file_path}",
            extra_data={
                'file_path': file_path,
                'file_type': file_type,
                'total_duration_seconds': total_duration,
                'status': 'failed'
            }
        )
        
        # Return structured error response
        return {
            "status": "failed", 
            "error": str(e),
            "error_type": e.__class__.__name__,
            "correlation_id": correlation_id,
            "processing_duration": total_duration
        }
    finally:
        db.close()
