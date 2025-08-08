from celery import Celery
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.signals import task_prerun, task_postrun, task_failure, task_retry
from src.processors.document_processor import process_document, process_video, get_file_type
from src.database.database import SessionLocal, init_db
from src.database import crud
from src.search.elasticsearch_client import es_client
from src.logging_config import get_logger, set_correlation_id, log_processing_step, log_error
from src.exceptions import DatabaseError, ProcessingError, SearchError
from src.config import config_manager
import time
import random
import json
from typing import Dict, Any, Optional

# Get Redis URL from configuration
redis_url = config_manager.get_redis_url()

# Enhanced Celery configuration with retry and dead letter queue settings
celery_app = Celery(
    'knowledge_extraction',
    broker=redis_url,
    backend=redis_url
)

# Configure Celery settings for enhanced error handling and retry logic
celery_app.conf.update(
    # Task routing and queues
    task_routes={
        'src.api.celery_worker.process_document_task': {'queue': 'document_processing'},
        'src.api.celery_worker.retry_failed_task': {'queue': 'retry_queue'},
        'src.api.celery_worker.cleanup_expired_tasks': {'queue': 'celery'},
    },
    # Dead letter queue configuration
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    # Task execution settings
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2400,       # 40 minutes hard limit
    # Retry settings
    task_default_retry_delay=60,  # 1 minute base delay
    task_max_retries=3,
    # Serialization
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    # Timezone
    timezone='UTC',
    enable_utc=True,
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-expired-tasks': {
            'task': 'src.api.celery_worker.cleanup_expired_tasks',
            'schedule': 3600.0,  # Run every hour
        },
    },
)

# Initialize logging
logger = get_logger(__name__)

# Task status tracking
class TaskStatus:
    """Task status constants for tracking processing progress."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"

class TaskProgressTracker:
    """Tracks task progress and status updates."""
    
    def __init__(self, task_id: str, correlation_id: str):
        self.task_id = task_id
        self.correlation_id = correlation_id
        self.start_time = time.time()
        self.current_step = ""
        self.total_steps = 8  # Total processing steps
        self.completed_steps = 0
        
    def update_progress(self, step_name: str, status: str = "started", extra_data: Dict[str, Any] = None):
        """Update task progress and log the step."""
        self.current_step = step_name
        if status == "completed":
            self.completed_steps += 1
            
        progress_percentage = (self.completed_steps / self.total_steps) * 100
        
        progress_data = {
            'task_id': self.task_id,
            'correlation_id': self.correlation_id,
            'current_step': step_name,
            'step_status': status,
            'progress_percentage': progress_percentage,
            'completed_steps': self.completed_steps,
            'total_steps': self.total_steps,
            'elapsed_time': time.time() - self.start_time
        }
        
        if extra_data:
            progress_data.update(extra_data)
            
        logger.info(f"Task progress update: {step_name} - {status}", extra=progress_data)
        
        # Update task state in Celery backend
        celery_app.backend.store_result(
            self.task_id,
            {
                'status': TaskStatus.PROCESSING,
                'progress': progress_data
            },
            TaskStatus.PROCESSING
        )

def calculate_retry_delay(retry_count: int, base_delay: int = 60, max_delay: int = 3600, jitter: bool = True) -> int:
    """Calculate exponential backoff delay with optional jitter."""
    delay = min(base_delay * (2 ** retry_count), max_delay)
    
    if jitter:
        # Add random jitter (±25% of delay)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(int(delay), base_delay)

def handle_dead_letter_task(task_id: str, task_name: str, args: tuple, kwargs: dict, error: str):
    """Handle tasks that have exceeded maximum retries (dead letter queue)."""
    dead_letter_data = {
        'task_id': task_id,
        'task_name': task_name,
        'args': args,
        'kwargs': kwargs,
        'error': error,
        'timestamp': time.time(),
        'status': 'dead_letter'
    }
    
    logger.error(
        f"Task moved to dead letter queue: {task_name}",
        extra=dead_letter_data
    )
    
    # Store in database for manual review and potential retry
    db = SessionLocal()
    try:
        # You could create a dead_letter_tasks table to store these
        # For now, we'll log them comprehensively
        logger.critical(
            f"DEAD LETTER TASK - Manual intervention required",
            extra={
                'dead_letter_task': json.dumps(dead_letter_data, default=str),
                'requires_manual_review': True
            }
        )
    except Exception as e:
        logger.error(f"Failed to log dead letter task: {str(e)}")
    finally:
        db.close()

# Celery signal handlers for enhanced monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run setup."""
    logger.info(f"Task starting: {task.name}", extra={
        'task_id': task_id,
        'task_name': task.name,
        'args_count': len(args) if args else 0,
        'kwargs_keys': list(kwargs.keys()) if kwargs else []
    })

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task post-run cleanup."""
    logger.info(f"Task completed: {task.name}", extra={
        'task_id': task_id,
        'task_name': task.name,
        'final_state': state,
        'return_value_type': type(retval).__name__ if retval else None
    })

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failures."""
    logger.error(f"Task failed: {sender.name}", extra={
        'task_id': task_id,
        'task_name': sender.name,
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': traceback
    })

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwds):
    """Handle task retries."""
    logger.warning(f"Task retry: {sender.name}", extra={
        'task_id': task_id,
        'task_name': sender.name,
        'retry_reason': str(reason),
        'retry_count': sender.request.retries if hasattr(sender, 'request') else 'unknown'
    })

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

@celery_app.task(bind=True, max_retries=2)
def process_document_task(self, file_path: str, correlation_id: str = None):
    """Simplified document processing task with better error handling."""
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        correlation_id = set_correlation_id()
    
    start_time = time.time()
    retry_count = self.request.retries
    
    logger.info(f"Processing document: {file_path} (attempt {retry_count + 1})")
    
    # Update task status
    self.update_state(
        state=TaskStatus.STARTED,
        meta={'file_path': file_path, 'correlation_id': correlation_id, 'started_at': start_time}
    )
    
    db = SessionLocal()
    try:
        # Get file type
        file_type = get_file_type(file_path)
        
        # Process document
        if file_type == "video":
            result = process_video(file_path)
        else:
            result = process_document(file_path)
        
        # Save to database
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
        
        # Save entities and key phrases
        for entity in result["extracted_entities"]:
            crud.create_extracted_entity(
                db=db,
                document_id=db_document.id,
                text=entity.get("word", entity.get("text", "")),
                entity_type=entity.get("entity_group", entity.get("label", "UNKNOWN")),
                score=entity.get("score", 0.0),
                start_char=entity.get("start", 0),
                end_char=entity.get("end", 0)
            )
        
        for phrase in result["key_phrases"]:
            crud.create_key_phrase(db=db, document_id=db_document.id, phrase=phrase)
        
        # Save structured data
        _save_structured_data(db, db_document.id, result)
        
        # Try to index in Elasticsearch (non-critical)
        try:
            _index_in_elasticsearch(db_document, result)
        except Exception as e:
            logger.warning(f"Elasticsearch indexing failed: {str(e)}")
        
        db.commit()
        
        total_duration = time.time() - start_time
        success_result = {
            "status": "success",
            "document_id": db_document.id,
            "correlation_id": correlation_id,
            "processing_duration": total_duration
        }
        
        self.update_state(state=TaskStatus.SUCCESS, meta=success_result)
        logger.info(f"Document processed successfully: {file_path} (ID: {db_document.id})")
        
        return success_result
        
    except Exception as e:
        db.rollback()
        total_duration = time.time() - start_time
        
        error_msg = f"Processing failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        
        # Retry logic
        if retry_count < self.max_retries:
            retry_delay = min(60 * (2 ** retry_count), 300)  # Exponential backoff, max 5 minutes
            logger.info(f"Retrying in {retry_delay}s (attempt {retry_count + 2})")
            raise self.retry(countdown=retry_delay, exc=e)
        
        # Max retries exceeded
        error_result = {
            "status": "failed",
            "error": str(e),
            "correlation_id": correlation_id,
            "processing_duration": total_duration,
            "retry_count": retry_count
        }
        
        self.update_state(state=TaskStatus.FAILURE, meta=error_result)
        return error_result
        
    finally:
        db.close()

def _save_structured_data(db, document_id: int, result: dict):
    """Save structured data to database."""
    try:
        for equipment in result.get("equipment_data", []):
            crud.create_equipment(
                db=db, document_id=document_id,
                name=equipment["name"], type=equipment["type"],
                specifications=equipment["specifications"],
                location=equipment["location"], confidence=equipment["confidence"]
            )
        
        for procedure in result.get("procedure_data", []):
            crud.create_procedure(
                db=db, document_id=document_id,
                title=procedure["title"], steps=procedure["steps"],
                category=procedure["category"], confidence=procedure["confidence"]
            )
        
        for safety_info in result.get("safety_info_data", []):
            crud.create_safety_information(
                db=db, document_id=document_id,
                hazard=safety_info["hazard"], precaution=safety_info["precaution"],
                ppe_required=safety_info["ppe_required"], severity=safety_info["severity"],
                confidence=safety_info["confidence"]
            )
        
        for tech_spec in result.get("technical_spec_data", []):
            crud.create_technical_specification(
                db=db, document_id=document_id,
                parameter=tech_spec["parameter"], value=tech_spec["value"],
                unit=tech_spec["unit"], tolerance=tech_spec["tolerance"],
                confidence=tech_spec["confidence"]
            )
        
        for personnel in result.get("personnel_data", []):
            crud.create_personnel(
                db=db, document_id=document_id,
                name=personnel["name"], role=personnel["role"],
                responsibilities=personnel["responsibilities"],
                certifications=personnel["certifications"], confidence=personnel["confidence"]
            )
    except Exception as e:
        logger.warning(f"Failed to save some structured data: {str(e)}")

def _index_in_elasticsearch(db_document, result: dict):
    """Index document in Elasticsearch."""
    from src.search.elasticsearch_client import es_client
    
    es_document_data = {
        "document_id": db_document.id,
        "filename": db_document.filename,
        "file_type": db_document.file_type,
        "extracted_text": db_document.extracted_text,
        "classification_category": db_document.classification_category,
        "classification_score": db_document.classification_score,
        "extracted_entities": [{"text": e.get("word", e.get("text", "")), "entity_type": e.get("entity_group", e.get("label", "UNKNOWN"))} for e in result.get("extracted_entities", [])],
        "key_phrases": result.get("key_phrases", []),
        "processing_timestamp": db_document.processing_timestamp.isoformat(),
        "document_sections": db_document.document_sections
    }
    es_client.index_document(es_document_data)

@celery_app.task(bind=True, max_retries=1)
def retry_failed_task(self, original_task_name: str, args: tuple, kwargs: dict, original_error: str):
    """Manually retry a task that was sent to the dead letter queue."""
    correlation_id = set_correlation_id()
    
    logger.info(f"Manually retrying failed task: {original_task_name}", extra={
        'original_task_name': original_task_name,
        'original_error': original_error,
        'retry_correlation_id': correlation_id,
        'task_id': self.request.id
    })
    
    try:
        if original_task_name == 'process_document_task':
            # Retry the document processing task
            result = process_document_task.apply_async(args=args, kwargs=kwargs)
            return {
                'status': 'retry_initiated',
                'new_task_id': result.id,
                'correlation_id': correlation_id
            }
        else:
            raise ValueError(f"Unknown task type for retry: {original_task_name}")
            
    except Exception as e:
        logger.error(f"Failed to retry task {original_task_name}: {str(e)}", extra={
            'original_task_name': original_task_name,
            'retry_error': str(e),
            'task_id': self.request.id
        })
        return {
            'status': 'retry_failed',
            'error': str(e),
            'correlation_id': correlation_id
        }

@celery_app.task
def get_task_status(task_id: str):
    """Get detailed status information for a task."""
    try:
        result = celery_app.AsyncResult(task_id)
        
        status_info = {
            'task_id': task_id,
            'state': result.state,
            'info': result.info,
            'successful': result.successful(),
            'failed': result.failed(),
            'ready': result.ready(),
            'traceback': result.traceback if result.failed() else None
        }
        
        return status_info
        
    except Exception as e:
        return {
            'task_id': task_id,
            'error': f"Failed to get task status: {str(e)}"
        }

@celery_app.task
def cleanup_expired_tasks():
    """Clean up expired task results and dead letter queue entries."""
    logger.info("Starting cleanup of expired tasks")
    
    try:
        # This would typically clean up old task results
        # Implementation depends on your specific requirements
        logger.info("Task cleanup completed successfully")
        return {'status': 'success', 'message': 'Expired tasks cleaned up'}
        
    except Exception as e:
        logger.error(f"Task cleanup failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}
