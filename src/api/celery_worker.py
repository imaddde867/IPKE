"""
EXPLAINIUM - Consolidated Celery Worker

Clean, professional task queue implementation for document processing
with proper error handling and monitoring.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from celery import Celery
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.signals import task_prerun, task_postrun, task_failure

# Internal imports
from src.core.config import config
from src.database.database import get_db_session, db_transaction
from src.database import crud, models
from src.processors.optimized_processor import OptimizedDocumentProcessor
from src.logging_config import get_logger

logger = get_logger(__name__)

# Initialize Celery app
celery_app = Celery(
    'explainium',
    broker=config.celery.broker_url,
    backend=config.celery.result_backend
)

# Configure Celery
celery_app.conf.update(
    # Task routing
    task_routes={
        'src.api.celery_worker.process_document_task': {'queue': 'document_processing'},
        'src.api.celery_worker.cleanup_old_tasks': {'queue': 'maintenance'},
    },
    
    # Task execution settings
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=2400,       # 40 minutes hard limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Serialization
    task_serializer=config.celery.task_serializer,
    result_serializer=config.celery.result_serializer,
    accept_content=config.celery.accept_content,
    
    # Timezone
    timezone=config.celery.timezone,
    enable_utc=config.celery.enable_utc,
    
    # Result backend
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-tasks': {
            'task': 'src.api.celery_worker.cleanup_old_tasks',
            'schedule': 3600.0,  # Every hour
        },
    },
)

# Initialize optimized document processor and apply M4 optimizations
document_processor = OptimizedDocumentProcessor()
try:
    document_processor.optimize_for_m4()
except Exception:
    pass


# Celery signal handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run setup"""
    logger.info(f"Starting task {task.name} with ID {task_id}")
    
    # Update task status in database
    try:
        with get_db_session() as db:
            crud.update_task_progress(
                db=db,
                task_id=task_id,
                status="running",
                current_step="Starting task"
            )
    except Exception as e:
        logger.error(f"Failed to update task pre-run status: {e}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task post-run cleanup"""
    logger.info(f"Completed task {task.name} with ID {task_id}, state: {state}")
    
    # Update task status in database
    try:
        with get_db_session() as db:
            crud.update_task_progress(
                db=db,
                task_id=task_id,
                status="completed" if state == "SUCCESS" else "failed",
                progress_percentage=100 if state == "SUCCESS" else None,
                result=retval if state == "SUCCESS" else None
            )
    except Exception as e:
        logger.error(f"Failed to update task post-run status: {e}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failures"""
    logger.error(f"Task {task_id} failed: {exception}")
    
    # Update task status in database
    try:
        with get_db_session() as db:
            crud.update_task_progress(
                db=db,
                task_id=task_id,
                status="failed",
                error_message=str(exception),
                current_step="Task failed"
            )
    except Exception as e:
        logger.error(f"Failed to update task failure status: {e}")


# Main document processing task
@celery_app.task(bind=True, name='process_document_task')
def process_document_task(self, file_path: str, document_id: int) -> Dict[str, Any]:
    """
    Process a document and extract knowledge
    
    Args:
        file_path: Path to the document file
        document_id: Database ID of the document
        
    Returns:
        Dictionary containing processing results
    """
    task_id = self.request.id
    logger.info(f"Processing document {document_id} at {file_path} (Task: {task_id})")
    
    try:
        # Update document status to processing
        with get_db_session() as db:
            crud.update_document_status(
                db=db,
                document_id=document_id,
                status=models.ProcessingStatus.PROCESSING
            )
            
            # Create processing task record
            crud.create_processing_task(
                db=db,
                task_id=task_id,
                task_type="document_processing",
                document_id=document_id,
                total_steps=5
            )
        
        # Update progress: Starting processing
        self.update_state(
            state='PROGRESS',
            meta={'current': 1, 'total': 5, 'status': 'Starting document processing'}
        )
        
        # Process the document using the optimized processor
        logger.info(f"Starting document processing for {file_path}")
        try:
            optimized_result = document_processor.process_document_sync(file_path)
        except Exception as e:
            logger.error(f"Optimized processing failed: {e}")
            raise
        
        # Adapt optimized result to legacy knowledge dict expected by DB layer
        knowledge = {
            'processes': [],
            'decision_points': [],
            'compliance_items': [],
            'risk_assessments': [],
            'entities': [],
            'overall_confidence': optimized_result.confidence_score,
            'processing_metadata': {
                'processing_time': optimized_result.processing_time,
                'processing_method': optimized_result.processing_method,
                'optimization_level': optimized_result.optimization_level,
                'format_detected': optimized_result.format_detected,
                'confidence_score': optimized_result.confidence_score,
            }
        }
        
        for ent in optimized_result.entities:
            try:
                knowledge['entities'].append({
                    'text': ent.get('content') or ent.get('core_content', ''),
                    'label': ent.get('entity_type', ent.get('category', 'unknown')),
                    'confidence': float(ent.get('confidence', ent.get('quality_score', 0.5)))
                })
            except Exception:
                continue
        
        # Build a minimal processing_result dict for downstream references
        processing_result = {
            'knowledge': knowledge,
            'processing_time': optimized_result.processing_time,
            'content': {
                'text': '',
                'page_count': None,
                'language': 'en'
            }
        }
        
        # Update progress: Document processed
        self.update_state(
            state='PROGRESS',
            meta={'current': 2, 'total': 5, 'status': 'Document content extracted'}
        )
        
        # Store extracted knowledge in database
        with db_transaction() as db:
            knowledge = processing_result.get('knowledge', {})
            
            # Update progress: Storing processes
            self.update_state(
                state='PROGRESS',
                meta={'current': 3, 'total': 5, 'status': 'Storing extracted processes'}
            )
            
            # Store processes
            processes_created = 0
            for process_data in knowledge.get('processes', []):
                try:
                    crud.create_process(
                        db=db,
                        document_id=document_id,
                        name=process_data['name'],
                        description=process_data['description'],
                        domain=models.KnowledgeDomain(process_data['domain']),
                        hierarchy_level=models.HierarchyLevel(process_data['hierarchy_level']),
                        criticality=models.CriticalityLevel(process_data['criticality']),
                        confidence=process_data['confidence'],
                        steps=process_data.get('steps', []),
                        prerequisites=process_data.get('prerequisites', []),
                        success_criteria=process_data.get('success_criteria', []),
                        responsible_parties=process_data.get('responsible_parties', []),
                        estimated_duration=process_data.get('duration'),
                        frequency=process_data.get('frequency'),
                        source_text=process_data.get('description', '')[:1000]  # Limit length
                    )
                    processes_created += 1
                except Exception as e:
                    logger.error(f"Failed to create process: {e}")
            
            # Update progress: Storing decision points
            self.update_state(
                state='PROGRESS',
                meta={'current': 4, 'total': 5, 'status': 'Storing decision points'}
            )
            
            # Store decision points
            decisions_created = 0
            for decision_data in knowledge.get('decision_points', []):
                try:
                    crud.create_decision_point(
                        db=db,
                        document_id=document_id,
                        name=decision_data['name'],
                        description=decision_data['description'],
                        decision_type=decision_data.get('decision_type', 'general'),
                        criteria=decision_data.get('criteria', {}),
                        outcomes=decision_data.get('outcomes', []),
                        authority_level=decision_data.get('authority_level', 'general'),
                        confidence=decision_data['confidence'],
                        source_text=decision_data.get('description', '')[:1000]
                    )
                    decisions_created += 1
                except Exception as e:
                    logger.error(f"Failed to create decision point: {e}")
            
            # Store compliance items
            compliance_created = 0
            for compliance_data in knowledge.get('compliance_items', []):
                try:
                    crud.create_compliance_item(
                        db=db,
                        document_id=document_id,
                        regulation_name=compliance_data['regulation_name'],
                        requirement=compliance_data['requirement'],
                        status=models.ComplianceStatus.PENDING,
                        confidence=compliance_data['confidence'],
                        regulation_section=compliance_data.get('section'),
                        responsible_party=compliance_data.get('responsible_party'),
                        source_text=compliance_data.get('requirement', '')[:1000]
                    )
                    compliance_created += 1
                except Exception as e:
                    logger.error(f"Failed to create compliance item: {e}")
            
            # Store risk assessments
            risks_created = 0
            for risk_data in knowledge.get('risk_assessments', []):
                try:
                    crud.create_risk_assessment(
                        db=db,
                        document_id=document_id,
                        hazard=risk_data['hazard'],
                        risk_description=risk_data['risk_description'],
                        likelihood=risk_data.get('likelihood', 'unknown'),
                        impact=risk_data.get('impact', 'unknown'),
                        overall_risk_level=models.RiskLevel(risk_data.get('overall_risk_level', 'medium')),
                        confidence=risk_data['confidence'],
                        mitigation_strategies=risk_data.get('mitigation_strategies', []),
                        source_text=risk_data.get('risk_description', '')[:1000]
                    )
                    risks_created += 1
                except Exception as e:
                    logger.error(f"Failed to create risk assessment: {e}")
            
            # Store knowledge entities
            entities_created = 0
            for entity_data in knowledge.get('entities', []):
                try:
                    crud.create_knowledge_entity(
                        db=db,
                        document_id=document_id,
                        text=entity_data['text'],
                        label=entity_data['label'],
                        confidence=entity_data['confidence'],
                        context=entity_data.get('context', '')[:500]  # Limit context length
                    )
                    entities_created += 1
                except Exception as e:
                    logger.error(f"Failed to create knowledge entity: {e}")
        
        # Update progress: Finalizing
        self.update_state(
            state='PROGRESS',
            meta={'current': 5, 'total': 5, 'status': 'Finalizing processing'}
        )
        
        # Update document status to completed
        with get_db_session() as db:
            document = crud.update_document_status(
                db=db,
                document_id=document_id,
                status=models.ProcessingStatus.COMPLETED
            )
            
            # Update document metadata
            if document:
                text_content = (processing_result.get('content', {}) or {}).get('text', '') or ''
                document.total_pages = (processing_result.get('content', {}) or {}).get('page_count')
                document.total_words = len(text_content.split()) if text_content else 0
                document.total_characters = len(text_content)
                document.language = (processing_result.get('content', {}) or {}).get('language', 'en')
                db.commit()
        
        # Prepare result summary
        result = {
            'document_id': document_id,
            'filename': os.path.basename(file_path),
            'processing_status': 'completed',
            'knowledge_extracted': {
                'processes': processes_created,
                'decision_points': decisions_created,
                'compliance_items': compliance_created,
                'risk_assessments': risks_created,
                'entities': entities_created
            },
            'overall_confidence': knowledge.get('overall_confidence', 0.0),
            'processing_time': processing_result.get('processing_time'),
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed document {document_id}: {result['knowledge_extracted']}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        
        # Update document status to failed
        try:
            with get_db_session() as db:
                crud.update_document_status(
                    db=db,
                    document_id=document_id,
                    status=models.ProcessingStatus.FAILED,
                    error_message=str(e)
                )
        except Exception as db_error:
            logger.error(f"Failed to update document status after error: {db_error}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying document processing for {document_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries), exc=e)
        else:
            logger.error(f"Max retries exceeded for document {document_id}")
            raise e


# Maintenance tasks
@celery_app.task(name='cleanup_old_tasks')
def cleanup_old_tasks() -> Dict[str, Any]:
    """Clean up old completed processing tasks"""
    try:
        with get_db_session() as db:
            deleted_count = crud.cleanup_old_tasks(db, days_old=7)
            
        logger.info(f"Cleaned up {deleted_count} old processing tasks")
        return {
            'status': 'completed',
            'deleted_tasks': deleted_count,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old tasks: {e}")
        raise e


@celery_app.task(name='cleanup_orphaned_records')
def cleanup_orphaned_records() -> Dict[str, Any]:
    """Clean up orphaned database records"""
    try:
        with get_db_session() as db:
            cleanup_counts = crud.cleanup_orphaned_records(db)
            
        logger.info(f"Cleaned up orphaned records: {cleanup_counts}")
        return {
            'status': 'completed',
            'cleanup_counts': cleanup_counts,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup orphaned records: {e}")
        raise e


# Utility functions for API
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a processing task"""
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        # Get task details from database
        with get_db_session() as db:
            task_record = crud.get_processing_task(db, task_id)
        
        # Prepare status response
        status_info = {
            'task_id': task_id,
            'status': task_result.state,
            'current': 0,
            'total': 100,
            'message': 'Unknown status'
        }
        
        if task_result.state == 'PENDING':
            status_info.update({
                'current': 0,
                'total': 100,
                'message': 'Task is waiting to be processed'
            })
        elif task_result.state == 'PROGRESS':
            if task_result.info:
                status_info.update({
                    'current': task_result.info.get('current', 0),
                    'total': task_result.info.get('total', 100),
                    'message': task_result.info.get('status', 'Processing...')
                })
        elif task_result.state == 'SUCCESS':
            status_info.update({
                'current': 100,
                'total': 100,
                'message': 'Task completed successfully',
                'result': task_result.result
            })
        elif task_result.state == 'FAILURE':
            status_info.update({
                'current': 0,
                'total': 100,
                'message': f'Task failed: {str(task_result.info)}',
                'error': str(task_result.info)
            })
        
        # Add database task information if available
        if task_record:
            status_info.update({
                'created_at': task_record.created_at.isoformat() if task_record.created_at else None,
                'started_at': task_record.started_at.isoformat() if task_record.started_at else None,
                'completed_at': task_record.completed_at.isoformat() if task_record.completed_at else None,
                'retry_count': task_record.retry_count,
                'document_id': task_record.document_id
            })
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {
            'task_id': task_id,
            'status': 'UNKNOWN',
            'current': 0,
            'total': 100,
            'message': f'Failed to get task status: {str(e)}',
            'error': str(e)
        }


def get_worker_stats() -> Dict[str, Any]:
    """Get Celery worker statistics"""
    try:
        # Get active tasks
        active_tasks = celery_app.control.inspect().active()
        
        # Get worker stats
        stats = celery_app.control.inspect().stats()
        
        # Get registered tasks
        registered_tasks = celery_app.control.inspect().registered()
        
        return {
            'active_tasks': active_tasks,
            'worker_stats': stats,
            'registered_tasks': registered_tasks,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Health check for Celery
def celery_health_check() -> Dict[str, Any]:
    """Check Celery health status"""
    try:
        # Check if workers are available
        worker_stats = celery_app.control.inspect().stats()
        
        if worker_stats:
            return {
                'status': 'healthy',
                'workers': len(worker_stats),
                'worker_names': list(worker_stats.keys()),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'unhealthy',
                'error': 'No workers available',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }