"""
EXPLAINIUM - Consolidated CRUD Operations

Clean, professional database operations that provide a comprehensive
interface for all database interactions with proper error handling.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import (
    Document, Process, DecisionPoint, ComplianceItem, RiskAssessment,
    KnowledgeEntity, ProcessingTask, SystemMetrics,
    KnowledgeDomain, HierarchyLevel, CriticalityLevel,
    ComplianceStatus, RiskLevel, ProcessingStatus
)

logger = logging.getLogger(__name__)


class CRUDError(Exception):
    """Custom exception for CRUD operations"""
    pass


# Document CRUD Operations
def create_document(
    db: Session,
    filename: str,
    file_path: str,
    file_size: int,
    content_type: str,
    file_hash: Optional[str] = None,
    uploaded_by: Optional[str] = None
) -> Document:
    """Create a new document record"""
    try:
        document = Document(
            filename=filename,
            original_filename=filename,
            file_path=file_path,
            file_size=file_size,
            content_type=content_type,
            file_hash=file_hash,
            uploaded_by=uploaded_by,
            processing_status=ProcessingStatus.PENDING
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Created document: {filename} (ID: {document.id})")
        return document
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create document: {e}")
        raise CRUDError(f"Failed to create document: {str(e)}")


def get_document(db: Session, document_id: int) -> Optional[Document]:
    """Get document by ID"""
    try:
        return db.query(Document).filter(Document.id == document_id).first()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise CRUDError(f"Failed to get document: {str(e)}")


def get_documents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    status: Optional[ProcessingStatus] = None,
    uploaded_by: Optional[str] = None
) -> List[Document]:
    """Get documents with optional filtering"""
    try:
        query = db.query(Document)
        
        if status:
            query = query.filter(Document.processing_status == status)
        if uploaded_by:
            query = query.filter(Document.uploaded_by == uploaded_by)
        
        return query.order_by(desc(Document.uploaded_at)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get documents: {e}")
        raise CRUDError(f"Failed to get documents: {str(e)}")


def update_document_status(
    db: Session,
    document_id: int,
    status: ProcessingStatus,
    error_message: Optional[str] = None
) -> Optional[Document]:
    """Update document processing status"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None
        
        document.processing_status = status
        if status == ProcessingStatus.PROCESSING:
            document.processing_started_at = datetime.now()
        elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            document.processing_completed_at = datetime.now()
        
        if error_message:
            document.processing_error = error_message
        
        db.commit()
        db.refresh(document)
        return document
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to update document status: {e}")
        raise CRUDError(f"Failed to update document status: {str(e)}")


def delete_document(db: Session, document_id: int) -> bool:
    """Delete document and all related records"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False
        
        db.delete(document)
        db.commit()
        logger.info(f"Deleted document: {document_id}")
        return True
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise CRUDError(f"Failed to delete document: {str(e)}")


# Process CRUD Operations
def create_process(
    db: Session,
    document_id: int,
    name: str,
    description: str,
    domain: KnowledgeDomain,
    hierarchy_level: HierarchyLevel,
    criticality: CriticalityLevel,
    confidence: float,
    steps: Optional[List[str]] = None,
    prerequisites: Optional[List[str]] = None,
    success_criteria: Optional[List[str]] = None,
    responsible_parties: Optional[List[str]] = None,
    estimated_duration: Optional[str] = None,
    frequency: Optional[str] = None,
    source_text: Optional[str] = None,
    source_page: Optional[int] = None
) -> Process:
    """Create a new process record"""
    try:
        process = Process(
            document_id=document_id,
            name=name,
            description=description,
            domain=domain,
            hierarchy_level=hierarchy_level,
            criticality=criticality,
            confidence=confidence,
            steps=steps or [],
            prerequisites=prerequisites or [],
            success_criteria=success_criteria or [],
            responsible_parties=responsible_parties or [],
            estimated_duration=estimated_duration,
            frequency=frequency,
            source_text=source_text,
            source_page=source_page
        )
        db.add(process)
        db.commit()
        db.refresh(process)
        logger.info(f"Created process: {name} (ID: {process.id})")
        return process
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create process: {e}")
        raise CRUDError(f"Failed to create process: {str(e)}")


def get_processes(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    domain: Optional[str] = None,
    hierarchy_level: Optional[str] = None,
    confidence_threshold: float = 0.0,
    document_id: Optional[int] = None
) -> List[Process]:
    """Get processes with optional filtering"""
    try:
        query = db.query(Process)
        
        if document_id:
            query = query.filter(Process.document_id == document_id)
        if domain:
            query = query.filter(Process.domain == KnowledgeDomain(domain))
        if hierarchy_level:
            query = query.filter(Process.hierarchy_level == HierarchyLevel(hierarchy_level))
        if confidence_threshold > 0:
            query = query.filter(Process.confidence >= confidence_threshold)
        
        return query.order_by(desc(Process.confidence)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get processes: {e}")
        raise CRUDError(f"Failed to get processes: {str(e)}")


def get_process(db: Session, process_id: int) -> Optional[Process]:
    """Get process by ID"""
    try:
        return db.query(Process).filter(Process.id == process_id).first()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get process {process_id}: {e}")
        raise CRUDError(f"Failed to get process: {str(e)}")


def update_process(
    db: Session,
    process_id: int,
    **kwargs
) -> Optional[Process]:
    """Update process fields"""
    try:
        process = db.query(Process).filter(Process.id == process_id).first()
        if not process:
            return None
        
        for key, value in kwargs.items():
            if hasattr(process, key):
                setattr(process, key, value)
        
        process.last_updated = datetime.now()
        db.commit()
        db.refresh(process)
        return process
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to update process {process_id}: {e}")
        raise CRUDError(f"Failed to update process: {str(e)}")


def delete_process(db: Session, process_id: int) -> bool:
    """Delete process"""
    try:
        process = db.query(Process).filter(Process.id == process_id).first()
        if not process:
            return False
        
        db.delete(process)
        db.commit()
        logger.info(f"Deleted process: {process_id}")
        return True
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to delete process {process_id}: {e}")
        raise CRUDError(f"Failed to delete process: {str(e)}")


# Decision Point CRUD Operations
def create_decision_point(
    db: Session,
    document_id: int,
    name: str,
    description: str,
    decision_type: str,
    criteria: Dict[str, Any],
    outcomes: List[Dict[str, Any]],
    authority_level: str,
    confidence: float,
    escalation_path: Optional[str] = None,
    source_text: Optional[str] = None,
    source_page: Optional[int] = None
) -> DecisionPoint:
    """Create a new decision point record"""
    try:
        decision_point = DecisionPoint(
            document_id=document_id,
            name=name,
            description=description,
            decision_type=decision_type,
            criteria=criteria,
            outcomes=outcomes,
            authority_level=authority_level,
            escalation_path=escalation_path,
            confidence=confidence,
            source_text=source_text,
            source_page=source_page
        )
        db.add(decision_point)
        db.commit()
        db.refresh(decision_point)
        logger.info(f"Created decision point: {name} (ID: {decision_point.id})")
        return decision_point
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create decision point: {e}")
        raise CRUDError(f"Failed to create decision point: {str(e)}")


def get_decision_points(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    document_id: Optional[int] = None,
    decision_type: Optional[str] = None,
    confidence_threshold: float = 0.0
) -> List[DecisionPoint]:
    """Get decision points with optional filtering"""
    try:
        query = db.query(DecisionPoint)
        
        if document_id:
            query = query.filter(DecisionPoint.document_id == document_id)
        if decision_type:
            query = query.filter(DecisionPoint.decision_type == decision_type)
        if confidence_threshold > 0:
            query = query.filter(DecisionPoint.confidence >= confidence_threshold)
        
        return query.order_by(desc(DecisionPoint.confidence)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get decision points: {e}")
        raise CRUDError(f"Failed to get decision points: {str(e)}")


# Compliance Item CRUD Operations
def create_compliance_item(
    db: Session,
    document_id: int,
    regulation_name: str,
    requirement: str,
    status: ComplianceStatus,
    confidence: float,
    regulation_section: Optional[str] = None,
    regulation_authority: Optional[str] = None,
    requirement_type: Optional[str] = None,
    responsible_party: Optional[str] = None,
    review_frequency: Optional[str] = None,
    next_review_date: Optional[datetime] = None,
    source_text: Optional[str] = None,
    source_page: Optional[int] = None
) -> ComplianceItem:
    """Create a new compliance item record"""
    try:
        compliance_item = ComplianceItem(
            document_id=document_id,
            regulation_name=regulation_name,
            regulation_section=regulation_section,
            regulation_authority=regulation_authority,
            requirement=requirement,
            requirement_type=requirement_type,
            status=status,
            responsible_party=responsible_party,
            review_frequency=review_frequency,
            next_review_date=next_review_date,
            confidence=confidence,
            source_text=source_text,
            source_page=source_page
        )
        db.add(compliance_item)
        db.commit()
        db.refresh(compliance_item)
        logger.info(f"Created compliance item: {regulation_name} (ID: {compliance_item.id})")
        return compliance_item
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create compliance item: {e}")
        raise CRUDError(f"Failed to create compliance item: {str(e)}")


def get_compliance_items(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    document_id: Optional[int] = None,
    status: Optional[ComplianceStatus] = None,
    regulation_authority: Optional[str] = None,
    review_due_days: Optional[int] = None
) -> List[ComplianceItem]:
    """Get compliance items with optional filtering"""
    try:
        query = db.query(ComplianceItem)
        
        if document_id:
            query = query.filter(ComplianceItem.document_id == document_id)
        if status:
            query = query.filter(ComplianceItem.status == status)
        if regulation_authority:
            query = query.filter(ComplianceItem.regulation_authority == regulation_authority)
        if review_due_days is not None:
            due_date = datetime.now() + timedelta(days=review_due_days)
            query = query.filter(ComplianceItem.next_review_date <= due_date)
        
        return query.order_by(asc(ComplianceItem.next_review_date)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get compliance items: {e}")
        raise CRUDError(f"Failed to get compliance items: {str(e)}")


def update_compliance_status(
    db: Session,
    compliance_id: int,
    status: ComplianceStatus,
    next_review_date: Optional[datetime] = None
) -> Optional[ComplianceItem]:
    """Update compliance item status"""
    try:
        compliance_item = db.query(ComplianceItem).filter(ComplianceItem.id == compliance_id).first()
        if not compliance_item:
            return None
        
        compliance_item.status = status
        compliance_item.last_review_date = datetime.now()
        if next_review_date:
            compliance_item.next_review_date = next_review_date
        compliance_item.last_updated = datetime.now()
        
        db.commit()
        db.refresh(compliance_item)
        return compliance_item
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to update compliance status: {e}")
        raise CRUDError(f"Failed to update compliance status: {str(e)}")


# Risk Assessment CRUD Operations
def create_risk_assessment(
    db: Session,
    document_id: int,
    hazard: str,
    risk_description: str,
    likelihood: str,
    impact: str,
    overall_risk_level: RiskLevel,
    confidence: float,
    hazard_category: Optional[str] = None,
    mitigation_strategies: Optional[List[str]] = None,
    control_measures: Optional[List[str]] = None,
    monitoring_requirements: Optional[str] = None,
    risk_owner: Optional[str] = None,
    assessor: Optional[str] = None,
    assessment_date: Optional[datetime] = None,
    next_assessment_date: Optional[datetime] = None,
    source_text: Optional[str] = None,
    source_page: Optional[int] = None
) -> RiskAssessment:
    """Create a new risk assessment record"""
    try:
        risk_assessment = RiskAssessment(
            document_id=document_id,
            hazard=hazard,
            hazard_category=hazard_category,
            risk_description=risk_description,
            likelihood=likelihood,
            impact=impact,
            overall_risk_level=overall_risk_level,
            mitigation_strategies=mitigation_strategies or [],
            control_measures=control_measures or [],
            monitoring_requirements=monitoring_requirements,
            risk_owner=risk_owner,
            assessor=assessor,
            confidence=confidence,
            assessment_date=assessment_date or datetime.now(),
            next_assessment_date=next_assessment_date,
            source_text=source_text,
            source_page=source_page
        )
        db.add(risk_assessment)
        db.commit()
        db.refresh(risk_assessment)
        logger.info(f"Created risk assessment: {hazard} (ID: {risk_assessment.id})")
        return risk_assessment
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create risk assessment: {e}")
        raise CRUDError(f"Failed to create risk assessment: {str(e)}")


def get_risk_assessments(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    document_id: Optional[int] = None,
    overall_risk_level: Optional[RiskLevel] = None,
    hazard_category: Optional[str] = None,
    assessment_due_days: Optional[int] = None
) -> List[RiskAssessment]:
    """Get risk assessments with optional filtering"""
    try:
        query = db.query(RiskAssessment)
        
        if document_id:
            query = query.filter(RiskAssessment.document_id == document_id)
        if overall_risk_level:
            query = query.filter(RiskAssessment.overall_risk_level == overall_risk_level)
        if hazard_category:
            query = query.filter(RiskAssessment.hazard_category == hazard_category)
        if assessment_due_days is not None:
            due_date = datetime.now() + timedelta(days=assessment_due_days)
            query = query.filter(RiskAssessment.next_assessment_date <= due_date)
        
        return query.order_by(desc(RiskAssessment.overall_risk_level)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get risk assessments: {e}")
        raise CRUDError(f"Failed to get risk assessments: {str(e)}")


# Knowledge Entity CRUD Operations
def create_knowledge_entity(
    db: Session,
    document_id: int,
    text: str,
    label: str,
    confidence: float,
    start_position: Optional[int] = None,
    end_position: Optional[int] = None,
    context: Optional[str] = None,
    extraction_method: Optional[str] = None
) -> KnowledgeEntity:
    """Create a new knowledge entity record"""
    try:
        entity = KnowledgeEntity(
            document_id=document_id,
            text=text,
            label=label,
            confidence=confidence,
            start_position=start_position,
            end_position=end_position,
            context=context,
            extraction_method=extraction_method
        )
        db.add(entity)
        db.commit()
        db.refresh(entity)
        return entity
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create knowledge entity: {e}")
        raise CRUDError(f"Failed to create knowledge entity: {str(e)}")


def get_knowledge_entities(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    document_id: Optional[int] = None,
    label: Optional[str] = None,
    confidence_threshold: float = 0.0
) -> List[KnowledgeEntity]:
    """Get knowledge entities with optional filtering"""
    try:
        query = db.query(KnowledgeEntity)
        
        if document_id:
            query = query.filter(KnowledgeEntity.document_id == document_id)
        if label:
            query = query.filter(KnowledgeEntity.label == label)
        if confidence_threshold > 0:
            query = query.filter(KnowledgeEntity.confidence >= confidence_threshold)
        
        return query.order_by(desc(KnowledgeEntity.confidence)).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get knowledge entities: {e}")
        raise CRUDError(f"Failed to get knowledge entities: {str(e)}")


# Processing Task CRUD Operations
def create_processing_task(
    db: Session,
    task_id: str,
    task_type: str,
    document_id: Optional[int] = None,
    total_steps: Optional[int] = None
) -> ProcessingTask:
    """Create a new processing task record"""
    try:
        task = ProcessingTask(
            task_id=task_id,
            document_id=document_id,
            task_type=task_type,
            status="pending",
            total_steps=total_steps
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to create processing task: {e}")
        raise CRUDError(f"Failed to create processing task: {str(e)}")


def update_task_progress(
    db: Session,
    task_id: str,
    status: str,
    progress_percentage: Optional[int] = None,
    current_step: Optional[str] = None,
    error_message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None
) -> Optional[ProcessingTask]:
    """Update processing task progress"""
    try:
        task = db.query(ProcessingTask).filter(ProcessingTask.task_id == task_id).first()
        if not task:
            return None
        
        task.status = status
        if progress_percentage is not None:
            task.progress_percentage = progress_percentage
        if current_step:
            task.current_step = current_step
        if error_message:
            task.error_message = error_message
        if result:
            task.result = result
        
        if status == "running" and not task.started_at:
            task.started_at = datetime.now()
        elif status in ["completed", "failed"]:
            task.completed_at = datetime.now()
        
        db.commit()
        db.refresh(task)
        return task
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to update task progress: {e}")
        raise CRUDError(f"Failed to update task progress: {str(e)}")


def get_processing_task(db: Session, task_id: str) -> Optional[ProcessingTask]:
    """Get processing task by task ID"""
    try:
        return db.query(ProcessingTask).filter(ProcessingTask.task_id == task_id).first()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get processing task {task_id}: {e}")
        raise CRUDError(f"Failed to get processing task: {str(e)}")


# Search and Analytics Operations
def search_knowledge(
    db: Session,
    query: str,
    domain: Optional[str] = None,
    hierarchy_level: Optional[str] = None,
    confidence_threshold: float = 0.7,
    max_results: int = 50
) -> List[Dict[str, Any]]:
    """Search across all knowledge types"""
    try:
        results = []
        
        # Search processes
        process_query = db.query(Process)
        if domain:
            process_query = process_query.filter(Process.domain == KnowledgeDomain(domain))
        if hierarchy_level:
            process_query = process_query.filter(Process.hierarchy_level == HierarchyLevel(hierarchy_level))
        
        process_query = process_query.filter(
            and_(
                Process.confidence >= confidence_threshold,
                or_(
                    Process.name.ilike(f"%{query}%"),
                    Process.description.ilike(f"%{query}%")
                )
            )
        )
        
        processes = process_query.limit(max_results // 4).all()
        for process in processes:
            results.append({
                'type': 'process',
                'id': process.id,
                'name': process.name,
                'description': process.description,
                'domain': process.domain.value,
                'hierarchy_level': process.hierarchy_level.value,
                'confidence': process.confidence,
                'document_id': process.document_id
            })
        
        # Search decision points
        decision_query = db.query(DecisionPoint).filter(
            and_(
                DecisionPoint.confidence >= confidence_threshold,
                or_(
                    DecisionPoint.name.ilike(f"%{query}%"),
                    DecisionPoint.description.ilike(f"%{query}%")
                )
            )
        )
        
        decisions = decision_query.limit(max_results // 4).all()
        for decision in decisions:
            results.append({
                'type': 'decision_point',
                'id': decision.id,
                'name': decision.name,
                'description': decision.description,
                'decision_type': decision.decision_type,
                'authority_level': decision.authority_level,
                'confidence': decision.confidence,
                'document_id': decision.document_id
            })
        
        # Search compliance items
        compliance_query = db.query(ComplianceItem).filter(
            and_(
                ComplianceItem.confidence >= confidence_threshold,
                or_(
                    ComplianceItem.regulation_name.ilike(f"%{query}%"),
                    ComplianceItem.requirement.ilike(f"%{query}%")
                )
            )
        )
        
        compliance_items = compliance_query.limit(max_results // 4).all()
        for item in compliance_items:
            results.append({
                'type': 'compliance_item',
                'id': item.id,
                'regulation_name': item.regulation_name,
                'requirement': item.requirement,
                'status': item.status.value,
                'confidence': item.confidence,
                'document_id': item.document_id
            })
        
        # Search risk assessments
        risk_query = db.query(RiskAssessment).filter(
            and_(
                RiskAssessment.confidence >= confidence_threshold,
                or_(
                    RiskAssessment.hazard.ilike(f"%{query}%"),
                    RiskAssessment.risk_description.ilike(f"%{query}%")
                )
            )
        )
        
        risks = risk_query.limit(max_results // 4).all()
        for risk in risks:
            results.append({
                'type': 'risk_assessment',
                'id': risk.id,
                'hazard': risk.hazard,
                'risk_description': risk.risk_description,
                'overall_risk_level': risk.overall_risk_level.value,
                'confidence': risk.confidence,
                'document_id': risk.document_id
            })
        
        # Sort by confidence and limit results
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:max_results]
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to search knowledge: {e}")
        raise CRUDError(f"Failed to search knowledge: {str(e)}")


def get_knowledge_analytics(db: Session) -> Dict[str, Any]:
    """Get knowledge analytics and statistics"""
    try:
        analytics = {}
        
        # Document statistics
        total_documents = db.query(func.count(Document.id)).scalar()
        completed_documents = db.query(func.count(Document.id)).filter(
            Document.processing_status == ProcessingStatus.COMPLETED
        ).scalar()
        
        analytics['documents'] = {
            'total': total_documents,
            'completed': completed_documents,
            'completion_rate': (completed_documents / total_documents * 100) if total_documents > 0 else 0
        }
        
        # Process statistics
        process_stats = db.query(
            Process.domain,
            func.count(Process.id).label('count'),
            func.avg(Process.confidence).label('avg_confidence')
        ).group_by(Process.domain).all()
        
        analytics['processes'] = {
            'total': db.query(func.count(Process.id)).scalar(),
            'by_domain': {stat.domain.value: {'count': stat.count, 'avg_confidence': float(stat.avg_confidence)} 
                         for stat in process_stats}
        }
        
        # Compliance statistics
        compliance_stats = db.query(
            ComplianceItem.status,
            func.count(ComplianceItem.id).label('count')
        ).group_by(ComplianceItem.status).all()
        
        analytics['compliance'] = {
            'total': db.query(func.count(ComplianceItem.id)).scalar(),
            'by_status': {stat.status.value: stat.count for stat in compliance_stats}
        }
        
        # Risk statistics
        risk_stats = db.query(
            RiskAssessment.overall_risk_level,
            func.count(RiskAssessment.id).label('count')
        ).group_by(RiskAssessment.overall_risk_level).all()
        
        analytics['risks'] = {
            'total': db.query(func.count(RiskAssessment.id)).scalar(),
            'by_level': {stat.overall_risk_level.value: stat.count for stat in risk_stats}
        }
        
        return analytics
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to get knowledge analytics: {e}")
        raise CRUDError(f"Failed to get knowledge analytics: {str(e)}")


# Bulk operations
def bulk_create_processes(db: Session, processes: List[Dict[str, Any]]) -> List[Process]:
    """Bulk create processes"""
    try:
        process_objects = []
        for process_data in processes:
            process = Process(**process_data)
            process_objects.append(process)
        
        db.add_all(process_objects)
        db.commit()
        
        for process in process_objects:
            db.refresh(process)
        
        logger.info(f"Bulk created {len(process_objects)} processes")
        return process_objects
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to bulk create processes: {e}")
        raise CRUDError(f"Failed to bulk create processes: {str(e)}")


def bulk_create_knowledge_entities(db: Session, entities: List[Dict[str, Any]]) -> List[KnowledgeEntity]:
    """Bulk create knowledge entities"""
    try:
        entity_objects = []
        for entity_data in entities:
            entity = KnowledgeEntity(**entity_data)
            entity_objects.append(entity)
        
        db.add_all(entity_objects)
        db.commit()
        
        for entity in entity_objects:
            db.refresh(entity)
        
        logger.info(f"Bulk created {len(entity_objects)} knowledge entities")
        return entity_objects
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to bulk create knowledge entities: {e}")
        raise CRUDError(f"Failed to bulk create knowledge entities: {str(e)}")


# Cleanup operations
def cleanup_old_tasks(db: Session, days_old: int = 7) -> int:
    """Clean up old completed processing tasks"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = db.query(ProcessingTask).filter(
            and_(
                ProcessingTask.status.in_(["completed", "failed"]),
                ProcessingTask.completed_at < cutoff_date
            )
        ).delete()
        
        db.commit()
        logger.info(f"Cleaned up {deleted_count} old processing tasks")
        return deleted_count
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to cleanup old tasks: {e}")
        raise CRUDError(f"Failed to cleanup old tasks: {str(e)}")


def cleanup_orphaned_records(db: Session) -> Dict[str, int]:
    """Clean up orphaned records (records without valid document references)"""
    try:
        cleanup_counts = {}
        
        # Find orphaned processes
        orphaned_processes = db.query(Process).filter(
            ~Process.document_id.in_(db.query(Document.id))
        ).delete(synchronize_session=False)
        cleanup_counts['processes'] = orphaned_processes
        
        # Find orphaned decision points
        orphaned_decisions = db.query(DecisionPoint).filter(
            ~DecisionPoint.document_id.in_(db.query(Document.id))
        ).delete(synchronize_session=False)
        cleanup_counts['decision_points'] = orphaned_decisions
        
        # Find orphaned compliance items
        orphaned_compliance = db.query(ComplianceItem).filter(
            ~ComplianceItem.document_id.in_(db.query(Document.id))
        ).delete(synchronize_session=False)
        cleanup_counts['compliance_items'] = orphaned_compliance
        
        # Find orphaned risk assessments
        orphaned_risks = db.query(RiskAssessment).filter(
            ~RiskAssessment.document_id.in_(db.query(Document.id))
        ).delete(synchronize_session=False)
        cleanup_counts['risk_assessments'] = orphaned_risks
        
        # Find orphaned knowledge entities
        orphaned_entities = db.query(KnowledgeEntity).filter(
            ~KnowledgeEntity.document_id.in_(db.query(Document.id))
        ).delete(synchronize_session=False)
        cleanup_counts['knowledge_entities'] = orphaned_entities
        
        db.commit()
        logger.info(f"Cleaned up orphaned records: {cleanup_counts}")
        return cleanup_counts
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Failed to cleanup orphaned records: {e}")
        raise CRUDError(f"Failed to cleanup orphaned records: {str(e)}")