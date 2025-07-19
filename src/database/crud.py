from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, func, text
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union, Tuple
from src.database.models import Document, ExtractedEntity, KeyPhrase, Equipment, Procedure, SafetyInformation, TechnicalSpecification, Personnel
from src.database.database import get_db_session
from src.exceptions import DatabaseError, ValidationError
from src.logging_config import get_logger, log_error
from datetime import datetime
import time

logger = get_logger(__name__)

class TransactionManager:
    """Context manager for database transactions with proper error handling."""
    
    def __init__(self, db: Session):
        self.db = db
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug("Starting database transaction")
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            logger.error(f"Transaction failed after {duration:.3f}s, rolling back: {exc_val}")
            try:
                self.db.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            return False
        else:
            try:
                self.db.commit()
                logger.debug(f"Transaction committed successfully in {duration:.3f}s")
            except Exception as commit_error:
                logger.error(f"Commit failed after {duration:.3f}s: {commit_error}")
                try:
                    self.db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Rollback after commit failure failed: {rollback_error}")
                raise DatabaseError(
                    "Transaction commit failed",
                    operation="commit_transaction",
                    details={"commit_error": str(commit_error)}
                )
        return True

@contextmanager
def transaction_scope(db: Session):
    """Context manager for database transactions."""
    with TransactionManager(db):
        yield db

class BulkOperations:
    """Utility class for bulk database operations."""
    
    @staticmethod
    def bulk_insert_entities(db: Session, document_id: int, entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Bulk insert extracted entities for a document."""
        if not entities:
            return []
        
        try:
            with transaction_scope(db):
                db_entities = []
                for entity_data in entities:
                    # Validate entity data
                    if not entity_data.get('text') or not entity_data.get('entity_type'):
                        logger.warning(f"Skipping invalid entity data: {entity_data}")
                        continue
                    
                    db_entity = ExtractedEntity(
                        document_id=document_id,
                        text=entity_data['text'].strip(),
                        entity_type=entity_data['entity_type'].strip(),
                        score=entity_data.get('score', 0.0),
                        start_char=entity_data.get('start_char', 0),
                        end_char=entity_data.get('end_char', 0)
                    )
                    db_entities.append(db_entity)
                
                if db_entities:
                    db.add_all(db_entities)
                    db.flush()  # Flush to get IDs without committing
                    
                    logger.info(f"Bulk inserted {len(db_entities)} entities for document {document_id}")
                
                return db_entities
                
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Bulk insert entities failed for document {document_id}",
                operation="bulk_insert_entities",
                table="extracted_entities",
                details={'document_id': document_id, 'entity_count': len(entities)}
            ) from e
    
    @staticmethod
    def bulk_insert_keyphrases(db: Session, document_id: int, phrases: List[str]) -> List[KeyPhrase]:
        """Bulk insert keyphrases for a document."""
        if not phrases:
            return []
        
        try:
            with transaction_scope(db):
                db_phrases = []
                for phrase in phrases:
                    if phrase and phrase.strip():
                        db_phrase = KeyPhrase(
                            document_id=document_id,
                            phrase=phrase.strip()
                        )
                        db_phrases.append(db_phrase)
                
                if db_phrases:
                    db.add_all(db_phrases)
                    db.flush()
                    
                    logger.info(f"Bulk inserted {len(db_phrases)} keyphrases for document {document_id}")
                
                return db_phrases
                
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Bulk insert keyphrases failed for document {document_id}",
                operation="bulk_insert_keyphrases",
                table="key_phrases",
                details={'document_id': document_id, 'phrase_count': len(phrases)}
            ) from e
    
    @staticmethod
    def bulk_insert_equipment(db: Session, document_id: int, equipment_list: List[Dict[str, Any]]) -> List[Equipment]:
        """Bulk insert equipment for a document."""
        if not equipment_list:
            return []
        
        try:
            with transaction_scope(db):
                db_equipment = []
                for equipment_data in equipment_list:
                    if not equipment_data.get('name') or not equipment_data.get('type'):
                        logger.warning(f"Skipping invalid equipment data: {equipment_data}")
                        continue
                    
                    db_equip = Equipment(
                        document_id=document_id,
                        name=equipment_data['name'],
                        type=equipment_data['type'],
                        specifications=equipment_data.get('specifications', {}),
                        location=equipment_data.get('location'),
                        confidence=equipment_data.get('confidence')
                    )
                    db_equipment.append(db_equip)
                
                if db_equipment:
                    db.add_all(db_equipment)
                    db.flush()
                    
                    logger.info(f"Bulk inserted {len(db_equipment)} equipment items for document {document_id}")
                
                return db_equipment
                
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Bulk insert equipment failed for document {document_id}",
                operation="bulk_insert_equipment",
                table="equipment",
                details={'document_id': document_id, 'equipment_count': len(equipment_list)}
            ) from e

class PaginatedQuery:
    """Utility class for paginated database queries."""
    
    @staticmethod
    def paginate(query, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """Apply pagination to a query and return results with metadata."""
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 20
        
        total = query.count()
        items = query.offset((page - 1) * per_page).limit(per_page).all()
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page,
            'has_prev': page > 1,
            'has_next': page * per_page < total
        }

def create_document(db: Session, filename: str, file_type: str, extracted_text: str, metadata_json: dict, classification_category: str, classification_score: float, status: str, document_sections: dict):
    """Create a new document with proper transaction management."""
    # Validate inputs
    if not filename or not filename.strip():
        raise ValidationError("Filename cannot be empty", field="filename")
    
    if not file_type or not file_type.strip():
        raise ValidationError("File type cannot be empty", field="file_type")
    
    if classification_score < 0 or classification_score > 1:
        raise ValidationError(
            "Classification score must be between 0 and 1",
            field="classification_score",
            value=classification_score
        )
    
    try:
        with transaction_scope(db):
            db_document = Document(
                filename=filename,
                file_type=file_type,
                extracted_text=extracted_text or "",
                metadata_json=metadata_json or {},
                classification_category=classification_category or "unclassified",
                classification_score=classification_score,
                status=status,
                processing_timestamp=datetime.utcnow(),
                document_sections=document_sections or {},
                processing_started_at=datetime.utcnow() if status == "processing" else None
            )
            
            db.add(db_document)
            db.flush()  # Get the ID without committing
            
            logger.info(
                f"Document created successfully: {filename}",
                extra={'document_id': db_document.id, 'filename': filename, 'file_type': file_type}
            )
            
            return db_document
        
    except IntegrityError as e:
        raise DatabaseError(
            f"Database integrity error creating document: {filename}",
            operation="create_document",
            table="documents",
            details={'filename': filename, 'integrity_error': str(e)}
        ) from e
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating document: {filename}",
            operation="create_document",
            table="documents",
            details={'filename': filename}
        ) from e

def create_document_with_entities(
    db: Session, 
    filename: str, 
    file_type: str, 
    extracted_text: str, 
    metadata_json: dict, 
    classification_category: str, 
    classification_score: float, 
    status: str, 
    document_sections: dict,
    entities: List[Dict[str, Any]] = None,
    keyphrases: List[str] = None,
    equipment_list: List[Dict[str, Any]] = None
) -> Document:
    """Create a document with all related entities in a single transaction."""
    try:
        with transaction_scope(db):
            # Create the document
            db_document = Document(
                filename=filename,
                file_type=file_type,
                extracted_text=extracted_text or "",
                metadata_json=metadata_json or {},
                classification_category=classification_category or "unclassified",
                classification_score=classification_score,
                status=status,
                processing_timestamp=datetime.utcnow(),
                document_sections=document_sections or {},
                processing_started_at=datetime.utcnow() if status == "processing" else None
            )
            
            db.add(db_document)
            db.flush()  # Get the document ID
            
            # Bulk insert related entities
            if entities:
                BulkOperations.bulk_insert_entities(db, db_document.id, entities)
            
            if keyphrases:
                BulkOperations.bulk_insert_keyphrases(db, db_document.id, keyphrases)
            
            if equipment_list:
                BulkOperations.bulk_insert_equipment(db, db_document.id, equipment_list)
            
            logger.info(
                f"Document with entities created successfully: {filename}",
                extra={
                    'document_id': db_document.id, 
                    'filename': filename, 
                    'entity_count': len(entities) if entities else 0,
                    'keyphrase_count': len(keyphrases) if keyphrases else 0,
                    'equipment_count': len(equipment_list) if equipment_list else 0
                }
            )
            
            return db_document
            
    except Exception as e:
        logger.error(f"Failed to create document with entities: {filename}, error: {e}")
        raise

def create_extracted_entity(db: Session, document_id: int, text: str, entity_type: str, score: float, start_char: int, end_char: int):
    """Create a single extracted entity with proper transaction management."""
    # Validate inputs
    if not text or not text.strip():
        raise ValidationError("Entity text cannot be empty", field="text")
    
    if not entity_type or not entity_type.strip():
        raise ValidationError("Entity type cannot be empty", field="entity_type")
    
    if score < 0 or score > 1:
        raise ValidationError(
            "Entity score must be between 0 and 1",
            field="score",
            value=score
        )
    
    if start_char < 0 or end_char < 0 or start_char >= end_char:
        raise ValidationError(
            "Invalid character positions",
            field="character_positions",
            details={'start_char': start_char, 'end_char': end_char}
        )
    
    try:
        with transaction_scope(db):
            db_entity = ExtractedEntity(
                document_id=document_id,
                text=text.strip(),
                entity_type=entity_type.strip(),
                score=score,
                start_char=start_char,
                end_char=end_char
            )
            
            db.add(db_entity)
            db.flush()
            return db_entity
        
    except IntegrityError as e:
        raise DatabaseError(
            f"Database integrity error creating entity for document {document_id}",
            operation="create_extracted_entity",
            table="extracted_entities",
            details={'document_id': document_id, 'text': text, 'integrity_error': str(e)}
        ) from e
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating entity for document {document_id}",
            operation="create_extracted_entity",
            table="extracted_entities",
            details={'document_id': document_id, 'text': text}
        ) from e

def create_key_phrase(db: Session, document_id: int, phrase: str):
    """Create a single keyphrase with transaction management."""
    try:
        with transaction_scope(db):
            db_key_phrase = KeyPhrase(
                document_id=document_id,
                phrase=phrase
            )
            db.add(db_key_phrase)
            db.flush()
            return db_key_phrase
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating keyphrase for document {document_id}",
            operation="create_key_phrase",
            table="key_phrases"
        ) from e

def create_equipment(db: Session, document_id: int, name: str, type: str, specifications: dict, location: str = None, confidence: float = None):
    """Create equipment with transaction management."""
    try:
        with transaction_scope(db):
            db_equipment = Equipment(
                document_id=document_id,
                name=name,
                type=type,
                specifications=specifications,
                location=location,
                confidence=confidence
            )
            db.add(db_equipment)
            db.flush()
            return db_equipment
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating equipment for document {document_id}",
            operation="create_equipment",
            table="equipment"
        ) from e

def create_procedure(db: Session, document_id: int, title: str, steps: list, category: str = None, confidence: float = None):
    """Create procedure with transaction management."""
    try:
        with transaction_scope(db):
            db_procedure = Procedure(
                document_id=document_id,
                title=title,
                steps=steps,
                category=category,
                confidence=confidence
            )
            db.add(db_procedure)
            db.flush()
            return db_procedure
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating procedure for document {document_id}",
            operation="create_procedure",
            table="procedures"
        ) from e

def create_safety_information(db: Session, document_id: int, hazard: str, precaution: str, ppe_required: str, severity: str = None, confidence: float = None):
    """Create safety information with transaction management."""
    try:
        with transaction_scope(db):
            db_safety_info = SafetyInformation(
                document_id=document_id,
                hazard=hazard,
                precaution=precaution,
                ppe_required=ppe_required,
                severity=severity,
                confidence=confidence
            )
            db.add(db_safety_info)
            db.flush()
            return db_safety_info
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating safety information for document {document_id}",
            operation="create_safety_information",
            table="safety_information"
        ) from e

def create_technical_specification(db: Session, document_id: int, parameter: str, value: str, unit: str = None, tolerance: str = None, confidence: float = None):
    """Create technical specification with transaction management."""
    try:
        with transaction_scope(db):
            db_tech_spec = TechnicalSpecification(
                document_id=document_id,
                parameter=parameter,
                value=value,
                unit=unit,
                tolerance=tolerance,
                confidence=confidence
            )
            db.add(db_tech_spec)
            db.flush()
            return db_tech_spec
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating technical specification for document {document_id}",
            operation="create_technical_specification",
            table="technical_specifications"
        ) from e

def create_personnel(db: Session, document_id: int, name: str, role: str, responsibilities: str = None, certifications: list = None, confidence: float = None):
    """Create personnel with transaction management."""
    try:
        with transaction_scope(db):
            db_personnel = Personnel(
                document_id=document_id,
                name=name,
                role=role,
                responsibilities=responsibilities,
                certifications=certifications,
                confidence=confidence
            )
            db.add(db_personnel)
            db.flush()
            return db_personnel
    except SQLAlchemyError as e:
        raise DatabaseError(
            f"Database error creating personnel for document {document_id}",
            operation="create_personnel",
            table="personnel"
        ) from e

# Enhanced query functions with pagination and filtering

def get_documents(db: Session, page: int = 1, per_page: int = 20, status: str = None, file_type: str = None) -> Dict[str, Any]:
    """Get documents with pagination and optional filtering."""
    query = db.query(Document)
    
    if status:
        query = query.filter(Document.status == status)
    if file_type:
        query = query.filter(Document.file_type == file_type)
    
    query = query.order_by(Document.processing_timestamp.desc())
    
    return PaginatedQuery.paginate(query, page, per_page)

def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
    """Get a document by ID with all related entities."""
    return db.query(Document).filter(Document.id == document_id).first()

def update_document_status(db: Session, document_id: int, status: str, error_message: str = None) -> bool:
    """Update document status with transaction management."""
    try:
        with transaction_scope(db):
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            document.status = status
            if error_message:
                document.last_error = error_message
            
            if status == "processing":
                document.processing_started_at = datetime.utcnow()
            elif status in ["processed", "failed"]:
                document.processing_completed_at = datetime.utcnow()
                if document.processing_started_at:
                    duration = (document.processing_completed_at - document.processing_started_at).total_seconds()
                    document.processing_duration_seconds = duration
            
            db.flush()
            return True
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to update document status: {e}")
        return False

def increment_document_retry_count(db: Session, document_id: int) -> bool:
    """Increment document retry count."""
    try:
        with transaction_scope(db):
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            document.retry_count = (document.retry_count or 0) + 1
            db.flush()
            return True
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to increment retry count: {e}")
        return False

def get_processing_statistics(db: Session) -> Dict[str, Any]:
    """Get processing statistics across all documents."""
    try:
        stats = {}
        
        # Document counts by status
        status_counts = db.query(
            Document.status, 
            func.count(Document.id)
        ).group_by(Document.status).all()
        
        stats['status_counts'] = {status: count for status, count in status_counts}
        
        # Average processing time
        avg_duration = db.query(
            func.avg(Document.processing_duration_seconds)
        ).filter(Document.processing_duration_seconds.isnot(None)).scalar()
        
        stats['average_processing_time_seconds'] = float(avg_duration) if avg_duration else 0
        
        # Documents with retries
        retry_count = db.query(func.count(Document.id)).filter(Document.retry_count > 0).scalar()
        stats['documents_with_retries'] = retry_count
        
        # Total entities extracted
        entity_count = db.query(func.count(ExtractedEntity.id)).scalar()
        stats['total_entities_extracted'] = entity_count
        
        return stats
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to get processing statistics: {e}")
        return {"error": str(e)}

def delete_document_cascade(db: Session, document_id: int) -> bool:
    """Delete a document and all related entities in a single transaction."""
    try:
        with transaction_scope(db):
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            # Delete related entities (handled by cascade in model relationships)
            db.delete(document)
            db.flush()
            
            logger.info(f"Document {document_id} and all related entities deleted successfully")
            return True
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise DatabaseError(
            f"Failed to delete document {document_id}",
            operation="delete_document_cascade",
            table="documents"
        ) from e
