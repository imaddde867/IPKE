from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from src.database.models import Document, ExtractedEntity, KeyPhrase, Equipment, Procedure, SafetyInformation, TechnicalSpecification, Personnel
from src.exceptions import DatabaseError, ValidationError
from src.logging_config import get_logger, log_error
from datetime import datetime

logger = get_logger(__name__)

def create_document(db: Session, filename: str, file_type: str, extracted_text: str, metadata_json: dict, classification_category: str, classification_score: float, status: str, document_sections: dict):
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
        db_document = Document(
            filename=filename,
            file_type=file_type,
            extracted_text=extracted_text or "",
            metadata_json=metadata_json or {},
            classification_category=classification_category or "unclassified",
            classification_score=classification_score,
            status=status,
            processing_timestamp=datetime.utcnow(),
            document_sections=document_sections or {}
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        logger.info(
            f"Document created successfully: {filename}",
            extra={'document_id': db_document.id, 'filename': filename, 'file_type': file_type}
        )
        
        return db_document
        
    except IntegrityError as e:
        db.rollback()
        raise DatabaseError(
            f"Database integrity error creating document: {filename}",
            operation="create_document",
            table="documents",
            details={'filename': filename, 'integrity_error': str(e)}
        ) from e
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError(
            f"Database error creating document: {filename}",
            operation="create_document",
            table="documents",
            details={'filename': filename}
        ) from e

def create_extracted_entity(db: Session, document_id: int, text: str, entity_type: str, score: float, start_char: int, end_char: int):
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
        db_entity = ExtractedEntity(
            document_id=document_id,
            text=text.strip(),
            entity_type=entity_type.strip(),
            score=score,
            start_char=start_char,
            end_char=end_char
        )
        
        db.add(db_entity)
        db.commit()
        db.refresh(db_entity)
        return db_entity
        
    except IntegrityError as e:
        db.rollback()
        raise DatabaseError(
            f"Database integrity error creating entity for document {document_id}",
            operation="create_extracted_entity",
            table="extracted_entities",
            details={'document_id': document_id, 'text': text, 'integrity_error': str(e)}
        ) from e
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError(
            f"Database error creating entity for document {document_id}",
            operation="create_extracted_entity",
            table="extracted_entities",
            details={'document_id': document_id, 'text': text}
        ) from e

def create_key_phrase(db: Session, document_id: int, phrase: str):
    db_key_phrase = KeyPhrase(
        document_id=document_id,
        phrase=phrase
    )
    db.add(db_key_phrase)
    db.commit()
    db.refresh(db_key_phrase)
    return db_key_phrase

def create_equipment(db: Session, document_id: int, name: str, type: str, specifications: dict, location: str = None, confidence: float = None):
    db_equipment = Equipment(
        document_id=document_id,
        name=name,
        type=type,
        specifications=specifications,
        location=location,
        confidence=confidence
    )
    db.add(db_equipment)
    db.commit()
    db.refresh(db_equipment)
    return db_equipment

def create_procedure(db: Session, document_id: int, title: str, steps: list, category: str = None, confidence: float = None):
    db_procedure = Procedure(
        document_id=document_id,
        title=title,
        steps=steps,
        category=category,
        confidence=confidence
    )
    db.add(db_procedure)
    db.commit()
    db.refresh(db_procedure)
    return db_procedure

def create_safety_information(db: Session, document_id: int, hazard: str, precaution: str, ppe_required: str, severity: str = None, confidence: float = None):
    db_safety_info = SafetyInformation(
        document_id=document_id,
        hazard=hazard,
        precaution=precaution,
        ppe_required=ppe_required,
        severity=severity,
        confidence=confidence
    )
    db.add(db_safety_info)
    db.commit()
    db.refresh(db_safety_info)
    return db_safety_info

def create_technical_specification(db: Session, document_id: int, parameter: str, value: str, unit: str = None, tolerance: str = None, confidence: float = None):
    db_tech_spec = TechnicalSpecification(
        document_id=document_id,
        parameter=parameter,
        value=value,
        unit=unit,
        tolerance=tolerance,
        confidence=confidence
    )
    db.add(db_tech_spec)
    db.commit()
    db.refresh(db_tech_spec)
    return db_tech_spec

def create_personnel(db: Session, document_id: int, name: str, role: str, responsibilities: str = None, certifications: list = None, confidence: float = None):
    db_personnel = Personnel(
        document_id=document_id,
        name=name,
        role=role,
        responsibilities=responsibilities,
        certifications=certifications,
        confidence=confidence
    )
    db.add(db_personnel)
    db.commit()
    db.refresh(db_personnel)
    return db_personnel
