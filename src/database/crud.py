from sqlalchemy.orm import Session
from src.database.models import Document, ExtractedEntity, KeyPhrase, Equipment, Procedure, SafetyInformation, TechnicalSpecification, Personnel
from datetime import datetime

def create_document(db: Session, filename: str, file_type: str, extracted_text: str, metadata_json: dict, classification_category: str, classification_score: float, status: str, document_sections: dict):
    db_document = Document(
        filename=filename,
        file_type=file_type,
        extracted_text=extracted_text,
        metadata_json=metadata_json,
        classification_category=classification_category,
        classification_score=classification_score,
        status=status,
        processing_timestamp=datetime.utcnow(),
        document_sections=document_sections
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def create_extracted_entity(db: Session, document_id: int, text: str, entity_type: str, score: float, start_char: int, end_char: int):
    db_entity = ExtractedEntity(
        document_id=document_id,
        text=text,
        entity_type=entity_type,
        score=score,
        start_char=start_char,
        end_char=end_char
    )
    db.add(db_entity)
    db.commit()
    db.refresh(db_entity)
    return db_entity

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
