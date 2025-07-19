from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String)
    extracted_text = Column(Text)
    metadata_json = Column(JSON)  # Store Tika metadata as JSON
    classification_category = Column(String)
    classification_score = Column(Float)
    status = Column(String, index=True)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    document_sections = Column(JSON) # New column for extracted sections
    
    # Enhanced processing tracking fields
    processing_duration_seconds = Column(Float)
    retry_count = Column(Integer, default=0, index=True)
    last_error = Column(Text)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)

    entities = relationship("ExtractedEntity", back_populates="document")
    key_phrases = relationship("KeyPhrase", back_populates="document")
    equipment = relationship("Equipment", back_populates="document")
    procedures = relationship("Procedure", back_populates="document")
    safety_info = relationship("SafetyInformation", back_populates="document")
    technical_specs = relationship("TechnicalSpecification", back_populates="document")
    personnel = relationship("Personnel", back_populates="document")

class ExtractedEntity(Base):
    __tablename__ = "extracted_entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(String)
    entity_type = Column(String) # for example PER, ORG, LOC, or custom types like EQUIPMENT, PROCEDURE
    score = Column(Float)
    start_char = Column(Integer)
    end_char = Column(Integer)

    document = relationship("Document", back_populates="entities")

class KeyPhrase(Base):
    __tablename__ = "key_phrases"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    phrase = Column(String)

    document = relationship("Document", back_populates="key_phrases")

class Equipment(Base):
    __tablename__ = "equipment"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    name = Column(String, index=True)
    type = Column(String) #  for example = Pump, Motor, Valve, Sensor
    specifications = Column(JSON) # for example = {"power": "10HP", "voltage": "480V"}
    location = Column(String, nullable=True)
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="equipment")
    # List of EquipmentPersonnel associations for this equipment
    equipment_personnels = relationship("EquipmentPersonnel", back_populates="equipment", cascade="all, delete-orphan")
    # List of ProcedureEquipment associations for this equipment
    procedure_equipments = relationship("ProcedureEquipment", back_populates="equipment", cascade="all, delete-orphan")



class Step(Base):
    __tablename__ = "steps"
    id = Column(Integer, primary_key=True, index=True)
    procedure_id = Column(Integer, ForeignKey("procedures.id"))
    step_number = Column(Integer)
    description = Column(Text)
    expected_result = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)

    procedure = relationship("Procedure", back_populates="steps")

class Procedure(Base):
    __tablename__ = "procedures"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    title = Column(String, index=True)
    steps = relationship("Step", back_populates="procedure", cascade="all, delete-orphan")
    category = Column(String, nullable=True) # e.g., Startup, Shutdown, Maintenance
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="procedures")
    # List of ProcedureEquipment associations for this procedure
    procedure_equipments = relationship("ProcedureEquipment", back_populates="procedure", cascade="all, delete-orphan")
    # List of ProcedurePersonnel associations for this procedure
    procedure_personnels = relationship("ProcedurePersonnel", back_populates="procedure", cascade="all, delete-orphan")

class SafetyInformation(Base):
    __tablename__ = "safety_information"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    hazard = Column(String)
    precaution = Column(String)
    ppe_required = Column(String) # e.g., "Gloves, Hard Hat"
    severity = Column(String, nullable=True) # e.g., High, Medium, Low
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="safety_info")

class TechnicalSpecification(Base):
    __tablename__ = "technical_specifications"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    parameter = Column(String)
    value = Column(String)
    unit = Column(String)
    tolerance = Column(String, nullable=True)
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="technical_specs")

class Personnel(Base):
    __tablename__ = "personnel"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    name = Column(String, index=True)
    role = Column(String)
    responsibilities = Column(Text, nullable=True)
    certifications = Column(JSON) # e.g., ["OSHA 30", "First Aid"]
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="personnel")
    # List of EquipmentPersonnel associations for this personnel
    equipment_personnels = relationship("EquipmentPersonnel", back_populates="personnel", cascade="all, delete-orphan")
    # List of ProcedurePersonnel associations for this personnel
    procedure_personnels = relationship("ProcedurePersonnel", back_populates="personnel", cascade="all, delete-orphan")

class ProcedureEquipment(Base):
    __tablename__ = "procedure_equipment"
    id = Column(Integer, primary_key=True, index=True)
    procedure_id = Column(Integer, ForeignKey("procedures.id"))  # Link to Procedure
    equipment_id = Column(Integer, ForeignKey("equipment.id"))   # Link to Equipment

    # Relationship backrefs for easy access
    procedure = relationship("Procedure", back_populates="procedure_equipments")
    equipment = relationship("Equipment", back_populates="procedure_equipments")

class ProcedurePersonnel(Base):
    __tablename__ = "procedure_personnel"
    id = Column(Integer, primary_key=True, index=True)
    procedure_id = Column(Integer, ForeignKey("procedures.id"))  # Link to Procedure
    personnel_id = Column(Integer, ForeignKey("personnel.id"))   # Link to Personnel

    # Relationship backrefs for easy access
    procedure = relationship("Procedure", back_populates="procedure_personnels")
    personnel = relationship("Personnel", back_populates="procedure_personnels")

class EquipmentPersonnel(Base):
    __tablename__ = "equipment_personnel"
    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"))   # Link to Equipment
    personnel_id = Column(Integer, ForeignKey("personnel.id"))   # Link to Personnel

    # Relationship backrefs for easy access
    equipment = relationship("Equipment", back_populates="equipment_personnels")
    personnel = relationship("Personnel", back_populates="equipment_personnels")

class ProcessingLog(Base):
    """Track processing steps and errors for documents."""
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    step_name = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # started, completed, failed
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    error_message = Column(Text)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("Document")

class SystemHealth(Base):
    """Store system health check results and metrics."""
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # healthy, unhealthy, degraded
    response_time_ms = Column(Float)
    error_message = Column(Text)
    metadata_json = Column(JSON)
    checked_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)