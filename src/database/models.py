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
    status = Column(String)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    document_sections = Column(JSON) # New column for extracted sections

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
    entity_type = Column(String) # e.g., PER, ORG, LOC, or custom types like EQUIPMENT, PROCEDURE
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
    type = Column(String) # e.g., Pump, Motor, Valve, Sensor
    specifications = Column(JSON) # e.g., {"power": "10HP", "voltage": "480V"}
    location = Column(String, nullable=True)
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="equipment")

class Procedure(Base):
    __tablename__ = "procedures"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    title = Column(String, index=True)
    steps = Column(JSON) # e.g., [{"step_number": 1, "description": "..."}]
    category = Column(String, nullable=True) # e.g., Startup, Shutdown, Maintenance
    confidence = Column(Float, nullable=True) # New column for confidence

    document = relationship("Document", back_populates="procedures")

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