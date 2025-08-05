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


# Enhanced Knowledge Extraction Models

class KnowledgeItem(Base):
    """Core knowledge items extracted from enterprise documents."""
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True, index=True)
    process_id = Column(String(100), unique=True, index=True, nullable=False)  # Hierarchical ID
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    knowledge_type = Column(String(50), index=True, nullable=False)  # tacit, explicit, procedural
    domain = Column(String(50), index=True, nullable=False)  # operational, safety, equipment, etc.
    hierarchy_level = Column(Integer, index=True, nullable=False)  # 1-4 process hierarchy
    confidence_score = Column(Float, nullable=False, default=0.0)
    source_quality = Column(String(20), nullable=False, default='medium')  # high, medium, low
    completeness_index = Column(Float, nullable=False, default=0.0)  # 0.0-1.0
    criticality_level = Column(String(20), nullable=False, default='medium')  # critical, high, medium, low
    access_level = Column(String(20), nullable=False, default='internal')  # public, internal, restricted, confidential
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    # Relationships
    source_document = relationship("Document")
    workflow_dependencies_source = relationship("WorkflowDependency", foreign_keys="WorkflowDependency.source_process_id", back_populates="source_process")
    workflow_dependencies_target = relationship("WorkflowDependency", foreign_keys="WorkflowDependency.target_process_id", back_populates="target_process")
    decision_trees = relationship("DecisionTree", back_populates="knowledge_item", cascade="all, delete-orphan")
    knowledge_relationships_source = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.source_id", back_populates="source_knowledge")
    knowledge_relationships_target = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.target_id", back_populates="target_knowledge")


class WorkflowDependency(Base):
    """Dependencies between workflow processes."""
    __tablename__ = "workflow_dependencies"
    
    id = Column(Integer, primary_key=True, index=True)
    source_process_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    target_process_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    dependency_type = Column(String(50), nullable=False, index=True)  # prerequisite, parallel, downstream, conditional
    strength = Column(Float, nullable=False, default=0.5)  # dependency strength 0.0-1.0
    conditions = Column(JSON)  # conditional dependencies and requirements
    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    source_process = relationship("KnowledgeItem", foreign_keys=[source_process_id], back_populates="workflow_dependencies_source")
    target_process = relationship("KnowledgeItem", foreign_keys=[target_process_id], back_populates="workflow_dependencies_target")


class DecisionTree(Base):
    """Decision trees and decision points within processes."""
    __tablename__ = "decision_trees"
    
    id = Column(Integer, primary_key=True, index=True)
    process_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    decision_point = Column(String(255), nullable=False)
    decision_type = Column(String(50), nullable=False, index=True)  # binary, multiple_choice, conditional, threshold
    conditions = Column(JSON, nullable=False)  # decision conditions and criteria
    outcomes = Column(JSON, nullable=False)  # possible outcomes and their consequences
    confidence = Column(Float, nullable=False, default=0.0)
    priority = Column(String(20), nullable=False, default='medium')  # critical, high, medium, low
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    knowledge_item = relationship("KnowledgeItem", back_populates="decision_trees")


class OptimizationPattern(Base):
    """Patterns for process optimization and efficiency improvements."""
    __tablename__ = "optimization_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)  # resource, time, quality, cost, safety
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    domain = Column(String(50), nullable=False, index=True)  # operational, safety, equipment, etc.
    conditions = Column(JSON, nullable=False)  # conditions where pattern applies
    improvements = Column(JSON, nullable=False)  # expected improvements and metrics
    success_metrics = Column(JSON, nullable=False)  # KPIs and measurement criteria
    confidence = Column(Float, nullable=False, default=0.0)
    impact_level = Column(String(20), nullable=False, default='medium')  # high, medium, low
    implementation_complexity = Column(String(20), nullable=False, default='medium')  # high, medium, low
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    pattern_applications = relationship("OptimizationPatternApplication", back_populates="pattern", cascade="all, delete-orphan")


class OptimizationPatternApplication(Base):
    """Applications of optimization patterns to specific processes."""
    __tablename__ = "optimization_pattern_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_id = Column(Integer, ForeignKey("optimization_patterns.id"), nullable=False, index=True)
    process_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    applicability_score = Column(Float, nullable=False, default=0.0)  # 0.0-1.0
    expected_impact = Column(JSON)  # expected impact metrics
    implementation_notes = Column(Text)
    status = Column(String(20), nullable=False, default='identified')  # identified, planned, implemented, validated
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    pattern = relationship("OptimizationPattern", back_populates="pattern_applications")


class CommunicationFlow(Base):
    """Communication flows and information exchange patterns."""
    __tablename__ = "communication_flows"
    
    id = Column(Integer, primary_key=True, index=True)
    source_role = Column(String(100), nullable=False, index=True)
    target_role = Column(String(100), nullable=False, index=True)
    information_type = Column(String(100), nullable=False, index=True)
    communication_method = Column(String(50), nullable=False)  # verbal, written, digital, visual
    frequency = Column(String(50), nullable=False)  # continuous, daily, weekly, monthly, as_needed
    criticality = Column(String(20), nullable=False, default='medium')  # critical, high, medium, low
    formal_protocol = Column(Boolean, nullable=False, default=False)
    process_context = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class KnowledgeRelationship(Base):
    """Relationships between different knowledge items."""
    __tablename__ = "knowledge_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    target_id = Column(String(100), ForeignKey("knowledge_items.process_id"), nullable=False, index=True)
    relationship_type = Column(String(50), nullable=False, index=True)  # depends_on, enables, conflicts_with, enhances, requires
    strength = Column(Float, nullable=False, default=0.5)  # relationship strength 0.0-1.0
    bidirectional = Column(Boolean, nullable=False, default=False)
    relationship_metadata = Column(JSON)  # additional relationship metadata
    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    source_knowledge = relationship("KnowledgeItem", foreign_keys=[source_id], back_populates="knowledge_relationships_source")
    target_knowledge = relationship("KnowledgeItem", foreign_keys=[target_id], back_populates="knowledge_relationships_target")


class KnowledgeGap(Base):
    """Identified gaps in organizational knowledge."""
    __tablename__ = "knowledge_gaps"
    
    id = Column(Integer, primary_key=True, index=True)
    gap_type = Column(String(50), nullable=False, index=True)  # missing_documentation, inconsistent_info, outdated, incomplete_process
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    domain = Column(String(50), nullable=False, index=True)
    affected_processes = Column(JSON)  # list of affected process IDs
    impact_assessment = Column(JSON)  # risk and impact analysis
    priority = Column(String(20), nullable=False, default='medium', index=True)  # critical, high, medium, low
    status = Column(String(20), nullable=False, default='identified', index=True)  # identified, assigned, in_progress, resolved, deferred
    assigned_to = Column(String(100))  # person or team responsible
    due_date = Column(DateTime)
    resolution_notes = Column(Text)
    identified_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)
    
    # Relationships
    gap_evidence = relationship("KnowledgeGapEvidence", back_populates="knowledge_gap", cascade="all, delete-orphan")


class KnowledgeGapEvidence(Base):
    """Evidence supporting identified knowledge gaps."""
    __tablename__ = "knowledge_gap_evidence"
    
    id = Column(Integer, primary_key=True, index=True)
    gap_id = Column(Integer, ForeignKey("knowledge_gaps.id"), nullable=False, index=True)
    evidence_type = Column(String(50), nullable=False)  # missing_reference, conflicting_info, outdated_timestamp, incomplete_data
    description = Column(Text, nullable=False)
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    knowledge_gap = relationship("KnowledgeGap", back_populates="gap_evidence")
    source_document = relationship("Document")