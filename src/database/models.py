from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON, Boolean, Enum as SQLEnum, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from enum import Enum
import uuid

Base = declarative_base()

# Enums for standardized classifications
class KnowledgeDomain(Enum):
    OPERATIONAL = "operational"
    SAFETY_COMPLIANCE = "safety_compliance"
    EQUIPMENT_TECHNOLOGY = "equipment_technology"
    HUMAN_RESOURCES = "human_resources"
    QUALITY_ASSURANCE = "quality_assurance"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    MAINTENANCE = "maintenance"
    TRAINING = "training"

class HierarchyLevel(Enum):
    CORE_BUSINESS_FUNCTION = 1
    DEPARTMENT_OPERATION = 2
    INDIVIDUAL_PROCEDURE = 3
    SPECIFIC_STEP = 4

class CriticalityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

class RiskLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

# Association tables for many-to-many relationships
process_equipment_table = Table(
    'process_equipment_association',
    Base.metadata,
    Column('process_id', Integer, ForeignKey('processes.id')),
    Column('equipment_id', Integer, ForeignKey('equipment.id'))
)

process_personnel_table = Table(
    'process_personnel_association',
    Base.metadata,
    Column('process_id', Integer, ForeignKey('processes.id')),
    Column('personnel_id', Integer, ForeignKey('personnel.id'))
)

process_safety_table = Table(
    'process_safety_association',
    Base.metadata,
    Column('process_id', Integer, ForeignKey('processes.id')),
    Column('safety_info_id', Integer, ForeignKey('safety_information.id'))
)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String)
    extracted_text = Column(Text)
    metadata_json = Column(JSON)
    classification_category = Column(String)
    classification_score = Column(Float)
    status = Column(String, index=True)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    document_sections = Column(JSON)
    
    # Enhanced processing tracking fields
    processing_duration_seconds = Column(Float)
    retry_count = Column(Integer, default=0, index=True)
    last_error = Column(Text)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    
    # New fields for enhanced knowledge capture
    source_quality_score = Column(Float)
    content_completeness_score = Column(Float)
    knowledge_domains = Column(JSON)  # List of applicable domains
    regulatory_references = Column(JSON)  # List of regulatory standards mentioned
    version_info = Column(JSON)  # Document version tracking

    # Relationships
    entities = relationship("ExtractedEntity", back_populates="document", cascade="all, delete-orphan")
    key_phrases = relationship("KeyPhrase", back_populates="document", cascade="all, delete-orphan")
    equipment = relationship("Equipment", back_populates="document", cascade="all, delete-orphan")
    procedures = relationship("Procedure", back_populates="document", cascade="all, delete-orphan")
    safety_info = relationship("SafetyInformation", back_populates="document", cascade="all, delete-orphan")
    technical_specs = relationship("TechnicalSpecification", back_populates="document", cascade="all, delete-orphan")
    personnel = relationship("Personnel", back_populates="document", cascade="all, delete-orphan")
    processes = relationship("Process", back_populates="document", cascade="all, delete-orphan")
    compliance_items = relationship("ComplianceItem", back_populates="document", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="document", cascade="all, delete-orphan")

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
    type = Column(String)  # Pump, Motor, Valve, Sensor
    specifications = Column(JSON)  # {"power": "10HP", "voltage": "480V"}
    location = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Enhanced equipment fields
    manufacturer = Column(String, nullable=True)
    model_number = Column(String, nullable=True)
    serial_number = Column(String, nullable=True)
    installation_date = Column(DateTime, nullable=True)
    maintenance_schedule = Column(JSON, nullable=True)  # Maintenance requirements
    operational_parameters = Column(JSON, nullable=True)  # Operating ranges
    safety_requirements = Column(JSON, nullable=True)  # Safety protocols
    criticality_level = Column(SQLEnum(CriticalityLevel), default=CriticalityLevel.MEDIUM)

    document = relationship("Document", back_populates="equipment")
    # List of EquipmentPersonnel associations for this equipment
    equipment_personnels = relationship("EquipmentPersonnel", back_populates="equipment", cascade="all, delete-orphan")
    # List of ProcedureEquipment associations for this equipment
    procedure_equipments = relationship("ProcedureEquipment", back_populates="equipment", cascade="all, delete-orphan")
    # Many-to-many relationship with processes
    processes = relationship("Process", secondary=process_equipment_table, back_populates="equipment_used")



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
    ppe_required = Column(String)  # "Gloves, Hard Hat"
    severity = Column(SQLEnum(RiskLevel), default=RiskLevel.MEDIUM)
    confidence = Column(Float, nullable=True)
    
    # Enhanced safety fields
    hazard_category = Column(String, nullable=True)  # chemical, physical, biological, etc.
    affected_areas = Column(JSON, nullable=True)  # Areas where hazard applies
    emergency_procedures = Column(Text, nullable=True)
    regulatory_references = Column(JSON, nullable=True)  # OSHA, ISO standards
    training_requirements = Column(JSON, nullable=True)
    inspection_frequency = Column(String, nullable=True)
    responsible_party = Column(String, nullable=True)

    document = relationship("Document", back_populates="safety_info")
    # Many-to-many relationship with processes
    processes = relationship("Process", secondary=process_safety_table, back_populates="safety_requirements")

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
    certifications = Column(JSON)  # ["OSHA 30", "First Aid"]
    confidence = Column(Float, nullable=True)
    
    # Enhanced personnel fields
    department = Column(String, nullable=True)
    supervisor = Column(String, nullable=True)
    contact_information = Column(JSON, nullable=True)
    skill_level = Column(String, nullable=True)  # novice, intermediate, expert
    training_records = Column(JSON, nullable=True)
    authorization_levels = Column(JSON, nullable=True)  # What they're authorized to do
    shift_schedule = Column(JSON, nullable=True)
    emergency_contact = Column(JSON, nullable=True)

    document = relationship("Document", back_populates="personnel")
    # List of EquipmentPersonnel associations for this personnel
    equipment_personnels = relationship("EquipmentPersonnel", back_populates="personnel", cascade="all, delete-orphan")
    # List of ProcedurePersonnel associations for this personnel
    procedure_personnels = relationship("ProcedurePersonnel", back_populates="personnel", cascade="all, delete-orphan")
    # Many-to-many relationship with processes
    processes = relationship("Process", secondary=process_personnel_table, back_populates="personnel_involved")

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

class Process(Base):
    """Enhanced process model representing organizational workflows and procedures"""
    __tablename__ = "processes"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    process_id = Column(String, unique=True, index=True)  # Unique identifier for the process
    name = Column(String, index=True)
    description = Column(Text)
    knowledge_domain = Column(SQLEnum(KnowledgeDomain))
    hierarchy_level = Column(SQLEnum(HierarchyLevel))
    parent_process_id = Column(String, ForeignKey("processes.process_id"))
    
    # Operational details
    estimated_duration = Column(String)  # e.g., "2 hours", "30 minutes"
    frequency = Column(String)  # e.g., "daily", "weekly", "as needed"
    timing_constraints = Column(JSON)  # Start/end times, dependencies
    prerequisites = Column(JSON)  # Required conditions before starting
    success_criteria = Column(JSON)  # How to measure success
    
    # Resource requirements
    required_skills = Column(JSON)  # Skills needed to perform
    required_certifications = Column(JSON)  # Certifications needed
    required_tools = Column(JSON)  # Tools and equipment needed
    
    # Quality and compliance
    quality_standards = Column(JSON)  # Quality requirements
    compliance_requirements = Column(JSON)  # Regulatory requirements
    criticality_level = Column(SQLEnum(CriticalityLevel))
    
    # Knowledge capture metadata
    confidence_score = Column(Float)
    source_quality = Column(Float)
    completeness_index = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)
    
    # Relationships
    document = relationship("Document", back_populates="processes")
    parent_process = relationship("Process", remote_side="Process.process_id", backref="child_processes")
    steps = relationship("ProcessStep", back_populates="process", cascade="all, delete-orphan")
    decision_points = relationship("DecisionPoint", back_populates="process", cascade="all, delete-orphan")
    
    # Many-to-many relationships
    equipment_used = relationship("Equipment", secondary=process_equipment_table, back_populates="processes")
    personnel_involved = relationship("Personnel", secondary=process_personnel_table, back_populates="processes")
    safety_requirements = relationship("SafetyInformation", secondary=process_safety_table, back_populates="processes")

class ProcessStep(Base):
    """Individual steps within a process"""
    __tablename__ = "process_steps"

    id = Column(Integer, primary_key=True, index=True)
    process_id = Column(Integer, ForeignKey("processes.id"))
    step_number = Column(Integer)
    name = Column(String)
    description = Column(Text)
    expected_duration = Column(String)
    expected_result = Column(Text)
    verification_method = Column(String)  # How to verify completion
    safety_notes = Column(Text)
    quality_checkpoints = Column(JSON)
    confidence = Column(Float)
    
    # Dependencies and conditions
    depends_on_steps = Column(JSON)  # List of step IDs this depends on
    conditions = Column(JSON)  # Conditions that must be met
    
    process = relationship("Process", back_populates="steps")

class DecisionPoint(Base):
    """Decision points within processes"""
    __tablename__ = "decision_points"

    id = Column(Integer, primary_key=True, index=True)
    process_id = Column(Integer, ForeignKey("processes.id"))
    name = Column(String)
    description = Column(Text)
    decision_type = Column(String)  # binary, multiple_choice, conditional, threshold
    decision_criteria = Column(JSON)  # Criteria for making the decision
    possible_outcomes = Column(JSON)  # Possible outcomes and their consequences
    authority_level = Column(String)  # Who can make this decision
    escalation_path = Column(JSON)  # Escalation if decision can't be made
    confidence = Column(Float)
    
    process = relationship("Process", back_populates="decision_points")

class ComplianceItem(Base):
    """Compliance requirements and tracking"""
    __tablename__ = "compliance_items"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    regulation_name = Column(String)
    regulation_section = Column(String)
    requirement_description = Column(Text)
    compliance_status = Column(SQLEnum(ComplianceStatus))
    last_review_date = Column(DateTime)
    next_review_date = Column(DateTime)
    responsible_party = Column(String)
    evidence_location = Column(String)  # Where compliance evidence is stored
    notes = Column(Text)
    confidence = Column(Float)
    
    document = relationship("Document", back_populates="compliance_items")

class RiskAssessment(Base):
    """Risk assessments and mitigation strategies"""
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    risk_category = Column(String)  # safety, operational, financial, regulatory
    risk_description = Column(Text)
    likelihood = Column(SQLEnum(RiskLevel))
    impact = Column(SQLEnum(RiskLevel))
    overall_risk_level = Column(SQLEnum(RiskLevel))
    mitigation_strategies = Column(JSON)
    monitoring_requirements = Column(JSON)
    responsible_party = Column(String)
    review_frequency = Column(String)
    last_assessment_date = Column(DateTime)
    confidence = Column(Float)
    
    document = relationship("Document", back_populates="risk_assessments")