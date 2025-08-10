"""
EXPLAINIUM - Consolidated Database Models

Clean, professional database models that consolidate all data structures
into a single, well-organized module with proper relationships and constraints.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, JSON, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


# Enums for database fields
class KnowledgeDomain(PyEnum):
    """Knowledge domains for classification"""
    OPERATIONAL = "operational"
    SAFETY_COMPLIANCE = "safety_compliance"
    EQUIPMENT_TECHNOLOGY = "equipment_technology"
    HUMAN_RESOURCES = "human_resources"
    QUALITY_ASSURANCE = "quality_assurance"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    REGULATORY = "regulatory"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"


class HierarchyLevel(PyEnum):
    """Process hierarchy levels"""
    CORE_FUNCTION = "core_function"
    OPERATION = "operation"
    PROCEDURE = "procedure"
    SPECIFIC_STEP = "specific_step"


class CriticalityLevel(PyEnum):
    """Criticality levels for processes and requirements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceStatus(PyEnum):
    """Compliance status values"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class RiskLevel(PyEnum):
    """Risk level values"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ProcessingStatus(PyEnum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Core Models
class Document(Base):
    """Document model for uploaded files"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    file_hash = Column(String(64), unique=True, index=True)  # SHA-256 hash
    
    # Processing information
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    processing_error = Column(Text)
    
    # Metadata
    uploaded_at = Column(DateTime, default=func.now(), nullable=False)
    uploaded_by = Column(String(100))  # User identifier
    
    # Extracted content summary
    total_pages = Column(Integer)
    total_words = Column(Integer)
    total_characters = Column(Integer)
    language = Column(String(10))
    
    # Relationships
    processes = relationship("Process", back_populates="document", cascade="all, delete-orphan")
    decision_points = relationship("DecisionPoint", back_populates="document", cascade="all, delete-orphan")
    compliance_items = relationship("ComplianceItem", back_populates="document", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_status_date', 'processing_status', 'uploaded_at'),
        Index('idx_documents_filename_hash', 'filename', 'file_hash'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.processing_status}')>"


class Process(Base):
    """Process model for extracted organizational processes"""
    __tablename__ = "processes"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Process identification
    name = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Classification
    domain = Column(Enum(KnowledgeDomain), nullable=False, index=True)
    hierarchy_level = Column(Enum(HierarchyLevel), nullable=False, index=True)
    criticality = Column(Enum(CriticalityLevel), nullable=False, index=True)
    
    # Process details
    steps = Column(JSON)  # List of process steps
    prerequisites = Column(JSON)  # List of prerequisites
    success_criteria = Column(JSON)  # List of success criteria
    responsible_parties = Column(JSON)  # List of responsible parties
    
    # Timing information
    estimated_duration = Column(String(100))  # e.g., "30 minutes", "2 hours"
    frequency = Column(String(100))  # e.g., "daily", "weekly", "as needed"
    
    # Quality metrics
    confidence = Column(Float, nullable=False, index=True)
    extraction_method = Column(String(50))  # Method used for extraction
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Source information
    source_text = Column(Text)  # Original text from which process was extracted
    source_page = Column(Integer)  # Page number in source document
    source_section = Column(String(200))  # Section or chapter
    
    # Relationships
    document = relationship("Document", back_populates="processes")
    
    # Indexes
    __table_args__ = (
        Index('idx_processes_domain_hierarchy', 'domain', 'hierarchy_level'),
        Index('idx_processes_confidence_criticality', 'confidence', 'criticality'),
        Index('idx_processes_document_domain', 'document_id', 'domain'),
    )
    
    def __repr__(self):
        return f"<Process(id={self.id}, name='{self.name[:50]}...', domain='{self.domain}')>"


class DecisionPoint(Base):
    """Decision point model for extracted decision-making processes"""
    __tablename__ = "decision_points"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Decision identification
    name = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=False)
    decision_type = Column(String(50))  # e.g., "approval", "selection", "conditional"
    
    # Decision criteria and outcomes
    criteria = Column(JSON)  # Decision criteria as key-value pairs
    outcomes = Column(JSON)  # Possible outcomes and their conditions
    
    # Authority and escalation
    authority_level = Column(String(100))  # Required authority level
    escalation_path = Column(String(200))  # Escalation procedure
    
    # Quality metrics
    confidence = Column(Float, nullable=False, index=True)
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False)
    source_text = Column(Text)
    source_page = Column(Integer)
    
    # Relationships
    document = relationship("Document", back_populates="decision_points")
    
    # Indexes
    __table_args__ = (
        Index('idx_decision_points_type_authority', 'decision_type', 'authority_level'),
        Index('idx_decision_points_confidence', 'confidence'),
    )
    
    def __repr__(self):
        return f"<DecisionPoint(id={self.id}, name='{self.name[:50]}...', type='{self.decision_type}')>"


class ComplianceItem(Base):
    """Compliance item model for regulatory requirements"""
    __tablename__ = "compliance_items"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Regulation identification
    regulation_name = Column(String(200), nullable=False, index=True)
    regulation_section = Column(String(100))
    regulation_authority = Column(String(100))  # e.g., "OSHA", "EPA", "FDA"
    
    # Requirement details
    requirement = Column(Text, nullable=False)
    requirement_type = Column(String(50))  # e.g., "mandatory", "recommended", "guideline"
    
    # Compliance tracking
    status = Column(Enum(ComplianceStatus), nullable=False, index=True)
    responsible_party = Column(String(200))
    review_frequency = Column(String(50))  # e.g., "annual", "quarterly"
    next_review_date = Column(DateTime, index=True)
    last_review_date = Column(DateTime)
    
    # Quality metrics
    confidence = Column(Float, nullable=False, index=True)
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    source_text = Column(Text)
    source_page = Column(Integer)
    
    # Relationships
    document = relationship("Document", back_populates="compliance_items")
    
    # Indexes
    __table_args__ = (
        Index('idx_compliance_regulation_status', 'regulation_name', 'status'),
        Index('idx_compliance_authority_type', 'regulation_authority', 'requirement_type'),
        Index('idx_compliance_review_dates', 'next_review_date', 'last_review_date'),
    )
    
    def __repr__(self):
        return f"<ComplianceItem(id={self.id}, regulation='{self.regulation_name}', status='{self.status}')>"


class RiskAssessment(Base):
    """Risk assessment model for identified hazards and risks"""
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Hazard identification
    hazard = Column(String(500), nullable=False, index=True)
    hazard_category = Column(String(100))  # e.g., "chemical", "physical", "biological"
    risk_description = Column(Text, nullable=False)
    
    # Risk assessment
    likelihood = Column(String(50))  # e.g., "high", "medium", "low"
    impact = Column(String(50))  # e.g., "severe", "moderate", "minor"
    overall_risk_level = Column(Enum(RiskLevel), nullable=False, index=True)
    
    # Risk management
    mitigation_strategies = Column(JSON)  # List of mitigation strategies
    control_measures = Column(JSON)  # List of control measures
    monitoring_requirements = Column(Text)
    
    # Responsible parties
    risk_owner = Column(String(200))
    assessor = Column(String(200))
    
    # Quality metrics
    confidence = Column(Float, nullable=False, index=True)
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    assessment_date = Column(DateTime)
    next_assessment_date = Column(DateTime, index=True)
    source_text = Column(Text)
    source_page = Column(Integer)
    
    # Relationships
    document = relationship("Document", back_populates="risk_assessments")
    
    # Indexes
    __table_args__ = (
        Index('idx_risk_hazard_level', 'hazard_category', 'overall_risk_level'),
        Index('idx_risk_assessment_dates', 'assessment_date', 'next_assessment_date'),
        Index('idx_risk_confidence_level', 'confidence', 'overall_risk_level'),
    )
    
    def __repr__(self):
        return f"<RiskAssessment(id={self.id}, hazard='{self.hazard[:50]}...', level='{self.overall_risk_level}')>"


class KnowledgeEntity(Base):
    """Named entities extracted from documents with intelligent categorization"""
    __tablename__ = "knowledge_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Entity information
    text = Column(String(500), nullable=False, index=True)
    label = Column(String(50), nullable=False, index=True)  # e.g., "PERSON", "ORG", "EQUIPMENT", "DEFINITION", "METRIC", "ROLE"
    confidence = Column(Float, nullable=False)
    
    # Intelligent categorization fields
    entity_type = Column(String(50), index=True)  # process, policy, metric, role, requirement, risk, definition, etc.
    category = Column(String(50), index=True)  # process_intelligence, compliance_governance, etc.
    priority_level = Column(String(20), index=True)  # high, medium, low
    business_relevance = Column(Float, default=0.5)  # 0.0 to 1.0
    
    # Quality scores
    completeness_score = Column(Float, default=0.5)
    clarity_score = Column(Float, default=0.5)
    actionability_score = Column(Float, default=0.5)
    
    # Position in document
    start_position = Column(Integer)
    end_position = Column(Integer)
    context = Column(Text)  # Surrounding text for context
    
    # Enhanced context and relationships
    context_tags = Column(JSON)  # List of context tags
    relationships = Column(JSON)  # List of related entity identifiers
    structured_data = Column(JSON)  # Category-specific structured data
    
    # Source information
    source_section = Column(String(200))  # Section where entity was found
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False)
    extraction_method = Column(String(50))  # extraction method used
    
    # Indexes
    __table_args__ = (
        Index('idx_entities_text_label', 'text', 'label'),
        Index('idx_entities_document_label', 'document_id', 'label'),
        Index('idx_entities_confidence', 'confidence'),
    )
    
    def __repr__(self):
        return f"<KnowledgeEntity(id={self.id}, text='{self.text}', label='{self.label}')>"


class ProcessingTask(Base):
    """Processing task tracking for async operations"""
    __tablename__ = "processing_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)  # Celery task ID
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    
    # Task information
    task_type = Column(String(50), nullable=False)  # e.g., "document_processing", "knowledge_extraction"
    status = Column(String(20), nullable=False, index=True)  # e.g., "pending", "running", "completed", "failed"
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String(200))
    total_steps = Column(Integer)
    
    # Results and errors
    result = Column(JSON)  # Task result data
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    # Timing
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Metadata
    worker_id = Column(String(100))  # Celery worker ID
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Indexes
    __table_args__ = (
        Index('idx_tasks_status_created', 'status', 'created_at'),
        Index('idx_tasks_type_status', 'task_type', 'status'),
    )
    
    def __repr__(self):
        return f"<ProcessingTask(id={self.id}, task_id='{self.task_id}', status='{self.status}')>"


class ProcessedKnowledgeUnit(Base):
    """Processed knowledge units ready for database ingestion"""
    __tablename__ = "processed_knowledge_units"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Unit identification
    unit_identifier = Column(String(100), nullable=False, index=True)
    unit_type = Column(String(50), nullable=False, index=True)  # process, compliance, risk, etc.
    category = Column(String(50), nullable=False, index=True)
    
    # Quality metrics
    quality_score = Column(Float, nullable=False, index=True)
    business_relevance = Column(Float, nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    completeness_score = Column(Float, nullable=False)
    actionability_score = Column(Float, nullable=False)
    
    # Processing information
    primary_table = Column(String(50))  # Target database table
    related_entries_count = Column(Integer, default=0)
    synthesis_performed = Column(Boolean, default=False)
    
    # Content summary
    summary = Column(Text)
    synthesis_notes = Column(Text)
    
    # Metadata
    processed_at = Column(DateTime, default=func.now(), nullable=False)
    extraction_metadata = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_knowledge_units_quality', 'quality_score', 'business_relevance'),
        Index('idx_knowledge_units_type_category', 'unit_type', 'category'),
        Index('idx_knowledge_units_document', 'document_id', 'unit_type'),
    )
    
    def __repr__(self):
        return f"<ProcessedKnowledgeUnit(id={self.id}, identifier='{self.unit_identifier}', type='{self.unit_type}')>"


class SystemMetrics(Base):
    """System metrics and performance tracking"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric information
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))  # e.g., "seconds", "bytes", "count"
    
    # Context
    component = Column(String(50))  # e.g., "document_processor", "knowledge_extractor"
    operation = Column(String(100))  # e.g., "pdf_extraction", "nlp_processing"
    
    # Metadata
    recorded_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    session_id = Column(String(100))  # Processing session identifier
    
    # Additional data (avoid reserved attribute name 'metadata')
    extra_metadata = Column("metadata", JSON)  # DB column still named 'metadata'
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_name_time', 'metric_name', 'recorded_at'),
        Index('idx_metrics_component_operation', 'component', 'operation'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric='{self.metric_name}', value={self.metric_value})>"


# Association tables for many-to-many relationships (if needed in future)
# These can be uncommented and modified as needed

# process_dependencies = Table(
#     'process_dependencies',
#     Base.metadata,
#     Column('parent_process_id', Integer, ForeignKey('processes.id'), primary_key=True),
#     Column('child_process_id', Integer, ForeignKey('processes.id'), primary_key=True),
#     Column('dependency_type', String(50)),  # e.g., "prerequisite", "follows", "parallel"
#     Column('created_at', DateTime, default=func.now())
# )

# process_tags = Table(
#     'process_tags',
#     Base.metadata,
#     Column('process_id', Integer, ForeignKey('processes.id'), primary_key=True),
#     Column('tag', String(100), primary_key=True),
#     Column('created_at', DateTime, default=func.now())
# )


# Model utilities
def get_all_models():
    """Get all database models"""
    return [
        Document,
        Process,
        DecisionPoint,
        ComplianceItem,
        RiskAssessment,
        KnowledgeEntity,
        ProcessedKnowledgeUnit,
        ProcessingTask,
        SystemMetrics
    ]


def get_model_by_name(model_name: str):
    """Get model class by name"""
    models = {
        'Document': Document,
        'Process': Process,
        'DecisionPoint': DecisionPoint,
        'ComplianceItem': ComplianceItem,
        'RiskAssessment': RiskAssessment,
        'KnowledgeEntity': KnowledgeEntity,
        'ProcessedKnowledgeUnit': ProcessedKnowledgeUnit,
        'ProcessingTask': ProcessingTask,
        'SystemMetrics': SystemMetrics
    }
    return models.get(model_name)


def get_enum_values(enum_class):
    """Get all values from an enum class"""
    return [item.value for item in enum_class]