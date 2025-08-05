"""
Unit tests for enhanced knowledge extraction database models.
Tests model relationships, constraints, and data integrity.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from src.database.models import (
    Base, Document, KnowledgeItem, WorkflowDependency, DecisionTree,
    OptimizationPattern, OptimizationPatternApplication, CommunicationFlow,
    KnowledgeRelationship, KnowledgeGap, KnowledgeGapEvidence
)

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_knowledge.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_document(db_session):
    """Create a sample document for testing."""
    document = Document(
        filename="test_manual.pdf",
        file_type="pdf",
        extracted_text="Sample manual content",
        status="completed"
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return document


@pytest.fixture
def sample_knowledge_item(db_session, sample_document):
    """Create a sample knowledge item for testing."""
    knowledge_item = KnowledgeItem(
        process_id="PROC-001",
        name="Equipment Startup Procedure",
        description="Standard procedure for equipment startup",
        knowledge_type="procedural",
        domain="operational",
        hierarchy_level=3,
        confidence_score=0.85,
        source_quality="high",
        completeness_index=0.9,
        criticality_level="high",
        access_level="internal",
        source_document_id=sample_document.id
    )
    db_session.add(knowledge_item)
    db_session.commit()
    db_session.refresh(knowledge_item)
    return knowledge_item


class TestKnowledgeItem:
    """Test KnowledgeItem model functionality."""

    def test_create_knowledge_item(self, db_session, sample_document):
        """Test creating a knowledge item with all required fields."""
        knowledge_item = KnowledgeItem(
            process_id="PROC-TEST-001",
            name="Test Process",
            description="A test process description",
            knowledge_type="explicit",
            domain="safety",
            hierarchy_level=2,
            confidence_score=0.75,
            source_quality="medium",
            completeness_index=0.8,
            criticality_level="medium",
            access_level="internal",
            source_document_id=sample_document.id
        )
        
        db_session.add(knowledge_item)
        db_session.commit()
        
        # Verify the item was created
        retrieved = db_session.query(KnowledgeItem).filter_by(process_id="PROC-TEST-001").first()
        assert retrieved is not None
        assert retrieved.name == "Test Process"
        assert retrieved.knowledge_type == "explicit"
        assert retrieved.domain == "safety"
        assert retrieved.hierarchy_level == 2
        assert retrieved.confidence_score == 0.75

    def test_unique_process_id_constraint(self, db_session, sample_document):
        """Test that process_id must be unique."""
        # Create first knowledge item
        item1 = KnowledgeItem(
            process_id="PROC-UNIQUE-001",
            name="First Process",
            knowledge_type="explicit",
            domain="operational",
            hierarchy_level=1,
            confidence_score=0.8,
            source_quality="high",
            completeness_index=0.9,
            criticality_level="high",
            access_level="internal"
        )
        db_session.add(item1)
        db_session.commit()
        
        # Try to create second item with same process_id
        item2 = KnowledgeItem(
            process_id="PROC-UNIQUE-001",  # Same process_id
            name="Second Process",
            knowledge_type="tacit",
            domain="safety",
            hierarchy_level=2,
            confidence_score=0.7,
            source_quality="medium",
            completeness_index=0.8,
            criticality_level="medium",
            access_level="internal"
        )
        db_session.add(item2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_required_fields(self, db_session):
        """Test that required fields cannot be null."""
        # Test missing process_id
        with pytest.raises(IntegrityError):
            item = KnowledgeItem(
                name="Test Process",
                knowledge_type="explicit",
                domain="operational",
                hierarchy_level=1,
                confidence_score=0.8,
                source_quality="high",
                completeness_index=0.9,
                criticality_level="high",
                access_level="internal"
            )
            db_session.add(item)
            db_session.commit()

    def test_default_values(self, db_session):
        """Test that default values are set correctly."""
        knowledge_item = KnowledgeItem(
            process_id="PROC-DEFAULT-001",
            name="Test Process",
            knowledge_type="explicit",
            domain="operational",
            hierarchy_level=1
        )
        db_session.add(knowledge_item)
        db_session.commit()
        db_session.refresh(knowledge_item)
        
        assert knowledge_item.confidence_score == 0.0
        assert knowledge_item.source_quality == "medium"
        assert knowledge_item.completeness_index == 0.0
        assert knowledge_item.criticality_level == "medium"
        assert knowledge_item.access_level == "internal"
        assert knowledge_item.created_at is not None
        assert knowledge_item.updated_at is not None

    def test_document_relationship(self, db_session, sample_document):
        """Test relationship with Document model."""
        knowledge_item = KnowledgeItem(
            process_id="PROC-REL-001",
            name="Related Process",
            knowledge_type="explicit",
            domain="operational",
            hierarchy_level=1,
            source_document_id=sample_document.id
        )
        db_session.add(knowledge_item)
        db_session.commit()
        db_session.refresh(knowledge_item)
        
        # Test relationship access
        assert knowledge_item.source_document is not None
        assert knowledge_item.source_document.filename == "test_manual.pdf"


class TestWorkflowDependency:
    """Test WorkflowDependency model functionality."""

    def test_create_workflow_dependency(self, db_session, sample_knowledge_item):
        """Test creating a workflow dependency."""
        # Create second knowledge item
        target_item = KnowledgeItem(
            process_id="PROC-002",
            name="Target Process",
            knowledge_type="procedural",
            domain="operational",
            hierarchy_level=3,
            confidence_score=0.8,
            source_quality="high",
            completeness_index=0.85,
            criticality_level="medium",
            access_level="internal"
        )
        db_session.add(target_item)
        db_session.commit()
        
        # Create dependency
        dependency = WorkflowDependency(
            source_process_id=sample_knowledge_item.process_id,
            target_process_id=target_item.process_id,
            dependency_type="prerequisite",
            strength=0.9,
            conditions={"temperature": "> 20C", "pressure": "< 5 bar"},
            confidence=0.85
        )
        db_session.add(dependency)
        db_session.commit()
        
        # Verify creation
        retrieved = db_session.query(WorkflowDependency).first()
        assert retrieved is not None
        assert retrieved.dependency_type == "prerequisite"
        assert retrieved.strength == 0.9
        assert retrieved.conditions["temperature"] == "> 20C"

    def test_dependency_relationships(self, db_session, sample_knowledge_item):
        """Test relationships with KnowledgeItem."""
        target_item = KnowledgeItem(
            process_id="PROC-TARGET-001",
            name="Target Process",
            knowledge_type="procedural",
            domain="operational",
            hierarchy_level=3
        )
        db_session.add(target_item)
        db_session.commit()
        
        dependency = WorkflowDependency(
            source_process_id=sample_knowledge_item.process_id,
            target_process_id=target_item.process_id,
            dependency_type="parallel",
            strength=0.7,
            confidence=0.8
        )
        db_session.add(dependency)
        db_session.commit()
        db_session.refresh(dependency)
        
        # Test relationships
        assert dependency.source_process is not None
        assert dependency.target_process is not None
        assert dependency.source_process.name == "Equipment Startup Procedure"
        assert dependency.target_process.name == "Target Process"

    def test_default_strength(self, db_session, sample_knowledge_item):
        """Test default strength value."""
        target_item = KnowledgeItem(
            process_id="PROC-DEFAULT-002",
            name="Default Target",
            knowledge_type="explicit",
            domain="operational",
            hierarchy_level=1
        )
        db_session.add(target_item)
        db_session.commit()
        
        dependency = WorkflowDependency(
            source_process_id=sample_knowledge_item.process_id,
            target_process_id=target_item.process_id,
            dependency_type="downstream",
            confidence=0.7
        )
        db_session.add(dependency)
        db_session.commit()
        db_session.refresh(dependency)
        
        assert dependency.strength == 0.5  # Default value


class TestDecisionTree:
    """Test DecisionTree model functionality."""

    def test_create_decision_tree(self, db_session, sample_knowledge_item):
        """Test creating a decision tree."""
        decision_tree = DecisionTree(
            process_id=sample_knowledge_item.process_id,
            decision_point="Temperature Check",
            decision_type="threshold",
            conditions={
                "parameter": "temperature",
                "threshold": 50,
                "unit": "celsius"
            },
            outcomes={
                "above_threshold": "Continue process",
                "below_threshold": "Wait and recheck"
            },
            confidence=0.9,
            priority="high"
        )
        db_session.add(decision_tree)
        db_session.commit()
        
        # Verify creation
        retrieved = db_session.query(DecisionTree).first()
        assert retrieved is not None
        assert retrieved.decision_point == "Temperature Check"
        assert retrieved.decision_type == "threshold"
        assert retrieved.conditions["parameter"] == "temperature"
        assert retrieved.outcomes["above_threshold"] == "Continue process"

    def test_decision_tree_relationship(self, db_session, sample_knowledge_item):
        """Test relationship with KnowledgeItem."""
        decision_tree = DecisionTree(
            process_id=sample_knowledge_item.process_id,
            decision_point="Safety Check",
            decision_type="binary",
            conditions={"safety_status": "ok"},
            outcomes={"true": "proceed", "false": "stop"},
            confidence=0.95,
            priority="critical"
        )
        db_session.add(decision_tree)
        db_session.commit()
        db_session.refresh(decision_tree)
        
        # Test relationship
        assert decision_tree.knowledge_item is not None
        assert decision_tree.knowledge_item.name == "Equipment Startup Procedure"

    def test_default_priority(self, db_session, sample_knowledge_item):
        """Test default priority value."""
        decision_tree = DecisionTree(
            process_id=sample_knowledge_item.process_id,
            decision_point="Default Priority Check",
            decision_type="multiple_choice",
            conditions={"options": ["A", "B", "C"]},
            outcomes={"A": "path1", "B": "path2", "C": "path3"},
            confidence=0.8
        )
        db_session.add(decision_tree)
        db_session.commit()
        db_session.refresh(decision_tree)
        
        assert decision_tree.priority == "medium"  # Default value


class TestOptimizationPattern:
    """Test OptimizationPattern model functionality."""

    def test_create_optimization_pattern(self, db_session):
        """Test creating an optimization pattern."""
        pattern = OptimizationPattern(
            pattern_type="resource",
            name="Batch Processing Optimization",
            description="Optimize batch processing for better resource utilization",
            domain="operational",
            conditions={
                "batch_size": "> 100",
                "resource_availability": "high"
            },
            improvements={
                "efficiency_gain": "25%",
                "resource_reduction": "15%"
            },
            success_metrics={
                "throughput": "increase by 20%",
                "cost": "reduce by 10%"
            },
            confidence=0.85,
            impact_level="high",
            implementation_complexity="medium"
        )
        db_session.add(pattern)
        db_session.commit()
        
        # Verify creation
        retrieved = db_session.query(OptimizationPattern).first()
        assert retrieved is not None
        assert retrieved.pattern_type == "resource"
        assert retrieved.name == "Batch Processing Optimization"
        assert retrieved.improvements["efficiency_gain"] == "25%"

    def test_default_values(self, db_session):
        """Test default values for optimization pattern."""
        pattern = OptimizationPattern(
            pattern_type="time",
            name="Time Optimization",
            description="Reduce processing time",
            domain="operational",
            conditions={"load": "normal"},
            improvements={"time_saved": "30%"},
            success_metrics={"completion_time": "reduce by 30%"}
        )
        db_session.add(pattern)
        db_session.commit()
        db_session.refresh(pattern)
        
        assert pattern.confidence == 0.0
        assert pattern.impact_level == "medium"
        assert pattern.implementation_complexity == "medium"


class TestKnowledgeRelationship:
    """Test KnowledgeRelationship model functionality."""

    def test_create_knowledge_relationship(self, db_session, sample_knowledge_item):
        """Test creating a knowledge relationship."""
        # Create second knowledge item
        target_item = KnowledgeItem(
            process_id="PROC-REL-002",
            name="Related Process",
            knowledge_type="explicit",
            domain="safety",
            hierarchy_level=2
        )
        db_session.add(target_item)
        db_session.commit()
        
        # Create relationship
        relationship = KnowledgeRelationship(
            source_id=sample_knowledge_item.process_id,
            target_id=target_item.process_id,
            relationship_type="depends_on",
            strength=0.8,
            bidirectional=False,
            relationship_metadata={
                "dependency_reason": "safety requirement",
                "criticality": "high"
            },
            confidence=0.9
        )
        db_session.add(relationship)
        db_session.commit()
        
        # Verify creation
        retrieved = db_session.query(KnowledgeRelationship).first()
        assert retrieved is not None
        assert retrieved.relationship_type == "depends_on"
        assert retrieved.strength == 0.8
        assert retrieved.bidirectional is False
        assert retrieved.relationship_metadata["dependency_reason"] == "safety requirement"

    def test_relationship_access(self, db_session, sample_knowledge_item):
        """Test accessing related knowledge items."""
        target_item = KnowledgeItem(
            process_id="PROC-ACCESS-002",
            name="Target Item",
            knowledge_type="tacit",
            domain="equipment",
            hierarchy_level=4
        )
        db_session.add(target_item)
        db_session.commit()
        
        relationship = KnowledgeRelationship(
            source_id=sample_knowledge_item.process_id,
            target_id=target_item.process_id,
            relationship_type="enhances",
            strength=0.7,
            bidirectional=True,
            confidence=0.85
        )
        db_session.add(relationship)
        db_session.commit()
        db_session.refresh(relationship)
        
        # Test relationship access
        assert relationship.source_knowledge is not None
        assert relationship.target_knowledge is not None
        assert relationship.source_knowledge.name == "Equipment Startup Procedure"
        assert relationship.target_knowledge.name == "Target Item"


class TestKnowledgeGap:
    """Test KnowledgeGap model functionality."""

    def test_create_knowledge_gap(self, db_session):
        """Test creating a knowledge gap."""
        gap = KnowledgeGap(
            gap_type="missing_documentation",
            title="Missing Safety Procedures",
            description="Critical safety procedures are not documented",
            domain="safety",
            affected_processes=["PROC-001", "PROC-002", "PROC-003"],
            impact_assessment={
                "risk_level": "high",
                "affected_personnel": 25,
                "potential_incidents": "equipment damage, injury"
            },
            priority="critical",
            status="identified",
            assigned_to="safety_team@company.com"
        )
        db_session.add(gap)
        db_session.commit()
        
        # Verify creation
        retrieved = db_session.query(KnowledgeGap).first()
        assert retrieved is not None
        assert retrieved.gap_type == "missing_documentation"
        assert retrieved.title == "Missing Safety Procedures"
        assert retrieved.priority == "critical"
        assert "PROC-001" in retrieved.affected_processes

    def test_default_values(self, db_session):
        """Test default values for knowledge gap."""
        gap = KnowledgeGap(
            gap_type="inconsistent_info",
            title="Inconsistent Process Info",
            description="Process information varies across documents",
            domain="operational"
        )
        db_session.add(gap)
        db_session.commit()
        db_session.refresh(gap)
        
        assert gap.priority == "medium"
        assert gap.status == "identified"
        assert gap.identified_at is not None

    def test_knowledge_gap_evidence_relationship(self, db_session, sample_document):
        """Test relationship with KnowledgeGapEvidence."""
        gap = KnowledgeGap(
            gap_type="outdated",
            title="Outdated Equipment Manual",
            description="Equipment manual is from 2015",
            domain="equipment",
            priority="high",
            status="identified"
        )
        db_session.add(gap)
        db_session.commit()
        db_session.refresh(gap)
        
        # Add evidence
        evidence = KnowledgeGapEvidence(
            gap_id=gap.id,
            evidence_type="outdated_timestamp",
            description="Manual last updated in 2015, equipment upgraded in 2020",
            source_document_id=sample_document.id,
            confidence=0.95
        )
        db_session.add(evidence)
        db_session.commit()
        
        # Test relationship
        db_session.refresh(gap)
        assert len(gap.gap_evidence) == 1
        assert gap.gap_evidence[0].evidence_type == "outdated_timestamp"
        assert gap.gap_evidence[0].source_document.filename == "test_manual.pdf"


class TestModelIntegration:
    """Test integration between different models."""

    def test_cascade_delete_knowledge_item(self, db_session, sample_knowledge_item):
        """Test that deleting a knowledge item cascades to related records."""
        # Create related records
        decision_tree = DecisionTree(
            process_id=sample_knowledge_item.process_id,
            decision_point="Test Decision",
            decision_type="binary",
            conditions={"test": "value"},
            outcomes={"true": "yes", "false": "no"},
            confidence=0.8
        )
        db_session.add(decision_tree)
        db_session.commit()
        
        # Verify records exist
        assert db_session.query(DecisionTree).count() == 1
        
        # Delete knowledge item
        db_session.delete(sample_knowledge_item)
        db_session.commit()
        
        # Verify cascade delete worked
        assert db_session.query(DecisionTree).count() == 0

    def test_complex_relationship_queries(self, db_session, sample_knowledge_item):
        """Test complex queries across multiple related models."""
        # Create additional knowledge items
        item2 = KnowledgeItem(
            process_id="PROC-COMPLEX-002",
            name="Complex Process 2",
            knowledge_type="tacit",
            domain="safety",
            hierarchy_level=2
        )
        item3 = KnowledgeItem(
            process_id="PROC-COMPLEX-003",
            name="Complex Process 3",
            knowledge_type="explicit",
            domain="equipment",
            hierarchy_level=4
        )
        db_session.add_all([item2, item3])
        db_session.commit()
        
        # Create relationships
        rel1 = KnowledgeRelationship(
            source_id=sample_knowledge_item.process_id,
            target_id=item2.process_id,
            relationship_type="depends_on",
            strength=0.9,
            confidence=0.85
        )
        rel2 = KnowledgeRelationship(
            source_id=item2.process_id,
            target_id=item3.process_id,
            relationship_type="enables",
            strength=0.7,
            confidence=0.8
        )
        db_session.add_all([rel1, rel2])
        db_session.commit()
        
        # Test simpler query: Find all relationships of type "depends_on"
        depends_on_relationships = (
            db_session.query(KnowledgeRelationship)
            .filter(KnowledgeRelationship.relationship_type == "depends_on")
            .all()
        )
        
        # Should find the relationship we created
        assert len(depends_on_relationships) == 1
        assert depends_on_relationships[0].source_id == sample_knowledge_item.process_id
        assert depends_on_relationships[0].target_id == item2.process_id
        
        # Test finding knowledge items by domain
        safety_items = (
            db_session.query(KnowledgeItem)
            .filter(KnowledgeItem.domain == "safety")
            .all()
        )
        
        assert len(safety_items) == 1
        assert safety_items[0].name == "Complex Process 2"