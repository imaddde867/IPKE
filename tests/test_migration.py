"""
Test the Alembic migration for enhanced knowledge extraction models.
"""

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from src.database.models import Base

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_migration.db"
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


def test_knowledge_extraction_tables_created(db_session):
    """Test that all new knowledge extraction tables are created."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    # Check that all new tables exist
    expected_tables = [
        'knowledge_items',
        'workflow_dependencies',
        'decision_trees',
        'optimization_patterns',
        'optimization_pattern_applications',
        'communication_flows',
        'knowledge_relationships',
        'knowledge_gaps',
        'knowledge_gap_evidence'
    ]
    
    for table in expected_tables:
        assert table in table_names, f"Table {table} was not created"


def test_knowledge_items_table_structure(db_session):
    """Test the structure of the knowledge_items table."""
    inspector = inspect(engine)
    columns = inspector.get_columns('knowledge_items')
    column_names = [col['name'] for col in columns]
    
    # Check required columns exist
    required_columns = [
        'id', 'process_id', 'name', 'description', 'knowledge_type',
        'domain', 'hierarchy_level', 'confidence_score', 'source_quality',
        'completeness_index', 'criticality_level', 'access_level',
        'source_document_id', 'created_at', 'updated_at'
    ]
    
    for column in required_columns:
        assert column in column_names, f"Column {column} missing from knowledge_items table"
    
    # Check indexes
    indexes = inspector.get_indexes('knowledge_items')
    index_columns = [idx['column_names'] for idx in indexes]
    
    # Should have index on process_id (unique)
    assert ['process_id'] in index_columns
    # Should have indexes on commonly queried fields
    assert ['domain'] in index_columns
    assert ['knowledge_type'] in index_columns


def test_foreign_key_constraints(db_session):
    """Test that foreign key constraints are properly set up."""
    inspector = inspect(engine)
    
    # Test knowledge_items foreign key to documents
    fks = inspector.get_foreign_keys('knowledge_items')
    document_fk = next((fk for fk in fks if fk['referred_table'] == 'documents'), None)
    assert document_fk is not None, "Foreign key to documents table missing"
    assert 'source_document_id' in document_fk['constrained_columns']
    
    # Test workflow_dependencies foreign keys to knowledge_items
    fks = inspector.get_foreign_keys('workflow_dependencies')
    assert len(fks) == 2, "Should have 2 foreign keys to knowledge_items"
    
    fk_columns = []
    for fk in fks:
        assert fk['referred_table'] == 'knowledge_items'
        fk_columns.extend(fk['constrained_columns'])
    
    assert 'source_process_id' in fk_columns
    assert 'target_process_id' in fk_columns


def test_json_columns_support(db_session):
    """Test that JSON columns are properly supported."""
    inspector = inspect(engine)
    
    # Check decision_trees table has JSON columns
    columns = inspector.get_columns('decision_trees')
    json_columns = [col for col in columns if 'JSON' in str(col['type'])]
    
    # Should have conditions and outcomes as JSON columns
    json_column_names = [col['name'] for col in json_columns]
    assert 'conditions' in json_column_names
    assert 'outcomes' in json_column_names


def test_table_relationships_integrity(db_session):
    """Test that table relationships maintain referential integrity."""
    from src.database.models import (
        KnowledgeItem, WorkflowDependency, DecisionTree, Document
    )
    
    # Create a document first
    document = Document(
        filename="test.pdf",
        file_type="pdf",
        extracted_text="test content",
        status="completed"
    )
    db_session.add(document)
    db_session.commit()
    
    # Create a knowledge item
    knowledge_item = KnowledgeItem(
        process_id="TEST-001",
        name="Test Process",
        knowledge_type="explicit",
        domain="operational",
        hierarchy_level=1,
        source_document_id=document.id
    )
    db_session.add(knowledge_item)
    db_session.commit()
    
    # Create a decision tree linked to the knowledge item
    decision_tree = DecisionTree(
        process_id=knowledge_item.process_id,
        decision_point="Test Decision",
        decision_type="binary",
        conditions={"test": "condition"},
        outcomes={"true": "proceed", "false": "stop"},
        confidence=0.8
    )
    db_session.add(decision_tree)
    db_session.commit()
    
    # Verify relationships work
    retrieved_item = db_session.query(KnowledgeItem).filter_by(process_id="TEST-001").first()
    assert retrieved_item is not None
    assert retrieved_item.source_document.filename == "test.pdf"
    assert len(retrieved_item.decision_trees) == 1
    assert retrieved_item.decision_trees[0].decision_point == "Test Decision"


def test_cascade_delete_behavior(db_session):
    """Test that cascade delete works properly for related records."""
    from src.database.models import KnowledgeItem, DecisionTree
    
    # Create knowledge item with related decision tree
    knowledge_item = KnowledgeItem(
        process_id="CASCADE-001",
        name="Cascade Test Process",
        knowledge_type="explicit",
        domain="operational",
        hierarchy_level=1
    )
    db_session.add(knowledge_item)
    db_session.commit()
    
    decision_tree = DecisionTree(
        process_id=knowledge_item.process_id,
        decision_point="Cascade Test Decision",
        decision_type="binary",
        conditions={"test": "cascade"},
        outcomes={"true": "yes", "false": "no"},
        confidence=0.9
    )
    db_session.add(decision_tree)
    db_session.commit()
    
    # Verify both records exist
    assert db_session.query(KnowledgeItem).filter_by(process_id="CASCADE-001").first() is not None
    assert db_session.query(DecisionTree).filter_by(process_id="CASCADE-001").first() is not None
    
    # Delete the knowledge item
    db_session.delete(knowledge_item)
    db_session.commit()
    
    # Verify cascade delete worked
    assert db_session.query(KnowledgeItem).filter_by(process_id="CASCADE-001").first() is None
    assert db_session.query(DecisionTree).filter_by(process_id="CASCADE-001").first() is None