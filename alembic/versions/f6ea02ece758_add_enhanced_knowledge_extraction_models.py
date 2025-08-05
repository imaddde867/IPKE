"""Add enhanced knowledge extraction models

Revision ID: f6ea02ece758
Revises: 
Create Date: 2025-08-05 20:53:11.184097

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6ea02ece758'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create knowledge_items table
    op.create_table('knowledge_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('process_id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('knowledge_type', sa.String(length=50), nullable=False),
        sa.Column('domain', sa.String(length=50), nullable=False),
        sa.Column('hierarchy_level', sa.Integer(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('source_quality', sa.String(length=20), nullable=False),
        sa.Column('completeness_index', sa.Float(), nullable=False),
        sa.Column('criticality_level', sa.String(length=20), nullable=False),
        sa.Column('access_level', sa.String(length=20), nullable=False),
        sa.Column('source_document_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['source_document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_items_created_at'), 'knowledge_items', ['created_at'], unique=False)
    op.create_index(op.f('ix_knowledge_items_domain'), 'knowledge_items', ['domain'], unique=False)
    op.create_index(op.f('ix_knowledge_items_hierarchy_level'), 'knowledge_items', ['hierarchy_level'], unique=False)
    op.create_index(op.f('ix_knowledge_items_id'), 'knowledge_items', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_items_knowledge_type'), 'knowledge_items', ['knowledge_type'], unique=False)
    op.create_index(op.f('ix_knowledge_items_name'), 'knowledge_items', ['name'], unique=False)
    op.create_index(op.f('ix_knowledge_items_process_id'), 'knowledge_items', ['process_id'], unique=True)
    op.create_index(op.f('ix_knowledge_items_updated_at'), 'knowledge_items', ['updated_at'], unique=False)

    # Create workflow_dependencies table
    op.create_table('workflow_dependencies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_process_id', sa.String(length=100), nullable=False),
        sa.Column('target_process_id', sa.String(length=100), nullable=False),
        sa.Column('dependency_type', sa.String(length=50), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('conditions', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['source_process_id'], ['knowledge_items.process_id'], ),
        sa.ForeignKeyConstraint(['target_process_id'], ['knowledge_items.process_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_dependencies_dependency_type'), 'workflow_dependencies', ['dependency_type'], unique=False)
    op.create_index(op.f('ix_workflow_dependencies_id'), 'workflow_dependencies', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_dependencies_source_process_id'), 'workflow_dependencies', ['source_process_id'], unique=False)
    op.create_index(op.f('ix_workflow_dependencies_target_process_id'), 'workflow_dependencies', ['target_process_id'], unique=False)

    # Create decision_trees table
    op.create_table('decision_trees',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('process_id', sa.String(length=100), nullable=False),
        sa.Column('decision_point', sa.String(length=255), nullable=False),
        sa.Column('decision_type', sa.String(length=50), nullable=False),
        sa.Column('conditions', sa.JSON(), nullable=False),
        sa.Column('outcomes', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['process_id'], ['knowledge_items.process_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_decision_trees_decision_type'), 'decision_trees', ['decision_type'], unique=False)
    op.create_index(op.f('ix_decision_trees_id'), 'decision_trees', ['id'], unique=False)
    op.create_index(op.f('ix_decision_trees_process_id'), 'decision_trees', ['process_id'], unique=False)

    # Create optimization_patterns table
    op.create_table('optimization_patterns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('domain', sa.String(length=50), nullable=False),
        sa.Column('conditions', sa.JSON(), nullable=False),
        sa.Column('improvements', sa.JSON(), nullable=False),
        sa.Column('success_metrics', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('impact_level', sa.String(length=20), nullable=False),
        sa.Column('implementation_complexity', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_optimization_patterns_domain'), 'optimization_patterns', ['domain'], unique=False)
    op.create_index(op.f('ix_optimization_patterns_id'), 'optimization_patterns', ['id'], unique=False)
    op.create_index(op.f('ix_optimization_patterns_pattern_type'), 'optimization_patterns', ['pattern_type'], unique=False)

    # Create optimization_pattern_applications table
    op.create_table('optimization_pattern_applications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pattern_id', sa.Integer(), nullable=False),
        sa.Column('process_id', sa.String(length=100), nullable=False),
        sa.Column('applicability_score', sa.Float(), nullable=False),
        sa.Column('expected_impact', sa.JSON(), nullable=True),
        sa.Column('implementation_notes', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['pattern_id'], ['optimization_patterns.id'], ),
        sa.ForeignKeyConstraint(['process_id'], ['knowledge_items.process_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_optimization_pattern_applications_id'), 'optimization_pattern_applications', ['id'], unique=False)
    op.create_index(op.f('ix_optimization_pattern_applications_pattern_id'), 'optimization_pattern_applications', ['pattern_id'], unique=False)
    op.create_index(op.f('ix_optimization_pattern_applications_process_id'), 'optimization_pattern_applications', ['process_id'], unique=False)

    # Create communication_flows table
    op.create_table('communication_flows',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_role', sa.String(length=100), nullable=False),
        sa.Column('target_role', sa.String(length=100), nullable=False),
        sa.Column('information_type', sa.String(length=100), nullable=False),
        sa.Column('communication_method', sa.String(length=50), nullable=False),
        sa.Column('frequency', sa.String(length=50), nullable=False),
        sa.Column('criticality', sa.String(length=20), nullable=False),
        sa.Column('formal_protocol', sa.Boolean(), nullable=False),
        sa.Column('process_context', sa.String(length=100), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['process_context'], ['knowledge_items.process_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_communication_flows_id'), 'communication_flows', ['id'], unique=False)
    op.create_index(op.f('ix_communication_flows_information_type'), 'communication_flows', ['information_type'], unique=False)
    op.create_index(op.f('ix_communication_flows_process_context'), 'communication_flows', ['process_context'], unique=False)
    op.create_index(op.f('ix_communication_flows_source_role'), 'communication_flows', ['source_role'], unique=False)
    op.create_index(op.f('ix_communication_flows_target_role'), 'communication_flows', ['target_role'], unique=False)

    # Create knowledge_relationships table
    op.create_table('knowledge_relationships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.String(length=100), nullable=False),
        sa.Column('target_id', sa.String(length=100), nullable=False),
        sa.Column('relationship_type', sa.String(length=50), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('bidirectional', sa.Boolean(), nullable=False),
        sa.Column('relationship_metadata', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['knowledge_items.process_id'], ),
        sa.ForeignKeyConstraint(['target_id'], ['knowledge_items.process_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_relationships_id'), 'knowledge_relationships', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_relationship_type'), 'knowledge_relationships', ['relationship_type'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_source_id'), 'knowledge_relationships', ['source_id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_target_id'), 'knowledge_relationships', ['target_id'], unique=False)

    # Create knowledge_gaps table
    op.create_table('knowledge_gaps',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gap_type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('domain', sa.String(length=50), nullable=False),
        sa.Column('affected_processes', sa.JSON(), nullable=True),
        sa.Column('impact_assessment', sa.JSON(), nullable=True),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('assigned_to', sa.String(length=100), nullable=True),
        sa.Column('due_date', sa.DateTime(), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('identified_at', sa.DateTime(), nullable=False),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_gaps_domain'), 'knowledge_gaps', ['domain'], unique=False)
    op.create_index(op.f('ix_knowledge_gaps_gap_type'), 'knowledge_gaps', ['gap_type'], unique=False)
    op.create_index(op.f('ix_knowledge_gaps_id'), 'knowledge_gaps', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_gaps_identified_at'), 'knowledge_gaps', ['identified_at'], unique=False)
    op.create_index(op.f('ix_knowledge_gaps_priority'), 'knowledge_gaps', ['priority'], unique=False)
    op.create_index(op.f('ix_knowledge_gaps_status'), 'knowledge_gaps', ['status'], unique=False)

    # Create knowledge_gap_evidence table
    op.create_table('knowledge_gap_evidence',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gap_id', sa.Integer(), nullable=False),
        sa.Column('evidence_type', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('source_document_id', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['gap_id'], ['knowledge_gaps.id'], ),
        sa.ForeignKeyConstraint(['source_document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_gap_evidence_gap_id'), 'knowledge_gap_evidence', ['gap_id'], unique=False)
    op.create_index(op.f('ix_knowledge_gap_evidence_id'), 'knowledge_gap_evidence', ['id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_index(op.f('ix_knowledge_gap_evidence_id'), table_name='knowledge_gap_evidence')
    op.drop_index(op.f('ix_knowledge_gap_evidence_gap_id'), table_name='knowledge_gap_evidence')
    op.drop_table('knowledge_gap_evidence')
    
    op.drop_index(op.f('ix_knowledge_gaps_status'), table_name='knowledge_gaps')
    op.drop_index(op.f('ix_knowledge_gaps_priority'), table_name='knowledge_gaps')
    op.drop_index(op.f('ix_knowledge_gaps_identified_at'), table_name='knowledge_gaps')
    op.drop_index(op.f('ix_knowledge_gaps_id'), table_name='knowledge_gaps')
    op.drop_index(op.f('ix_knowledge_gaps_gap_type'), table_name='knowledge_gaps')
    op.drop_index(op.f('ix_knowledge_gaps_domain'), table_name='knowledge_gaps')
    op.drop_table('knowledge_gaps')
    
    op.drop_index(op.f('ix_knowledge_relationships_target_id'), table_name='knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_relationships_source_id'), table_name='knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_relationships_relationship_type'), table_name='knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_relationships_id'), table_name='knowledge_relationships')
    op.drop_table('knowledge_relationships')
    
    op.drop_index(op.f('ix_communication_flows_target_role'), table_name='communication_flows')
    op.drop_index(op.f('ix_communication_flows_source_role'), table_name='communication_flows')
    op.drop_index(op.f('ix_communication_flows_process_context'), table_name='communication_flows')
    op.drop_index(op.f('ix_communication_flows_information_type'), table_name='communication_flows')
    op.drop_index(op.f('ix_communication_flows_id'), table_name='communication_flows')
    op.drop_table('communication_flows')
    
    op.drop_index(op.f('ix_optimization_pattern_applications_process_id'), table_name='optimization_pattern_applications')
    op.drop_index(op.f('ix_optimization_pattern_applications_pattern_id'), table_name='optimization_pattern_applications')
    op.drop_index(op.f('ix_optimization_pattern_applications_id'), table_name='optimization_pattern_applications')
    op.drop_table('optimization_pattern_applications')
    
    op.drop_index(op.f('ix_optimization_patterns_pattern_type'), table_name='optimization_patterns')
    op.drop_index(op.f('ix_optimization_patterns_id'), table_name='optimization_patterns')
    op.drop_index(op.f('ix_optimization_patterns_domain'), table_name='optimization_patterns')
    op.drop_table('optimization_patterns')
    
    op.drop_index(op.f('ix_decision_trees_process_id'), table_name='decision_trees')
    op.drop_index(op.f('ix_decision_trees_id'), table_name='decision_trees')
    op.drop_index(op.f('ix_decision_trees_decision_type'), table_name='decision_trees')
    op.drop_table('decision_trees')
    
    op.drop_index(op.f('ix_workflow_dependencies_target_process_id'), table_name='workflow_dependencies')
    op.drop_index(op.f('ix_workflow_dependencies_source_process_id'), table_name='workflow_dependencies')
    op.drop_index(op.f('ix_workflow_dependencies_id'), table_name='workflow_dependencies')
    op.drop_index(op.f('ix_workflow_dependencies_dependency_type'), table_name='workflow_dependencies')
    op.drop_table('workflow_dependencies')
    
    op.drop_index(op.f('ix_knowledge_items_updated_at'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_process_id'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_name'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_knowledge_type'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_id'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_hierarchy_level'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_domain'), table_name='knowledge_items')
    op.drop_index(op.f('ix_knowledge_items_created_at'), table_name='knowledge_items')
    op.drop_table('knowledge_items')
