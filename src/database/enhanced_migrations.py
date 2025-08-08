"""
Enhanced Database Migrations for EXPLAINIUM

This module provides migration scripts to upgrade the database schema
to support comprehensive organizational knowledge management features.
"""

import logging
from typing import List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from alembic import command
from alembic.config import Config

from src.config import config_manager
from src.logging_config import get_logger
from src.database.models import Base

logger = get_logger(__name__)

class EnhancedMigrationManager:
    """Manages database migrations for enhanced features"""
    
    def __init__(self):
        self.database_url = config_manager.get_database_url()
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def run_enhanced_migrations(self):
        """Run all enhanced migrations"""
        logger.info("Starting enhanced database migrations")
        
        try:
            # Create all new tables
            Base.metadata.create_all(bind=self.engine)
            
            # Run custom migrations
            self._add_enum_types()
            self._add_new_columns_to_existing_tables()
            self._create_association_tables()
            self._add_indexes_for_performance()
            self._populate_default_data()
            
            logger.info("Enhanced database migrations completed successfully")
            
        except Exception as e:
            logger.error(f"Error during enhanced migrations: {e}")
            raise
    
    def _add_enum_types(self):
        """Add custom enum types to the database"""
        logger.info("Adding custom enum types")
        
        enum_definitions = [
            """
            DO $$ BEGIN
                CREATE TYPE knowledge_domain AS ENUM (
                    'operational', 'safety_compliance', 'equipment_technology',
                    'human_resources', 'quality_assurance', 'environmental',
                    'financial', 'regulatory', 'maintenance', 'training'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """,
            """
            DO $$ BEGIN
                CREATE TYPE hierarchy_level AS ENUM (
                    '1', '2', '3', '4'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """,
            """
            DO $$ BEGIN
                CREATE TYPE criticality_level AS ENUM (
                    'critical', 'high', 'medium', 'low'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """,
            """
            DO $$ BEGIN
                CREATE TYPE compliance_status AS ENUM (
                    'compliant', 'non_compliant', 'under_review', 'not_applicable'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """,
            """
            DO $$ BEGIN
                CREATE TYPE risk_level AS ENUM (
                    'very_high', 'high', 'medium', 'low', 'very_low'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """
        ]
        
        with self.engine.connect() as conn:
            for enum_def in enum_definitions:
                try:
                    conn.execute(text(enum_def))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Enum creation warning: {e}")
    
    def _add_new_columns_to_existing_tables(self):
        """Add new columns to existing tables"""
        logger.info("Adding new columns to existing tables")
        
        column_additions = [
            # Documents table enhancements
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_quality_score FLOAT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_completeness_score FLOAT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS knowledge_domains JSON",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS regulatory_references JSON",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS version_info JSON",
            
            # Equipment table enhancements
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS manufacturer VARCHAR",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS model_number VARCHAR",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS serial_number VARCHAR",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS installation_date TIMESTAMP",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS maintenance_schedule JSON",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS operational_parameters JSON",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS safety_requirements JSON",
            "ALTER TABLE equipment ADD COLUMN IF NOT EXISTS criticality_level criticality_level DEFAULT 'medium'",
            
            # Personnel table enhancements
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS department VARCHAR",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS supervisor VARCHAR",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS contact_information JSON",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS skill_level VARCHAR",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS training_records JSON",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS authorization_levels JSON",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS shift_schedule JSON",
            "ALTER TABLE personnel ADD COLUMN IF NOT EXISTS emergency_contact JSON",
            
            # Safety information enhancements
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS hazard_category VARCHAR",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS affected_areas JSON",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS emergency_procedures TEXT",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS regulatory_references JSON",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS training_requirements JSON",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS inspection_frequency VARCHAR",
            "ALTER TABLE safety_information ADD COLUMN IF NOT EXISTS responsible_party VARCHAR"
        ]
        
        with self.engine.connect() as conn:
            for sql in column_additions:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Column addition warning: {e}")
    
    def _create_association_tables(self):
        """Create association tables for many-to-many relationships"""
        logger.info("Creating association tables")
        
        association_tables = [
            """
            CREATE TABLE IF NOT EXISTS process_equipment_association (
                process_id INTEGER REFERENCES processes(id) ON DELETE CASCADE,
                equipment_id INTEGER REFERENCES equipment(id) ON DELETE CASCADE,
                PRIMARY KEY (process_id, equipment_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS process_personnel_association (
                process_id INTEGER REFERENCES processes(id) ON DELETE CASCADE,
                personnel_id INTEGER REFERENCES personnel(id) ON DELETE CASCADE,
                PRIMARY KEY (process_id, personnel_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS process_safety_association (
                process_id INTEGER REFERENCES processes(id) ON DELETE CASCADE,
                safety_info_id INTEGER REFERENCES safety_information(id) ON DELETE CASCADE,
                PRIMARY KEY (process_id, safety_info_id)
            )
            """
        ]
        
        with self.engine.connect() as conn:
            for table_sql in association_tables:
                try:
                    conn.execute(text(table_sql))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Association table creation warning: {e}")
    
    def _add_indexes_for_performance(self):
        """Add indexes for better query performance"""
        logger.info("Adding performance indexes")
        
        indexes = [
            # Process indexes
            "CREATE INDEX IF NOT EXISTS idx_processes_domain ON processes(knowledge_domain)",
            "CREATE INDEX IF NOT EXISTS idx_processes_hierarchy ON processes(hierarchy_level)",
            "CREATE INDEX IF NOT EXISTS idx_processes_criticality ON processes(criticality_level)",
            "CREATE INDEX IF NOT EXISTS idx_processes_confidence ON processes(confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_processes_name ON processes(name)",
            "CREATE INDEX IF NOT EXISTS idx_processes_updated ON processes(last_updated)",
            
            # Compliance indexes
            "CREATE INDEX IF NOT EXISTS idx_compliance_regulation ON compliance_items(regulation_name)",
            "CREATE INDEX IF NOT EXISTS idx_compliance_status ON compliance_items(compliance_status)",
            "CREATE INDEX IF NOT EXISTS idx_compliance_responsible ON compliance_items(responsible_party)",
            "CREATE INDEX IF NOT EXISTS idx_compliance_review_date ON compliance_items(next_review_date)",
            
            # Risk assessment indexes
            "CREATE INDEX IF NOT EXISTS idx_risks_category ON risk_assessments(risk_category)",
            "CREATE INDEX IF NOT EXISTS idx_risks_level ON risk_assessments(overall_risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_risks_likelihood ON risk_assessments(likelihood)",
            "CREATE INDEX IF NOT EXISTS idx_risks_impact ON risk_assessments(impact)",
            "CREATE INDEX IF NOT EXISTS idx_risks_responsible ON risk_assessments(responsible_party)",
            
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_quality ON documents(source_quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_documents_completeness ON documents(content_completeness_score)",
            "CREATE INDEX IF NOT EXISTS idx_documents_processing_time ON documents(processing_timestamp)",
            
            # Full-text search indexes
            "CREATE INDEX IF NOT EXISTS idx_processes_text_search ON processes USING gin(to_tsvector('english', name || ' ' || description))",
            "CREATE INDEX IF NOT EXISTS idx_compliance_text_search ON compliance_items USING gin(to_tsvector('english', regulation_name || ' ' || requirement_description))",
            "CREATE INDEX IF NOT EXISTS idx_risks_text_search ON risk_assessments USING gin(to_tsvector('english', risk_description))"
        ]
        
        with self.engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
    
    def _populate_default_data(self):
        """Populate default data for enhanced features"""
        logger.info("Populating default data")
        
        try:
            session = self.SessionLocal()
            
            # Add any default data here if needed
            # For example, default process templates, compliance frameworks, etc.
            
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error populating default data: {e}")
    
    def backup_database(self, backup_path: str):
        """Create a database backup before migration"""
        logger.info(f"Creating database backup at {backup_path}")
        
        try:
            import subprocess
            import os
            
            # Extract database connection details
            db_config = config_manager.get_database_config()
            
            # Create backup using pg_dump
            backup_cmd = [
                "pg_dump",
                "-h", db_config.get("host", "localhost"),
                "-p", str(db_config.get("port", 5432)),
                "-U", db_config.get("user", "postgres"),
                "-d", db_config.get("database", "explainium"),
                "-f", backup_path,
                "--no-password"
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config.get("password", "")
            
            result = subprocess.run(backup_cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database backup created successfully")
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
    
    def validate_migration(self):
        """Validate that the migration was successful"""
        logger.info("Validating migration")
        
        validation_queries = [
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'processes'",
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'compliance_items'",
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'risk_assessments'",
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'process_steps'",
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'decision_points'",
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'documents' AND column_name = 'source_quality_score'",
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'equipment' AND column_name = 'criticality_level'"
        ]
        
        try:
            with self.engine.connect() as conn:
                for query in validation_queries:
                    result = conn.execute(text(query))
                    count = result.scalar()
                    logger.info(f"Validation query result: {count}")
                    
                    if count == 0:
                        logger.warning(f"Validation failed for query: {query}")
                        return False
            
            logger.info("Migration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

def run_enhanced_migrations():
    """Main function to run enhanced migrations"""
    migration_manager = EnhancedMigrationManager()
    
    try:
        # Create backup before migration
        backup_path = f"backup_pre_enhanced_migration_{int(datetime.utcnow().timestamp())}.sql"
        migration_manager.backup_database(backup_path)
        
        # Run migrations
        migration_manager.run_enhanced_migrations()
        
        # Validate migration
        if migration_manager.validate_migration():
            logger.info("Enhanced migrations completed successfully")
        else:
            logger.error("Migration validation failed")
            
    except Exception as e:
        logger.error(f"Enhanced migration failed: {e}")
        raise

if __name__ == "__main__":
    from datetime import datetime
    run_enhanced_migrations()
