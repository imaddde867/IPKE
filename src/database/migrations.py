"""
Database migration system for schema changes.
Provides utilities for managing database schema evolution.
"""

from sqlalchemy import text, inspect, MetaData, Table, Column, Integer, String, DateTime, Boolean
from sqlalchemy.exc import SQLAlchemyError
from src.database.database import engine, get_db_session
from src.database.models import Base
from src.logging_config import get_logger
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

logger = get_logger(__name__)

class MigrationError(Exception):
    """Raised when migration operations fail."""
    pass

class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, description: str, up_sql: str, down_sql: str = None):
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.applied_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary for storage."""
        return {
            "version": self.version,
            "description": self.description,
            "up_sql": self.up_sql,
            "down_sql": self.down_sql,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        """Create migration from dictionary."""
        migration = cls(
            version=data["version"],
            description=data["description"],
            up_sql=data["up_sql"],
            down_sql=data.get("down_sql")
        )
        if data.get("applied_at"):
            migration.applied_at = datetime.fromisoformat(data["applied_at"])
        return migration

class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self):
        self.migrations_table = "schema_migrations"
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if it doesn't exist."""
        try:
            with get_db_session() as db:
                # Check if migrations table exists
                inspector = inspect(engine)
                if not inspector.has_table(self.migrations_table):
                    logger.info("Creating schema_migrations table")
                    create_table_sql = f"""
                    CREATE TABLE {self.migrations_table} (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(50) UNIQUE NOT NULL,
                        description TEXT NOT NULL,
                        up_sql TEXT NOT NULL,
                        down_sql TEXT,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64)
                    )
                    """
                    db.execute(text(create_table_sql))
                    logger.info("Schema migrations table created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise MigrationError(f"Failed to create migrations table: {e}")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            with get_db_session() as db:
                result = db.execute(
                    text(f"SELECT version FROM {self.migrations_table} ORDER BY applied_at")
                )
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get applied migrations: {e}")
            raise MigrationError(f"Failed to get applied migrations: {e}")
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration version has been applied."""
        applied_migrations = self.get_applied_migrations()
        return version in applied_migrations
    
    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        if self.is_migration_applied(migration.version):
            logger.info(f"Migration {migration.version} already applied, skipping")
            return False
        
        try:
            with get_db_session() as db:
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                
                # Execute the migration SQL
                if migration.up_sql.strip():
                    for statement in migration.up_sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            db.execute(text(statement))
                
                # Record the migration as applied
                insert_sql = f"""
                INSERT INTO {self.migrations_table} 
                (version, description, up_sql, down_sql, applied_at) 
                VALUES (:version, :description, :up_sql, :down_sql, :applied_at)
                """
                db.execute(text(insert_sql), {
                    "version": migration.version,
                    "description": migration.description,
                    "up_sql": migration.up_sql,
                    "down_sql": migration.down_sql,
                    "applied_at": datetime.utcnow()
                })
                
                logger.info(f"Migration {migration.version} applied successfully")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            raise MigrationError(f"Failed to apply migration {migration.version}: {e}")
    
    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        if not self.is_migration_applied(version):
            logger.info(f"Migration {version} not applied, nothing to rollback")
            return False
        
        try:
            with get_db_session() as db:
                # Get migration details
                result = db.execute(
                    text(f"SELECT down_sql FROM {self.migrations_table} WHERE version = :version"),
                    {"version": version}
                )
                row = result.fetchone()
                
                if not row or not row[0]:
                    raise MigrationError(f"No rollback SQL found for migration {version}")
                
                down_sql = row[0]
                logger.info(f"Rolling back migration {version}")
                
                # Execute rollback SQL
                for statement in down_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        db.execute(text(statement))
                
                # Remove migration record
                db.execute(
                    text(f"DELETE FROM {self.migrations_table} WHERE version = :version"),
                    {"version": version}
                )
                
                logger.info(f"Migration {version} rolled back successfully")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            raise MigrationError(f"Failed to rollback migration {version}: {e}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            applied_migrations = self.get_applied_migrations()
            
            with get_db_session() as db:
                # Get database schema info
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                
                return {
                    "applied_migrations": applied_migrations,
                    "total_applied": len(applied_migrations),
                    "database_tables": tables,
                    "total_tables": len(tables),
                    "migrations_table_exists": self.migrations_table in tables
                }
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {"error": str(e)}

# Pre-defined migrations for common schema changes
INITIAL_MIGRATIONS = [
    Migration(
        version="001_add_processing_logs",
        description="Add processing logs table for tracking processing steps and errors",
        up_sql="""
        CREATE TABLE IF NOT EXISTS processing_logs (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            step_name VARCHAR(100) NOT NULL,
            status VARCHAR(20) NOT NULL,
            start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds FLOAT,
            error_message TEXT,
            metadata_json JSON,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_processing_logs_document_id ON processing_logs(document_id);
        CREATE INDEX IF NOT EXISTS idx_processing_logs_status ON processing_logs(status);
        CREATE INDEX IF NOT EXISTS idx_processing_logs_step_name ON processing_logs(step_name);
        """,
        down_sql="""
        DROP INDEX IF EXISTS idx_processing_logs_step_name;
        DROP INDEX IF EXISTS idx_processing_logs_status;
        DROP INDEX IF EXISTS idx_processing_logs_document_id;
        DROP TABLE IF EXISTS processing_logs;
        """
    ),
    Migration(
        version="002_add_system_health",
        description="Add system health table for storing health check results and metrics",
        up_sql="""
        CREATE TABLE IF NOT EXISTS system_health (
            id SERIAL PRIMARY KEY,
            service_name VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            response_time_ms FLOAT,
            error_message TEXT,
            metadata_json JSON,
            checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_system_health_service_name ON system_health(service_name);
        CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);
        CREATE INDEX IF NOT EXISTS idx_system_health_checked_at ON system_health(checked_at);
        """,
        down_sql="""
        DROP INDEX IF EXISTS idx_system_health_checked_at;
        DROP INDEX IF EXISTS idx_system_health_status;
        DROP INDEX IF EXISTS idx_system_health_service_name;
        DROP TABLE IF EXISTS system_health;
        """
    ),
    Migration(
        version="003_add_document_processing_fields",
        description="Add processing duration and retry fields to documents table",
        up_sql="""
        ALTER TABLE documents 
        ADD COLUMN IF NOT EXISTS processing_duration_seconds FLOAT,
        ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
        ADD COLUMN IF NOT EXISTS last_error TEXT,
        ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMP,
        ADD COLUMN IF NOT EXISTS processing_completed_at TIMESTAMP;
        
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_documents_retry_count ON documents(retry_count);
        """,
        down_sql="""
        DROP INDEX IF EXISTS idx_documents_retry_count;
        DROP INDEX IF EXISTS idx_documents_status;
        ALTER TABLE documents 
        DROP COLUMN IF EXISTS processing_completed_at,
        DROP COLUMN IF EXISTS processing_started_at,
        DROP COLUMN IF EXISTS last_error,
        DROP COLUMN IF EXISTS retry_count,
        DROP COLUMN IF EXISTS processing_duration_seconds;
        """
    )
]

# Global migration manager instance (lazy initialization)
_migration_manager = None

def get_migration_manager():
    """Get or create the migration manager instance."""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager

def run_migrations():
    """Run all pending migrations."""
    logger.info("Starting database migrations")
    applied_count = 0
    
    migration_manager = get_migration_manager()
    for migration in INITIAL_MIGRATIONS:
        if migration_manager.apply_migration(migration):
            applied_count += 1
    
    logger.info(f"Database migrations completed. Applied {applied_count} new migrations.")
    return applied_count

def get_migration_status():
    """Get current migration status."""
    migration_manager = get_migration_manager()
    return migration_manager.get_migration_status()