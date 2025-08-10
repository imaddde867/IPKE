"""
EXPLAINIUM - Consolidated Database System

A clean, professional database management system that consolidates
all database functionality into a single, well-organized module.
"""

import logging
from typing import Generator, Dict, Any, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import config
import os
import socket

logger = logging.getLogger(__name__)

def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

db_url = config.database.url
using_sqlite_fallback = False
if db_url.startswith("postgresql") and not _can_connect(config.database.host, config.database.port):
    # Fallback to local sqlite for developer manual testing
    fallback_path = os.getenv("SQLITE_FALLBACK_PATH", "dev_fallback.db")
    db_url = f"sqlite:///{fallback_path}"
    using_sqlite_fallback = True
    logging.getLogger(__name__).warning(
        f"Postgres not reachable at {config.database.host}:{config.database.port}; using SQLite fallback {fallback_path}" )

# Create engine with appropriate parameters based on database type
if db_url.startswith("sqlite"):
    engine = create_engine(
        db_url,
        echo=config.database.echo,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        db_url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        echo=config.database.echo
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Reuse models' declarative base so metadata contains actual table definitions
try:
    from src.database.models import Base as ModelsBase  # type: ignore
    Base = ModelsBase
except Exception:
    # Fallback (should not normally happen)
    Base = declarative_base()


class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.Base = Base
    
    def create_tables(self):
        """Create all database tables"""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            self.Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def check_connection(self) -> bool:
        """Check database connection"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT version()"))
                version = result.scalar()
                
                # Get connection pool status
                pool = self.engine.pool
                
                return {
                    "status": "connected",
                    "version": version,
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about database tables"""
        try:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            
            tables_info = {}
            for table_name, table in metadata.tables.items():
                tables_info[table_name] = {
                    "columns": [
                        {
                            "name": col.name,
                            "type": str(col.type),
                            "nullable": col.nullable,
                            "primary_key": col.primary_key,
                            "foreign_keys": [str(fk) for fk in col.foreign_keys]
                        }
                        for col in table.columns
                    ],
                    "indexes": [
                        {
                            "name": idx.name,
                            "columns": [col.name for col in idx.columns],
                            "unique": idx.unique
                        }
                        for idx in table.indexes
                    ]
                }
            
            return tables_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return {}


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session for FastAPI
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create tables"""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def reset_db():
    """Reset database - drop and recreate tables"""
    try:
        db_manager.drop_tables()
        db_manager.create_tables()
        logger.info("Database reset successfully")
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


def check_db_health() -> Dict[str, Any]:
    """Check database health and return status"""
    return db_manager.get_connection_info()


def get_db_session() -> Session:
    """Get a new database session (manual management)"""
    return SessionLocal()


@contextmanager
def db_transaction() -> Generator[Session, None, None]:
    """Context manager for database transactions"""
    with db_manager.get_session() as session:
        yield session


# Database utilities for testing
class TestDatabaseManager:
    """Database manager for testing with isolated transactions"""
    
    def __init__(self):
        self.connection = None
        self.transaction = None
        self.session = None
    
    def setup(self):
        """Setup test database session"""
        self.connection = engine.connect()
        self.transaction = self.connection.begin()
        self.session = sessionmaker(bind=self.connection)()
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.connection)
    
    def teardown(self):
        """Teardown test database session"""
        if self.session:
            self.session.close()
        if self.transaction:
            self.transaction.rollback()
        if self.connection:
            self.connection.close()
    
    def get_session(self) -> Session:
        """Get test session"""
        return self.session


# Migration utilities
class MigrationManager:
    """Simple migration management system"""
    
    def __init__(self):
        self.engine = engine
    
    def create_migration_table(self):
        """Create migration tracking table"""
        sql = """
        CREATE TABLE IF NOT EXISTS migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text(sql))
                connection.commit()
            logger.info("Migration table created")
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise
    
    def get_applied_migrations(self) -> list:
        """Get list of applied migrations"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text("SELECT version FROM migrations ORDER BY applied_at")
                )
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def apply_migration(self, version: str, description: str, sql: str):
        """Apply a migration"""
        try:
            with self.engine.connect() as connection:
                # Execute migration SQL
                connection.execute(text(sql))
                
                # Record migration
                connection.execute(
                    text("INSERT INTO migrations (version, description) VALUES (:version, :description)"),
                    {"version": version, "description": description}
                )
                connection.commit()
                
            logger.info(f"Applied migration {version}: {description}")
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            raise
    
    def rollback_migration(self, version: str, rollback_sql: str):
        """Rollback a migration"""
        try:
            with self.engine.connect() as connection:
                # Execute rollback SQL
                connection.execute(text(rollback_sql))
                
                # Remove migration record
                connection.execute(
                    text("DELETE FROM migrations WHERE version = :version"),
                    {"version": version}
                )
                connection.commit()
                
            logger.info(f"Rolled back migration {version}")
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            raise


# Global migration manager
migration_manager = MigrationManager()


def run_migrations():
    """Run database migrations"""
    try:
        migration_manager.create_migration_table()
        
        # Define migrations
        migrations = [
            {
                "version": "001",
                "description": "Create initial tables",
                "sql": """
                -- This will be handled by SQLAlchemy create_all
                SELECT 1;
                """
            },
            {
                "version": "002", 
                "description": "Add indexes for performance",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
                CREATE INDEX IF NOT EXISTS idx_processes_domain ON processes(domain);
                CREATE INDEX IF NOT EXISTS idx_processes_confidence ON processes(confidence);
                """
            }
        ]
        
        applied_migrations = migration_manager.get_applied_migrations()
        
        for migration in migrations:
            if migration["version"] not in applied_migrations:
                migration_manager.apply_migration(
                    migration["version"],
                    migration["description"],
                    migration["sql"]
                )
        
        logger.info("All migrations applied successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


# Database health check
def health_check() -> Dict[str, Any]:
    """Comprehensive database health check"""
    try:
        health_info = check_db_health()
        
        # Add additional health metrics
        with engine.connect() as connection:
            # Check if we can perform basic operations
            connection.execute(text("SELECT 1"))
            
            # Get database size (PostgreSQL specific)
            try:
                size_result = connection.execute(
                    text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                )
                health_info["database_size"] = size_result.scalar()
            except:
                health_info["database_size"] = "unknown"
            
            # Get active connections
            try:
                conn_result = connection.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                )
                health_info["active_connections"] = conn_result.scalar()
            except:
                health_info["active_connections"] = "unknown"
        
        health_info["status"] = "healthy"
        return health_info
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }