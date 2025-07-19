from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from src.database.models import Base
from src.config import config_manager
from src.logging_config import get_logger
import time

logger = get_logger(__name__)

# Get database URL from configuration
DATABASE_URL = config_manager.get_database_url()

# Enhanced engine configuration with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=config_manager.config.database.pool_size,
    max_overflow=config_manager.config.database.max_overflow,
    pool_pre_ping=config_manager.config.database.pool_pre_ping,
    pool_recycle=config_manager.config.database.pool_recycle,
    echo=config_manager.config.debug,  # Log SQL statements in debug mode
    connect_args={
        "connect_timeout": config_manager.config.database.connect_timeout,
        "application_name": "knowledge_extraction_system"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Add connection event listeners for monitoring
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set connection-level settings for PostgreSQL."""
    logger.debug("New database connection established")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log when a connection is checked out from the pool."""
    logger.debug("Connection checked out from pool")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log when a connection is returned to the pool."""
    logger.debug("Connection returned to pool")

def get_db():
    """Dependency to get database session with proper cleanup."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic transaction management."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Create database tables if they don't exist."""
    try:
        logger.info("Initializing database tables")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_connection_pool_status():
    """Get current connection pool status for monitoring."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid()
    }
