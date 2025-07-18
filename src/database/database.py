from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base

# Database connection string
# This should ideally come from environment variables for production
# DATABASE_URL = "postgresql://user:password@db:5432/knowledge_db"
DATABASE_URL = "postgresql://user:password@db:5432/knowledge_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
