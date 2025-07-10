from fastapi.testclient import TestClient
from src.api.main import app
from src.database.database import get_db, SessionLocal, Base, engine
from src.database.models import Document, ExtractedEntity, KeyPhrase

# Override the get_db dependency to use a test database
def override_get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Setup and Teardown for tests
def setup_function():
    # Create tables for testing
    Base.metadata.create_all(bind=engine)

def teardown_function():
    # Drop tables after testing
    Base.metadata.drop_all(bind=engine)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Industrial Knowledge Extraction System!"}

# Note: File upload and Celery task testing would be more complex
# and might require mocking Celery tasks or running a test Celery worker.
# For now, we'll focus on basic API endpoint functionality.

def test_get_all_documents_empty():
    response = client.get("/documents/")
    assert response.status_code == 200
    assert response.json() == []

def test_get_document_not_found():
    response = client.get("/documents/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}

# To test document creation and retrieval, we would need to mock the Celery task
# or directly call the crud functions, which is beyond a simple API test.
# For now, these tests serve as a basic setup.
