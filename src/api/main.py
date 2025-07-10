from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.celery_worker import process_document_task
from src.database.database import get_db, init_db
from src.database import models, crud
from src.search.elasticsearch_client import es_client
from sqlalchemy.orm import Session
import os

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Allow requests from your React frontend
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize database tables on startup
@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Industrial Knowledge Extraction System!"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    task = process_document_task.delay(file_location)
    return {"info": f"file '{file.filename}' saved at '{file_location}'. Task ID: {task.id}"}

@app.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.get("/documents/")
def get_all_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    documents = db.query(models.Document).offset(skip).limit(limit).all()
    return documents

@app.get("/documents/{document_id}/entities/")
def get_document_entities(document_id: int, db: Session = Depends(get_db)):
    entities = db.query(models.ExtractedEntity).filter(models.ExtractedEntity.document_id == document_id).all()
    if not entities:
        raise HTTPException(status_code=404, detail="Entities not found for this document")
    return entities

@app.get("/documents/{document_id}/keyphrases/")
def get_document_keyphrases(document_id: int, db: Session = Depends(get_db)):
    key_phrases = db.query(models.KeyPhrase).filter(models.KeyPhrase.document_id == document_id).all()
    if not key_phrases:
        raise HTTPException(status_code=404, detail="Key phrases not found for this document")
    return key_phrases

@app.get("/documents/{document_id}/equipment/")
def get_document_equipment(document_id: int, db: Session = Depends(get_db)):
    equipment = db.query(models.Equipment).filter(models.Equipment.document_id == document_id).all()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment data not found for this document")
    return equipment

@app.get("/documents/{document_id}/procedures/")
def get_document_procedures(document_id: int, db: Session = Depends(get_db)):
    procedures = db.query(models.Procedure).filter(models.Procedure.document_id == document_id).all()
    if not procedures:
        raise HTTPException(status_code=404, detail="Procedure data not found for this document")
    return procedures

@app.get("/documents/{document_id}/safety_info/")
def get_document_safety_info(document_id: int, db: Session = Depends(get_db)):
    safety_info = db.query(models.SafetyInformation).filter(models.SafetyInformation.document_id == document_id).all()
    if not safety_info:
        raise HTTPException(status_code=404, detail="Safety information not found for this document")
    return safety_info

@app.get("/documents/{document_id}/technical_specs/")
def get_document_technical_specs(document_id: int, db: Session = Depends(get_db)):
    technical_specs = db.query(models.TechnicalSpecification).filter(models.TechnicalSpecification.document_id == document_id).all()
    if not technical_specs:
        raise HTTPException(status_code=404, detail="Technical specifications not found for this document")
    return technical_specs

@app.get("/documents/{document_id}/personnel/")
def get_document_personnel(document_id: int, db: Session = Depends(get_db)):
    personnel = db.query(models.Personnel).filter(models.Personnel.document_id == document_id).all()
    if not personnel:
        raise HTTPException(status_code=404, detail="Personnel data not found for this document")
    return personnel

@app.get("/documents/{document_id}/sections/")
def get_document_sections(document_id: int, db: Session = Depends(get_db)):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document.document_sections

@app.get("/search/")
def search_documents(query: str, field: str = "extracted_text", size: int = 10):
    if field not in ["extracted_text", "filename", "classification_category", "extracted_entities.text", "key_phrases", "document_sections"]:
        raise HTTPException(status_code=400, detail="Invalid search field")
    
    results = es_client.search_documents(query=query, field=field, size=size)
    return results
