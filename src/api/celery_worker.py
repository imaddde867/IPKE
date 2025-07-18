from celery import Celery
from src.database.database import SessionLocal, init_db
from src.database import crud
from src.search.elasticsearch_client import es_client

celery_app = Celery(
    'knowledge_extraction',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

# Initialize database tables when Celery worker starts
@celery_app.on_after_configure.connect
def setup_database(sender, **kwargs):
    init_db()

@celery_app.task
def process_document_task(file_path: str):
    # Import document processor functions only when needed
    from src.processors.document_processor import process_document, process_video, get_file_type
    
    print(f"Received file for processing: {file_path}")
    file_type = get_file_type(file_path)
    
    db = SessionLocal()
    try:
        if file_type == "video":
            result = process_video(file_path)
        else:
            result = process_document(file_path)

        # Save processed data to database
        db_document = crud.create_document(
            db=db,
            filename=result["filename"],
            file_type=file_type,
            extracted_text=result["extracted_text"],
            metadata_json=result["metadata"],
            classification_category=result["classification"]["category"],
            classification_score=result["classification"]["score"],
            status=result["status"],
            document_sections=result["document_sections"]
        )

        # Save extracted entities
        es_entities = []
        for entity in result["extracted_entities"]:
            crud.create_extracted_entity(
                db=db,
                document_id=db_document.id,
                text=entity["word"],
                entity_type=entity["entity_group"],
                score=entity["score"],
                start_char=entity["start"],
                end_char=entity["end"]
            )
            es_entities.append({"text": entity["word"], "entity_type": entity["entity_group"]})
        
        # Save key phrases
        es_key_phrases = []
        for phrase in result["key_phrases"]:
            crud.create_key_phrase(
                db=db,
                document_id=db_document.id,
                phrase=phrase
            )
            es_key_phrases.append(phrase)

        # Save structured data
        for equipment in result["equipment_data"]:
            crud.create_equipment(
                db=db,
                document_id=db_document.id,
                name=equipment["name"],
                type=equipment["type"],
                specifications=equipment["specifications"],
                location=equipment["location"],
                confidence=equipment["confidence"]
            )
        
        for procedure in result["procedure_data"]:
            crud.create_procedure(
                db=db,
                document_id=db_document.id,
                title=procedure["title"],
                steps=procedure["steps"],
                category=procedure["category"],
                confidence=procedure["confidence"]
            )

        for safety_info in result["safety_info_data"]:
            crud.create_safety_information(
                db=db,
                document_id=db_document.id,
                hazard=safety_info["hazard"],
                precaution=safety_info["precaution"],
                ppe_required=safety_info["ppe_required"],
                severity=safety_info["severity"],
                confidence=safety_info["confidence"]
            )

        for tech_spec in result["technical_spec_data"]:
            crud.create_technical_specification(
                db=db,
                document_id=db_document.id,
                parameter=tech_spec["parameter"],
                value=tech_spec["value"],
                unit=tech_spec["unit"],
                tolerance=tech_spec["tolerance"],
                confidence=tech_spec["confidence"]
            )

        for personnel in result["personnel_data"]:
            crud.create_personnel(
                db=db,
                document_id=db_document.id,
                name=personnel["name"],
                role=personnel["role"],
                responsibilities=personnel["responsibilities"],
                certifications=personnel["certifications"],
                confidence=personnel["confidence"]
            )

        # Index document in Elasticsearch
        es_document_data = {
            "document_id": db_document.id,
            "filename": db_document.filename,
            "file_type": db_document.file_type,
            "extracted_text": db_document.extracted_text,
            "classification_category": db_document.classification_category,
            "classification_score": db_document.classification_score,
            "extracted_entities": es_entities,
            "key_phrases": es_key_phrases,
            "processing_timestamp": db_document.processing_timestamp.isoformat(),
            "document_sections": db_document.document_sections # Include sections in ES
        }
        es_client.index_document(es_document_data)

        return {"status": "success", "document_id": db_document.id}

    except Exception as e:
        db.rollback()
        print(f"Error processing document {file_path}: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()
