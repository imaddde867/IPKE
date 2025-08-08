"""
Enhanced FastAPI Main Application for EXPLAINIUM

This module provides comprehensive API endpoints for organizational knowledge
management including processes, hierarchies, compliance, risk assessments,
and advanced search capabilities.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

# Internal imports
from src.database.database import get_db
from src.database import models, crud
from src.database.models import (
    KnowledgeDomain, HierarchyLevel, CriticalityLevel,
    ComplianceStatus, RiskLevel
)
from src.processors.enhanced_document_processor import create_enhanced_processor
from src.logging_config import get_logger
from src.config import config_manager
from src.exceptions import ProcessingError, AIError, ServiceUnavailableError

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EXPLAINIUM - Enhanced Knowledge Extraction System",
    description="Advanced AI-powered system for extracting and structuring organizational knowledge",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "frontend" / "public"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize document processor
document_processor = create_enhanced_processor()

# Pydantic models for API requests/responses
class ProcessFilter(BaseModel):
    domain: Optional[KnowledgeDomain] = None
    hierarchy_level: Optional[HierarchyLevel] = None
    criticality_level: Optional[CriticalityLevel] = None
    confidence_threshold: Optional[float] = 0.0
    search_query: Optional[str] = None

class ComplianceFilter(BaseModel):
    regulation_name: Optional[str] = None
    status: Optional[ComplianceStatus] = None
    responsible_party: Optional[str] = None
    review_due_days: Optional[int] = None

class RiskFilter(BaseModel):
    category: Optional[str] = None
    likelihood: Optional[RiskLevel] = None
    impact: Optional[RiskLevel] = None
    overall_risk_level: Optional[RiskLevel] = None

class KnowledgeQuery(BaseModel):
    query: str
    domains: Optional[List[KnowledgeDomain]] = None
    hierarchy_levels: Optional[List[HierarchyLevel]] = None
    include_implicit: bool = True
    max_results: int = 50

class ProcessCreate(BaseModel):
    name: str
    description: str
    domain: KnowledgeDomain
    hierarchy_level: HierarchyLevel
    parent_process_id: Optional[str] = None
    estimated_duration: Optional[str] = None
    frequency: Optional[str] = None
    prerequisites: Optional[List[str]] = []
    success_criteria: Optional[List[str]] = []
    required_skills: Optional[List[str]] = []
    required_certifications: Optional[List[str]] = []
    quality_standards: Optional[List[str]] = []
    compliance_requirements: Optional[List[str]] = []
    criticality_level: CriticalityLevel = CriticalityLevel.MEDIUM

class ProcessUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    estimated_duration: Optional[str] = None
    frequency: Optional[str] = None
    prerequisites: Optional[List[str]] = None
    success_criteria: Optional[List[str]] = None
    criticality_level: Optional[CriticalityLevel] = None

class ComplianceItemCreate(BaseModel):
    regulation_name: str
    regulation_section: Optional[str] = None
    requirement_description: str
    responsible_party: str
    review_frequency: str = "Annual"
    evidence_location: Optional[str] = None
    notes: Optional[str] = None

class RiskAssessmentCreate(BaseModel):
    risk_category: str
    risk_description: str
    likelihood: RiskLevel
    impact: RiskLevel
    mitigation_strategies: List[str] = []
    monitoring_requirements: List[str] = []
    responsible_party: str
    review_frequency: str = "Quarterly"

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check including service dependencies"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "unknown",
            "document_processor": "unknown",
            "ai_models": "unknown"
        },
        "statistics": {
            "total_documents": 0,
            "total_processes": 0,
            "total_compliance_items": 0,
            "total_risk_assessments": 0
        }
    }
    
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
        
        # Get statistics
        health_status["statistics"]["total_documents"] = db.query(models.Document).count()
        health_status["statistics"]["total_processes"] = db.query(models.Process).count()
        health_status["statistics"]["total_compliance_items"] = db.query(models.ComplianceItem).count()
        health_status["statistics"]["total_risk_assessments"] = db.query(models.RiskAssessment).count()
        
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check document processor
        if document_processor:
            health_status["services"]["document_processor"] = "healthy"
        else:
            health_status["services"]["document_processor"] = "unhealthy"
    except Exception as e:
        health_status["services"]["document_processor"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check AI models
        if document_processor and document_processor.knowledge_extractor:
            health_status["services"]["ai_models"] = "healthy"
        else:
            health_status["services"]["ai_models"] = "degraded"
    except Exception as e:
        health_status["services"]["ai_models"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# Document upload and processing
@app.post("/upload/enhanced")
async def upload_document_enhanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Enhanced document upload with multi-format support"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Create document record
        document = models.Document(
            filename=file.filename,
            file_type=document_processor._get_file_type(file.filename),
            status="processing",
            processing_started_at=datetime.utcnow()
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document.id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document in background
        background_tasks.add_task(
            process_document_enhanced,
            document.id,
            str(file_path),
            db
        )
        
        return {
            "document_id": document.id,
            "filename": file.filename,
            "file_type": document.file_type,
            "status": "processing",
            "message": "Document uploaded successfully and processing started"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_enhanced(document_id: int, file_path: str, db: Session):
    """Enhanced background document processing"""
    try:
        # Get document record
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return
        
        # Process document
        metadata = {
            'document_id': document_id,
            'filename': document.filename,
            'file_type': document.file_type
        }
        
        results = document_processor.process_document(file_path, metadata)
        
        # Update document record
        document.extracted_text = results.get('extracted_text', '')
        document.document_sections = results.get('document_sections', {})
        document.status = "completed" if results['processing_metadata']['success'] else "failed"
        document.processing_completed_at = datetime.utcnow()
        document.processing_duration_seconds = (
            document.processing_completed_at - document.processing_started_at
        ).total_seconds()
        
        # Store confidence scores and metadata
        document.source_quality_score = results['confidence_scores'].get('overall', 0.0)
        document.content_completeness_score = results['processing_metadata'].get('success', 0.0)
        
        # Save extracted knowledge
        extracted_knowledge = results.get('extracted_knowledge', {})
        
        # Save processes
        for process_data in extracted_knowledge.get('processes', []):
            process = models.Process(
                document_id=document_id,
                process_id=process_data.process_id,
                name=process_data.name,
                description=process_data.description,
                knowledge_domain=process_data.domain,
                hierarchy_level=process_data.hierarchy_level,
                estimated_duration=process_data.estimated_duration,
                frequency=process_data.frequency,
                prerequisites=process_data.prerequisites,
                success_criteria=process_data.success_criteria,
                required_skills=process_data.required_skills,
                required_certifications=process_data.required_certifications,
                quality_standards=process_data.quality_standards,
                compliance_requirements=process_data.compliance_requirements,
                criticality_level=process_data.criticality_level,
                confidence_score=process_data.confidence_score
            )
            db.add(process)
        
        # Save compliance items
        for compliance_data in extracted_knowledge.get('compliance_items', []):
            compliance_item = models.ComplianceItem(
                document_id=document_id,
                regulation_name=compliance_data.regulation_name,
                regulation_section=compliance_data.section,
                requirement_description=compliance_data.description,
                compliance_status=compliance_data.status,
                responsible_party=compliance_data.responsible_party,
                evidence_location="",
                notes="",
                confidence=compliance_data.confidence
            )
            db.add(compliance_item)
        
        # Save risk assessments
        for risk_data in extracted_knowledge.get('risk_assessments', []):
            risk_assessment = models.RiskAssessment(
                document_id=document_id,
                risk_category=risk_data.category,
                risk_description=risk_data.description,
                likelihood=risk_data.likelihood,
                impact=risk_data.impact,
                overall_risk_level=max(risk_data.likelihood, risk_data.impact),
                mitigation_strategies=risk_data.mitigation_strategies,
                monitoring_requirements=risk_data.monitoring_requirements,
                responsible_party=risk_data.responsible_party,
                review_frequency="Quarterly",
                last_assessment_date=datetime.utcnow(),
                confidence=risk_data.confidence
            )
            db.add(risk_assessment)
        
        db.commit()
        logger.info(f"Document {document_id} processed successfully")
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Update document status to failed
        if document:
            document.status = "failed"
            document.last_error = str(e)
            document.processing_completed_at = datetime.utcnow()
            db.commit()

# Process management endpoints
@app.get("/processes")
async def get_processes(
    filter_params: ProcessFilter = Depends(),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get processes with filtering and pagination"""
    
    query = db.query(models.Process)
    
    # Apply filters
    if filter_params.domain:
        query = query.filter(models.Process.knowledge_domain == filter_params.domain)
    
    if filter_params.hierarchy_level:
        query = query.filter(models.Process.hierarchy_level == filter_params.hierarchy_level)
    
    if filter_params.criticality_level:
        query = query.filter(models.Process.criticality_level == filter_params.criticality_level)
    
    if filter_params.confidence_threshold:
        query = query.filter(models.Process.confidence_score >= filter_params.confidence_threshold)
    
    if filter_params.search_query:
        search_term = f"%{filter_params.search_query}%"
        query = query.filter(
            or_(
                models.Process.name.ilike(search_term),
                models.Process.description.ilike(search_term)
            )
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    processes = query.offset(skip).limit(limit).all()
    
    return {
        "processes": processes,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/processes/hierarchy")
async def get_process_hierarchy(
    domain: Optional[KnowledgeDomain] = None,
    db: Session = Depends(get_db)
):
    """Get hierarchical view of processes"""
    
    query = db.query(models.Process)
    if domain:
        query = query.filter(models.Process.knowledge_domain == domain)
    
    processes = query.all()
    
    # Build hierarchy
    hierarchy = {
        "core_functions": [],
        "departments": [],
        "procedures": [],
        "steps": []
    }
    
    level_map = {
        HierarchyLevel.CORE_BUSINESS_FUNCTION: "core_functions",
        HierarchyLevel.DEPARTMENT_OPERATION: "departments",
        HierarchyLevel.INDIVIDUAL_PROCEDURE: "procedures",
        HierarchyLevel.SPECIFIC_STEP: "steps"
    }
    
    for process in processes:
        level_key = level_map.get(process.hierarchy_level, "procedures")
        hierarchy[level_key].append({
            "id": process.id,
            "process_id": process.process_id,
            "name": process.name,
            "description": process.description,
            "domain": process.knowledge_domain.value if process.knowledge_domain else None,
            "criticality": process.criticality_level.value if process.criticality_level else None,
            "confidence": process.confidence_score,
            "parent_process_id": process.parent_process_id
        })
    
    return hierarchy

@app.post("/processes")
async def create_process(
    process_data: ProcessCreate,
    db: Session = Depends(get_db)
):
    """Create a new process"""
    
    # Generate unique process ID
    process_id = f"PROC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{process_data.name[:10].upper()}"
    
    process = models.Process(
        process_id=process_id,
        name=process_data.name,
        description=process_data.description,
        knowledge_domain=process_data.domain,
        hierarchy_level=process_data.hierarchy_level,
        parent_process_id=process_data.parent_process_id,
        estimated_duration=process_data.estimated_duration,
        frequency=process_data.frequency,
        prerequisites=process_data.prerequisites,
        success_criteria=process_data.success_criteria,
        required_skills=process_data.required_skills,
        required_certifications=process_data.required_certifications,
        quality_standards=process_data.quality_standards,
        compliance_requirements=process_data.compliance_requirements,
        criticality_level=process_data.criticality_level,
        confidence_score=1.0  # Manual entry has high confidence
    )
    
    db.add(process)
    db.commit()
    db.refresh(process)
    
    return process

@app.get("/processes/{process_id}")
async def get_process(process_id: str, db: Session = Depends(get_db)):
    """Get a specific process by ID"""
    
    process = db.query(models.Process).filter(models.Process.process_id == process_id).first()
    if not process:
        raise HTTPException(status_code=404, detail="Process not found")
    
    # Include related data
    result = {
        "process": process,
        "steps": process.steps,
        "decision_points": process.decision_points,
        "equipment_used": process.equipment_used,
        "personnel_involved": process.personnel_involved,
        "safety_requirements": process.safety_requirements
    }
    
    return result

@app.put("/processes/{process_id}")
async def update_process(
    process_id: str,
    process_update: ProcessUpdate,
    db: Session = Depends(get_db)
):
    """Update a process"""
    
    process = db.query(models.Process).filter(models.Process.process_id == process_id).first()
    if not process:
        raise HTTPException(status_code=404, detail="Process not found")
    
    # Update fields
    update_data = process_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(process, field, value)
    
    process.last_updated = datetime.utcnow()
    process.version += 1
    
    db.commit()
    db.refresh(process)
    
    return process

@app.delete("/processes/{process_id}")
async def delete_process(process_id: str, db: Session = Depends(get_db)):
    """Delete a process"""
    
    process = db.query(models.Process).filter(models.Process.process_id == process_id).first()
    if not process:
        raise HTTPException(status_code=404, detail="Process not found")
    
    db.delete(process)
    db.commit()
    
    return {"message": "Process deleted successfully"}

# Compliance management endpoints
@app.get("/compliance")
async def get_compliance_items(
    filter_params: ComplianceFilter = Depends(),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get compliance items with filtering"""
    
    query = db.query(models.ComplianceItem)
    
    # Apply filters
    if filter_params.regulation_name:
        query = query.filter(models.ComplianceItem.regulation_name.ilike(f"%{filter_params.regulation_name}%"))
    
    if filter_params.status:
        query = query.filter(models.ComplianceItem.compliance_status == filter_params.status)
    
    if filter_params.responsible_party:
        query = query.filter(models.ComplianceItem.responsible_party.ilike(f"%{filter_params.responsible_party}%"))
    
    if filter_params.review_due_days:
        cutoff_date = datetime.utcnow() + timedelta(days=filter_params.review_due_days)
        query = query.filter(models.ComplianceItem.next_review_date <= cutoff_date)
    
    total = query.count()
    compliance_items = query.offset(skip).limit(limit).all()
    
    return {
        "compliance_items": compliance_items,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.post("/compliance")
async def create_compliance_item(
    compliance_data: ComplianceItemCreate,
    db: Session = Depends(get_db)
):
    """Create a new compliance item"""
    
    compliance_item = models.ComplianceItem(
        regulation_name=compliance_data.regulation_name,
        regulation_section=compliance_data.regulation_section,
        requirement_description=compliance_data.requirement_description,
        compliance_status=ComplianceStatus.UNDER_REVIEW,
        responsible_party=compliance_data.responsible_party,
        evidence_location=compliance_data.evidence_location,
        notes=compliance_data.notes,
        last_review_date=datetime.utcnow(),
        next_review_date=datetime.utcnow() + timedelta(days=365 if compliance_data.review_frequency == "Annual" else 90),
        confidence=1.0
    )
    
    db.add(compliance_item)
    db.commit()
    db.refresh(compliance_item)
    
    return compliance_item

@app.get("/compliance/dashboard")
async def get_compliance_dashboard(db: Session = Depends(get_db)):
    """Get compliance dashboard data"""
    
    # Get compliance statistics
    total_items = db.query(models.ComplianceItem).count()
    compliant = db.query(models.ComplianceItem).filter(
        models.ComplianceItem.compliance_status == ComplianceStatus.COMPLIANT
    ).count()
    non_compliant = db.query(models.ComplianceItem).filter(
        models.ComplianceItem.compliance_status == ComplianceStatus.NON_COMPLIANT
    ).count()
    under_review = db.query(models.ComplianceItem).filter(
        models.ComplianceItem.compliance_status == ComplianceStatus.UNDER_REVIEW
    ).count()
    
    # Get items due for review
    cutoff_date = datetime.utcnow() + timedelta(days=30)
    due_for_review = db.query(models.ComplianceItem).filter(
        models.ComplianceItem.next_review_date <= cutoff_date
    ).all()
    
    # Get compliance by regulation
    regulation_stats = db.query(
        models.ComplianceItem.regulation_name,
        func.count(models.ComplianceItem.id).label('count'),
        func.avg(models.ComplianceItem.confidence).label('avg_confidence')
    ).group_by(models.ComplianceItem.regulation_name).all()
    
    return {
        "summary": {
            "total_items": total_items,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "under_review": under_review,
            "compliance_rate": (compliant / total_items * 100) if total_items > 0 else 0
        },
        "due_for_review": due_for_review,
        "by_regulation": [
            {
                "regulation": stat[0],
                "count": stat[1],
                "avg_confidence": float(stat[2]) if stat[2] else 0.0
            }
            for stat in regulation_stats
        ]
    }

# Risk management endpoints
@app.get("/risks")
async def get_risk_assessments(
    filter_params: RiskFilter = Depends(),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get risk assessments with filtering"""
    
    query = db.query(models.RiskAssessment)
    
    # Apply filters
    if filter_params.category:
        query = query.filter(models.RiskAssessment.risk_category == filter_params.category)
    
    if filter_params.likelihood:
        query = query.filter(models.RiskAssessment.likelihood == filter_params.likelihood)
    
    if filter_params.impact:
        query = query.filter(models.RiskAssessment.impact == filter_params.impact)
    
    if filter_params.overall_risk_level:
        query = query.filter(models.RiskAssessment.overall_risk_level == filter_params.overall_risk_level)
    
    total = query.count()
    risk_assessments = query.order_by(desc(models.RiskAssessment.overall_risk_level)).offset(skip).limit(limit).all()
    
    return {
        "risk_assessments": risk_assessments,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.post("/risks")
async def create_risk_assessment(
    risk_data: RiskAssessmentCreate,
    db: Session = Depends(get_db)
):
    """Create a new risk assessment"""
    
    # Calculate overall risk level
    risk_levels = {
        RiskLevel.VERY_LOW: 1,
        RiskLevel.LOW: 2,
        RiskLevel.MEDIUM: 3,
        RiskLevel.HIGH: 4,
        RiskLevel.VERY_HIGH: 5
    }
    
    likelihood_score = risk_levels.get(risk_data.likelihood, 3)
    impact_score = risk_levels.get(risk_data.impact, 3)
    overall_score = max(likelihood_score, impact_score)
    
    overall_risk_level = next(
        (level for level, score in risk_levels.items() if score == overall_score),
        RiskLevel.MEDIUM
    )
    
    risk_assessment = models.RiskAssessment(
        risk_category=risk_data.risk_category,
        risk_description=risk_data.risk_description,
        likelihood=risk_data.likelihood,
        impact=risk_data.impact,
        overall_risk_level=overall_risk_level,
        mitigation_strategies=risk_data.mitigation_strategies,
        monitoring_requirements=risk_data.monitoring_requirements,
        responsible_party=risk_data.responsible_party,
        review_frequency=risk_data.review_frequency,
        last_assessment_date=datetime.utcnow(),
        confidence=1.0
    )
    
    db.add(risk_assessment)
    db.commit()
    db.refresh(risk_assessment)
    
    return risk_assessment

@app.get("/risks/dashboard")
async def get_risk_dashboard(db: Session = Depends(get_db)):
    """Get risk management dashboard data"""
    
    # Get risk statistics
    total_risks = db.query(models.RiskAssessment).count()
    
    risk_distribution = db.query(
        models.RiskAssessment.overall_risk_level,
        func.count(models.RiskAssessment.id).label('count')
    ).group_by(models.RiskAssessment.overall_risk_level).all()
    
    # Get risks by category
    category_stats = db.query(
        models.RiskAssessment.risk_category,
        func.count(models.RiskAssessment.id).label('count'),
        func.avg(models.RiskAssessment.confidence).label('avg_confidence')
    ).group_by(models.RiskAssessment.risk_category).all()
    
    # Get high-risk items
    high_risks = db.query(models.RiskAssessment).filter(
        models.RiskAssessment.overall_risk_level.in_([RiskLevel.HIGH, RiskLevel.VERY_HIGH])
    ).all()
    
    return {
        "summary": {
            "total_risks": total_risks,
            "high_risk_count": len([r for r in high_risks if r.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]])
        },
        "risk_distribution": [
            {
                "level": dist[0].value if dist[0] else "unknown",
                "count": dist[1]
            }
            for dist in risk_distribution
        ],
        "by_category": [
            {
                "category": stat[0],
                "count": stat[1],
                "avg_confidence": float(stat[2]) if stat[2] else 0.0
            }
            for stat in category_stats
        ],
        "high_risks": high_risks[:10]  # Top 10 high risks
    }

# Advanced search and knowledge query endpoints
@app.post("/knowledge/search")
async def search_knowledge(
    query: KnowledgeQuery,
    db: Session = Depends(get_db)
):
    """Advanced knowledge search across all data types"""
    
    results = {
        "query": query.query,
        "processes": [],
        "compliance_items": [],
        "risk_assessments": [],
        "equipment": [],
        "personnel": [],
        "documents": []
    }
    
    search_term = f"%{query.query}%"
    
    try:
        # Search processes
        process_query = db.query(models.Process).filter(
            or_(
                models.Process.name.ilike(search_term),
                models.Process.description.ilike(search_term)
            )
        )
        
        if query.domains:
            process_query = process_query.filter(models.Process.knowledge_domain.in_(query.domains))
        
        if query.hierarchy_levels:
            process_query = process_query.filter(models.Process.hierarchy_level.in_(query.hierarchy_levels))
        
        results["processes"] = process_query.limit(query.max_results // 6).all()
        
        # Search compliance items
        compliance_query = db.query(models.ComplianceItem).filter(
            or_(
                models.ComplianceItem.regulation_name.ilike(search_term),
                models.ComplianceItem.requirement_description.ilike(search_term)
            )
        )
        results["compliance_items"] = compliance_query.limit(query.max_results // 6).all()
        
        # Search risk assessments
        risk_query = db.query(models.RiskAssessment).filter(
            or_(
                models.RiskAssessment.risk_description.ilike(search_term),
                models.RiskAssessment.risk_category.ilike(search_term)
            )
        )
        results["risk_assessments"] = risk_query.limit(query.max_results // 6).all()
        
        # Search equipment
        equipment_query = db.query(models.Equipment).filter(
            or_(
                models.Equipment.name.ilike(search_term),
                models.Equipment.type.ilike(search_term)
            )
        )
        results["equipment"] = equipment_query.limit(query.max_results // 6).all()
        
        # Search personnel
        personnel_query = db.query(models.Personnel).filter(
            or_(
                models.Personnel.name.ilike(search_term),
                models.Personnel.role.ilike(search_term)
            )
        )
        results["personnel"] = personnel_query.limit(query.max_results // 6).all()
        
        # Search documents
        document_query = db.query(models.Document).filter(
            or_(
                models.Document.filename.ilike(search_term),
                models.Document.extracted_text.ilike(search_term)
            )
        )
        results["documents"] = document_query.limit(query.max_results // 6).all()
        
    except Exception as e:
        logger.error(f"Error in knowledge search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return results

@app.get("/knowledge/analytics")
async def get_knowledge_analytics(db: Session = Depends(get_db)):
    """Get knowledge analytics and insights"""
    
    try:
        # Document statistics
        doc_stats = {
            "total_documents": db.query(models.Document).count(),
            "processed_documents": db.query(models.Document).filter(models.Document.status == "completed").count(),
            "failed_documents": db.query(models.Document).filter(models.Document.status == "failed").count(),
            "processing_documents": db.query(models.Document).filter(models.Document.status == "processing").count()
        }
        
        # Knowledge domain distribution
        domain_stats = db.query(
            models.Process.knowledge_domain,
            func.count(models.Process.id).label('count')
        ).group_by(models.Process.knowledge_domain).all()
        
        # Hierarchy level distribution
        hierarchy_stats = db.query(
            models.Process.hierarchy_level,
            func.count(models.Process.id).label('count')
        ).group_by(models.Process.hierarchy_level).all()
        
        # Confidence score analytics
        avg_confidence = db.query(func.avg(models.Process.confidence_score)).scalar() or 0.0
        
        # Recent activity
        recent_docs = db.query(models.Document).filter(
            models.Document.processing_timestamp >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        return {
            "document_statistics": doc_stats,
            "knowledge_distribution": {
                "by_domain": [
                    {
                        "domain": stat[0].value if stat[0] else "unknown",
                        "count": stat[1]
                    }
                    for stat in domain_stats
                ],
                "by_hierarchy": [
                    {
                        "level": stat[0].value if stat[0] else "unknown",
                        "count": stat[1]
                    }
                    for stat in hierarchy_stats
                ]
            },
            "quality_metrics": {
                "average_confidence": float(avg_confidence),
                "recent_activity": recent_docs
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export and reporting endpoints
@app.get("/export/processes")
async def export_processes(
    format: str = Query("json", regex="^(json|csv)$"),
    domain: Optional[KnowledgeDomain] = None,
    db: Session = Depends(get_db)
):
    """Export processes in various formats"""
    
    query = db.query(models.Process)
    if domain:
        query = query.filter(models.Process.knowledge_domain == domain)
    
    processes = query.all()
    
    if format == "json":
        return {"processes": [process.__dict__ for process in processes]}
    elif format == "csv":
        # Create CSV response
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Process ID', 'Name', 'Description', 'Domain', 'Hierarchy Level',
            'Criticality', 'Confidence Score', 'Duration', 'Frequency'
        ])
        
        # Write data
        for process in processes:
            writer.writerow([
                process.process_id,
                process.name,
                process.description,
                process.knowledge_domain.value if process.knowledge_domain else '',
                process.hierarchy_level.value if process.hierarchy_level else '',
                process.criticality_level.value if process.criticality_level else '',
                process.confidence_score,
                process.estimated_duration or '',
                process.frequency or ''
            ])
        
        output.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=processes.csv"}
        )

# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the main frontend application"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "public" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {"message": "EXPLAINIUM Enhanced Knowledge Extraction System API", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
