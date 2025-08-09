import os
import requests
import json
import ffmpeg
import whisper
import re
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.ai.ner_extractor import ner_extractor
from src.ai.classifier import classifier
from src.ai.keyphrase_extractor import keyphrase_extractor
from src.ai.knowledge_extraction_engine import knowledge_extraction_engine
from src.ai.relationship_mapper import (
    ProcessDependencyMapper, EquipmentMaintenanceCorrelator,
    SkillFunctionLinker, ComplianceProcedureConnector
)
from src.ai.knowledge_graph import KnowledgeGraphBuilder
from src.exceptions import ProcessingError, AIError, ServiceUnavailableError
from src.logging_config import get_logger, log_processing_step, log_error
from src.config import config_manager

logger = get_logger(__name__)

# Get Tika URL from configuration
TIKA_SERVER_URL = config_manager.get_tika_url()

# Define the candidate labels for document classification
CANDIDATE_LABELS = [
    "Operational Procedures",
    "Safety Documentation",
    "Training Materials",
    "Technical Specifications",
    "Maintenance Guides",
    "Quality Standards"
]

# Image processing components
class ImageFormat(Enum):
    """Supported image formats for OCR processing."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    TIF = "tif"
    WEBP = "webp"

@dataclass
class OCRResult:
    """Structured result from OCR processing."""
    text: str
    confidence: float
    text_blocks: List[Dict]
    processing_time: float
    image_quality_score: float

class ImagePreprocessor:
    """Optimized image preprocessing for OCR accuracy."""
    
    def __init__(self):
        self.min_confidence = 30
        self.optimal_dpi = 300
        
    def _validate_image(self, image: np.ndarray) -> None:
        """Validate image quality and dimensions."""
        if image is None:
            raise ProcessingError("Invalid image: image is None")
        
        if image.size == 0:
            raise ProcessingError("Invalid image: image is empty")
        
        if len(image.shape) < 2:
            raise ProcessingError("Invalid image: not a valid image format")
    
    def _resize_for_ocr(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Resize image to optimal DPI for OCR."""
        height, width = image.shape[:2]
        
        # Calculate optimal size (assuming 96 DPI baseline)
        scale_factor = target_dpi / 96.0
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        if new_width > 4000 or new_height > 4000:
            # Limit maximum size to prevent memory issues
            scale_factor = min(4000 / width, 4000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""
        if len(image.shape) == 3:
            # For color images, denoise each channel
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # For grayscale images
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Create binary image for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use adaptive thresholding for better results
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up small artifacts
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Complete image preprocessing pipeline."""
        self._validate_image(image)
        
        # Resize to optimal DPI
        resized = self._resize_for_ocr(image)
        
        # Enhance contrast
        enhanced = self._enhance_contrast(resized)
        
        # Denoise
        denoised = self._denoise_image(enhanced)
        
        # Binarize
        binary = self._binarize_image(denoised)
        
        return binary

class OptimizedOCRProcessor:
    """High-performance OCR processor with multiple strategies."""
    
    def __init__(self, tesseract_path: str = "/usr/bin/tesseract"):
        self.tesseract_path = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Optimized OCR configurations for different scenarios
        self.ocr_configs = {
            'default': '--oem 3 --psm 6',
            'single_line': '--oem 3 --psm 7',
            'single_word': '--oem 3 --psm 8',
            'sparse_text': '--oem 3 --psm 6',
            'uniform_block': '--oem 3 --psm 6'
        }
        
        self.preprocessor = ImagePreprocessor()
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score for OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Normalize scores (0-1 range)
        sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize to reasonable range
        contrast_score = min(contrast / 100, 1.0)
        
        # Combined quality score
        quality_score = (sharpness_score * 0.6 + contrast_score * 0.4)
        
        return max(0.0, min(1.0, quality_score))
    
    def _extract_with_multiple_configs(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using multiple OCR configurations for better accuracy."""
        best_result = {"text": "", "confidence": 0.0, "blocks": []}
        
        for config_name, config in self.ocr_configs.items():
            try:
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config=config
                )
                
                # Extract text and confidence
                text_parts = []
                confidence_scores = []
                text_blocks = []
                
                for i, conf in enumerate(ocr_data['conf']):
                    if conf > self.preprocessor.min_confidence:
                        text = ocr_data['text'][i].strip()
                        if text:
                            text_parts.append(text)
                            confidence_scores.append(conf)
                            text_blocks.append({
                                'text': text,
                                'confidence': conf,
                                'bbox': {
                                    'x': ocr_data['left'][i],
                                    'y': ocr_data['top'][i],
                                    'width': ocr_data['width'][i],
                                    'height': ocr_data['height'][i]
                                }
                            })
                
                if text_parts:
                    combined_text = ' '.join(text_parts)
                    avg_confidence = np.mean(confidence_scores)
                    
                    # Keep the best result
                    if avg_confidence > best_result["confidence"]:
                        best_result = {
                            "text": combined_text,
                            "confidence": avg_confidence,
                            "blocks": text_blocks
                        }
                        
            except Exception as e:
                logger.warning(f"OCR config {config_name} failed: {e}")
                continue
        
        return best_result["text"], best_result["confidence"], best_result["blocks"]
    
    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text from image with comprehensive processing."""
        import time
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ProcessingError(f"Could not read image: {image_path}")
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(image)
            
            # Preprocess image
            preprocessed = self.preprocessor.preprocess(image)
            
            # Extract text with multiple configurations
            text, confidence, text_blocks = self._extract_with_multiple_configs(preprocessed)
            
            # Clean up text
            cleaned_text = self._clean_text(text)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                text_blocks=text_blocks,
                processing_time=processing_time,
                image_quality_score=quality_score
            )
            
        except Exception as e:
            log_error(logger, e, f"OCR extraction failed for {image_path}")
            raise ProcessingError(
                f"OCR extraction failed for {image_path}",
                file_path=image_path,
                processing_stage="ocr_extraction"
            ) from e
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'[|]', 'I', text)  # Fix common OCR confusion
        text = re.sub(r'[0]', 'O', text)  # Fix common OCR confusion
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()

# Initialize OCR processor
ocr_processor = OptimizedOCRProcessor()

def get_file_type(file_path: str) -> str:
    # Basic file type detection based on extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
        return "document"
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return "image"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return "video"
    else:
        return "other"

# --- Helper for Section Extraction ---
def extract_sections(text: str) -> dict:
    sections = {}
    # Simple regex to find common section headers (e.g., 1. Introduction, 2. Procedure, etc.)
    # This can be greatly improved with more sophisticated parsing or ML models
    section_pattern = re.compile(r'^\s*(\d+\.?\d*\s*[A-Z][a-zA-Z0-9\s\-]+)\s*$\n', re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    if not matches:
        sections["Full Document"] = text
        return sections

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_index = match.end()
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(text)
        sections[section_title] = text[start_index:end_index].strip()
    
    # Add any leading text before the first section
    if matches[0].start() > 0:
        sections["Introduction/Preamble"] = text[:matches[0].start()].strip()

    return sections

# --- Structured Data Extraction Functions (Enhanced for Demonstration) ---

def _extract_equipment_data(text: str, entities: list) -> list[dict]:
    equipment_list = []
    processed_equipment = set()  # Track processed equipment to avoid duplicates
    
    # Enhanced equipment patterns with more specific matching
    equipment_patterns = {
        "Motor": r"\b(?:primary\s+motor|electric\s+motor|induction\s+motor|motor)\b",
        "Pump": r"\b(?:cooling\s+pump|centrifugal\s+pump|gear\s+pump|pump)\b",
        "Valve": r"\b(?:ball\s+valve|gate\s+valve|check\s+valve|valve)\b",
        "Sensor": r"\b(?:temperature\s+sensor|pressure\s+sensor|sensor)\b",
        "Compressor": r"\b(?:air\s+compressor|compressor)\b",
        "Tank": r"\b(?:storage\s+tank|pressure\s+tank|tank)\b"
    }
    
    # Enhanced specification patterns with better capture groups
    spec_patterns = {
        "power": r"(\d+(?:\.\d+)?)\s*(HP|kW|W|horsepower)\b",
        "voltage": r"(\d+)\s*(V|volt|volts)\b",
        "flow_rate": r"(\d+(?:\.\d+)?)\s*(GPM|gpm|l/s|m3/hr|gallons per minute)\b",
        "pressure": r"(\d+(?:\.\d+)?)\s*(PSI|psi|bar|kPa|pounds per square inch)\b",
        "temperature": r"(\d+(?:\.\d+)?)\s*°?\s*(F|C|fahrenheit|celsius)\b",
        "phase": r"(\d+)\s*-?\s*(phase)\b",
        "rpm": r"(\d+(?:\.\d+)?)\s*(RPM|rpm|revolutions per minute)\b",
        "amperage": r"(\d+(?:\.\d+)?)\s*(A|amp|amps|amperes)\b"
    }

    # First, look for structured equipment specifications (like "Primary Motor: 10 HP electric motor, 480V, 3-phase")
    structured_patterns = [
        r"([A-Za-z\s]+(?:motor|pump|valve|sensor|compressor|tank))\s*:\s*([^\n\r]+)",
        r"([A-Za-z\s]+(?:motor|pump|valve|sensor|compressor|tank))\s*-\s*([^\n\r]+)",
        r"([A-Za-z\s]+(?:motor|pump|valve|sensor|compressor|tank))\s*\|\s*([^\n\r]+)"
    ]
    
    for pattern in structured_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            equipment_name = match.group(1).strip()
            spec_text = match.group(2).strip()
            
            # Determine equipment type
            eq_type = "Generic"
            for type_name, type_pattern in equipment_patterns.items():
                if re.search(type_pattern, equipment_name, re.IGNORECASE):
                    eq_type = type_name
                    break
            
            # Extract specifications from the spec text
            specs = {}
            for spec_name, spec_re in spec_patterns.items():
                spec_match = re.search(spec_re, spec_text, re.IGNORECASE)
                if spec_match:
                    value = spec_match.group(1)
                    unit = spec_match.group(2)
                    specs[spec_name] = f"{value} {unit}"
            
            # Create unique key for deduplication
            eq_key = f"{equipment_name.lower()}_{eq_type.lower()}"
            if eq_key not in processed_equipment and specs:
                equipment_list.append({
                    "name": equipment_name,
                    "type": eq_type,
                    "specifications": specs,
                    "location": None,
                    "confidence": 0.9
                })
                processed_equipment.add(eq_key)

    # Then look for equipment mentions with nearby specifications (only if not already found in structured sections)
    for eq_type, pattern in equipment_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(0).strip()
            
            # Create unique key for deduplication
            eq_key = f"{name.lower()}_{eq_type.lower()}"
            if eq_key in processed_equipment:
                continue
            
            # Skip if this equipment was found in a specification section
            in_spec_section = False
            for existing_eq in equipment_list:
                if (existing_eq['type'] == eq_type and 
                    existing_eq['confidence'] >= 0.9 and  # High confidence from structured extraction
                    name.lower() in existing_eq['name'].lower()):
                    in_spec_section = True
                    break
            
            if in_spec_section:
                continue
            
            specs = {}
            # Search for specs in a focused window around the equipment name
            context_start = max(0, match.start() - 150)
            context_end = min(len(text), match.end() + 150)
            context_text = text[context_start:context_end]

            for spec_name, spec_re in spec_patterns.items():
                spec_match = re.search(spec_re, context_text, re.IGNORECASE)
                if spec_match:
                    value = spec_match.group(1)
                    unit = spec_match.group(2)
                    specs[spec_name] = f"{value} {unit}"
            
            # Only add if we found some specifications and it's not a duplicate
            if specs and len(specs) >= 2:  # Require at least 2 specs to avoid noise
                equipment_list.append({
                    "name": name,
                    "type": eq_type,
                    "specifications": specs,
                    "location": None,
                    "confidence": 0.7
                })
                processed_equipment.add(eq_key)

    # Look for equipment in specification sections
    spec_section_pattern = r"(?:EQUIPMENT\s+SPECIFICATIONS?|TECHNICAL\s+SPECIFICATIONS?|SPECIFICATIONS?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)"
    spec_section_match = re.search(spec_section_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    if spec_section_match:
        spec_section = spec_section_match.group(1)
        
        # Look for equipment lines in the specification section
        equipment_lines = re.findall(r"([A-Za-z\s]+(?:motor|pump|valve|sensor|compressor|tank))\s*:?\s*([^\n\r]+)", spec_section, re.IGNORECASE)
        
        for equipment_name, spec_line in equipment_lines:
            equipment_name = equipment_name.strip()
            
            # Determine equipment type
            eq_type = "Generic"
            for type_name, type_pattern in equipment_patterns.items():
                if re.search(type_pattern, equipment_name, re.IGNORECASE):
                    eq_type = type_name
                    break
            
            # Create unique key for deduplication
            eq_key = f"{equipment_name.lower()}_{eq_type.lower()}"
            if eq_key in processed_equipment:
                continue
            
            # Extract specifications
            specs = {}
            for spec_name, spec_re in spec_patterns.items():
                spec_match = re.search(spec_re, spec_line, re.IGNORECASE)
                if spec_match:
                    value = spec_match.group(1)
                    unit = spec_match.group(2)
                    specs[spec_name] = f"{value} {unit}"
            
            if specs:
                equipment_list.append({
                    "name": equipment_name,
                    "type": eq_type,
                    "specifications": specs,
                    "location": None,
                    "confidence": 0.95
                })
                processed_equipment.add(eq_key)

    return equipment_list

def _extract_procedure_data(text: str, entities: list) -> list[dict]:
    procedures = []
    processed_procedures = set()
    
    # Enhanced procedure patterns
    procedure_patterns = {
        "Safety Procedures": r"(?:SAFETY\s+PROCEDURES?|SAFETY\s+INSTRUCTIONS?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)",
        "Maintenance Schedule": r"(?:MAINTENANCE\s+SCHEDULE|MAINTENANCE\s+PROCEDURES?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)",
        "Operating Instructions": r"(?:OPERATING\s+INSTRUCTIONS?|OPERATION\s+PROCEDURES?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)",
        "Installation Guide": r"(?:INSTALLATION\s+GUIDE|INSTALLATION\s+PROCEDURES?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)",
        "Emergency Procedures": r"(?:EMERGENCY\s+PROCEDURES?|EMERGENCY\s+SHUTDOWN)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)"
    }
    
    # Extract procedures from structured sections
    for category, pattern in procedure_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            section_text = match.group(1).strip()
            if not section_text:
                continue
                
            steps = []
            
            # Look for different step formats
            step_patterns = [
                r"^\s*(\d+\.)\s*(.+?)(?=\n\s*\d+\.|$)",  # Numbered steps
                r"^\s*([•\-\*])\s*(.+?)(?=\n\s*[•\-\*]|$)",  # Bullet points
                r"^([A-Za-z]+:)\s*(.+?)(?=\n[A-Za-z]+:|$)"  # Daily:, Weekly:, etc.
            ]
            
            for step_pattern in step_patterns:
                step_matches = re.findall(step_pattern, section_text, re.MULTILINE | re.DOTALL)
                if step_matches:
                    for i, (marker, step_text) in enumerate(step_matches):
                        step_text = step_text.strip().replace('\n', ' ')
                        if step_text and len(step_text) > 5:  # Filter out very short steps
                            steps.append({
                                "step_number": i + 1,
                                "description": step_text
                            })
                    break  # Use the first pattern that matches
            
            # If no structured steps found, treat the whole section as one procedure
            if not steps and len(section_text) > 20:
                # Split by sentences or lines
                lines = [line.strip() for line in section_text.split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    if len(line) > 10:  # Filter out very short lines
                        steps.append({
                            "step_number": i + 1,
                            "description": line
                        })
            
            if steps:
                proc_key = category.lower()
                if proc_key not in processed_procedures:
                    procedures.append({
                        "title": category,
                        "steps": steps,
                        "category": category,
                        "confidence": 0.9
                    })
                    processed_procedures.add(proc_key)

    # Look for standalone procedure titles with following steps
    standalone_patterns = [
        r"\b(Procedure|Operating Instructions|Maintenance Steps|Installation Guide|Setup Process|Calibration Process)\b\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)"
    ]
    
    for pattern in standalone_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            title = match.group(1).strip()
            section_text = match.group(2).strip()
            
            proc_key = title.lower()
            if proc_key in processed_procedures or not section_text:
                continue
            
            steps = []
            # Extract numbered or bulleted steps
            step_matches = re.findall(r'^\s*(\d+\.|\*|\-)\s*(.+?)(?=\n\s*(\d+\.|\*|\-)|$)', section_text, re.MULTILINE | re.DOTALL)
            
            for i, step_tuple in enumerate(step_matches):
                step_text = step_tuple[1].strip().replace('\n', ' ')
                if step_text and len(step_text) > 5:
                    steps.append({
                        "step_number": i + 1,
                        "description": step_text
                    })
            
            if steps:
                procedures.append({
                    "title": title,
                    "steps": steps,
                    "category": "General",
                    "confidence": 0.8
                })
                processed_procedures.add(proc_key)

    return procedures

def _extract_safety_information(text: str, entities: list) -> list[dict]:
    safety_info_list = []
    processed_safety = set()
    
    # Look for safety sections first
    safety_section_pattern = r"(?:SAFETY\s+PROCEDURES?|SAFETY\s+INFORMATION|SAFETY\s+REQUIREMENTS?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)"
    safety_section_match = re.search(safety_section_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    if safety_section_match:
        safety_section = safety_section_match.group(1)
        
        # Extract safety requirements from the section
        safety_lines = [line.strip() for line in safety_section.split('\n') if line.strip()]
        
        for line in safety_lines:
            if len(line) < 10:  # Skip very short lines
                continue
                
            # Determine hazard type and PPE requirements
            hazard = "General Safety"
            precaution = line
            ppe_required = []
            severity = "Medium"
            
            # Check for specific hazards
            if any(word in line.lower() for word in ['danger', 'warning', 'hazard']):
                severity = "High"
                if 'danger' in line.lower():
                    hazard = "Danger"
                elif 'warning' in line.lower():
                    hazard = "Warning"
                else:
                    hazard = "Hazard"
            
            # Extract PPE requirements
            ppe_keywords = {
                'safety goggles': ['goggles', 'safety glasses', 'eye protection'],
                'protective gloves': ['gloves', 'hand protection'],
                'hard hat': ['helmet', 'hard hat', 'head protection'],
                'respirator': ['respirator', 'breathing protection'],
                'ear protection': ['ear protection', 'hearing protection'],
                'safety shoes': ['safety shoes', 'steel toe', 'foot protection']
            }
            
            for ppe_type, keywords in ppe_keywords.items():
                if any(keyword in line.lower() for keyword in keywords):
                    ppe_required.append(ppe_type)
            
            safety_key = line.lower()[:50]  # Use first 50 chars as key
            if safety_key not in processed_safety:
                safety_info_list.append({
                    "hazard": hazard,
                    "precaution": precaution,
                    "ppe_required": ', '.join(ppe_required) if ppe_required else 'See document',
                    "severity": severity,
                    "confidence": 0.9
                })
                processed_safety.add(safety_key)

    # Look for specific safety patterns throughout the document (only if not already covered)
    if not safety_info_list:  # Only run if no safety section was found
        safety_patterns = [
            r"Always\s+wear\s+([^.]+)",
            r"(?:Must|Should|Required to)\s+wear\s+([^.]+)",
            r"Emergency\s+shutdown\s+([^.]+)",
            r"In\s+case\s+of\s+([^.]+)",
            r"(?:Danger|Warning|Caution):\s*([^.]+)"
        ]
        
        for pattern in safety_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                full_text = match.group(0).strip()
                requirement = match.group(1).strip()
                
                # Determine hazard type
                hazard = "Safety Requirement"
                if 'emergency' in full_text.lower():
                    hazard = "Emergency Procedure"
                elif 'wear' in full_text.lower():
                    hazard = "PPE Requirement"
                elif any(word in full_text.lower() for word in ['danger', 'warning', 'caution']):
                    hazard = full_text.split(':')[0].strip()
                
                safety_key = full_text.lower()[:50]
                if safety_key not in processed_safety:
                    safety_info_list.append({
                        "hazard": hazard,
                        "precaution": requirement,
                        "ppe_required": requirement if 'wear' in full_text.lower() else 'See document',
                        "severity": "High" if 'danger' in full_text.lower() else "Medium",
                        "confidence": 0.8
                    })
                    processed_safety.add(safety_key)
    
    # Further extraction could use NER entities for specific chemicals (e.g., CHEM) or conditions
    return safety_info_list

def _extract_technical_specifications(text: str, entities: list) -> list[dict]:
    tech_specs = []
    # Look for common measurement patterns with units
    measurement_patterns = [
        r"(\d+\.?\d*)\s*(C|F|K)\b", # Temperature
        r"(\d+\.?\d*)\s*(psi|bar|kPa)\b", # Pressure
        r"(\d+\.?\d*)\s*(mm|cm|m|inch)\b", # Length/Dimension
        r"(\d+\.?\d*)\s*(kg|g|lb)\b", # Weight
        r"(\d+\.?\d*)\s*(Hz|kHz|MHz)\b", # Frequency
        r"(\d+\.?\d*)\s*(A|amp)\b", # Current
        r"(\d+\.?\d*)\s*(V|volt)\b" # Voltage
    ]

    for pattern in measurement_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2)
            parameter = "Unknown"
            if unit.lower() in ["c", "f", "k"]: parameter = "Temperature"
            elif unit.lower() in ["psi", "bar", "kpa"]: parameter = "Pressure"
            elif unit.lower() in ["mm", "cm", "m", "inch"]: parameter = "Dimension"
            elif unit.lower() in ["kg", "g", "lb"]: parameter = "Weight"
            elif unit.lower() in ["hz", "khz", "mhz"]: parameter = "Frequency"
            elif unit.lower() in ["a", "amp"]: parameter = "Current"
            elif unit.lower() in ["v", "volt"]: parameter = "Voltage"

            tech_specs.append({"parameter": parameter, "value": value, "unit": unit, "tolerance": None, "confidence": 0.9}) # Assign a default confidence
    
    # Look for tolerance patterns (e.g., +/- 0.5mm)
    tolerance_pattern = r"\b[+-]\s*(\d+\.?\d*)\s*([a-zA-Z%]+)\b"
    for spec in tech_specs:
        context_start = max(0, text.find(spec["value"]) - 50)
        context_end = min(len(text), text.find(spec["value"]) + 50)
        context_text = text[context_start:context_end]
        tol_match = re.search(tolerance_pattern, context_text, re.IGNORECASE)
        if tol_match:
            spec["tolerance"] = tol_match.group(0)
            spec["confidence"] = 0.95 # Higher confidence if tolerance found

    return tech_specs

def _extract_personnel_data(text: str, entities: list) -> list[dict]:
    personnel_list = []
    processed_personnel = set()
    
    # Look for personnel sections first
    personnel_section_pattern = r"(?:PERSONNEL|STAFF|TEAM|CONTACTS?)\s*:?\s*\n((?:[^\n]*\n)*?)(?:\n\s*[A-Z][A-Z\s]+:|$)"
    personnel_section_match = re.search(personnel_section_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    if personnel_section_match:
        personnel_section = personnel_section_match.group(1)
        
        # Look for name - role - certification patterns
        personnel_patterns = [
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*-\s*([^,\n]+?)(?:,\s*([^,\n]+?))?(?:\n|$)",  # Name - Role, Certification
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*:\s*([^,\n]+?)(?:,\s*([^,\n]+?))?(?:\n|$)",  # Name: Role, Certification
        ]
        
        for pattern in personnel_patterns:
            matches = re.finditer(pattern, personnel_section, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                role = match.group(2).strip() if match.group(2) else "Unknown"
                certification_text = match.group(3).strip() if match.group(3) else ""
                
                # Parse certifications
                certifications = []
                if certification_text:
                    # Look for common certification patterns
                    cert_patterns = [
                        r"PE\s+certified",
                        r"OSHA\s*\d+\s+certified",
                        r"[A-Z]{2,}\s+certified",
                        r"certified\s+[A-Za-z\s]+",
                        r"licensed\s+[A-Za-z\s]+"
                    ]
                    
                    for cert_pattern in cert_patterns:
                        cert_matches = re.findall(cert_pattern, certification_text, re.IGNORECASE)
                        certifications.extend(cert_matches)
                
                name_key = name.lower()
                if name_key not in processed_personnel:
                    personnel_list.append({
                        "name": name,
                        "role": role,
                        "responsibilities": None,
                        "certifications": list(set(certifications)) if certifications else [],
                        "confidence": 0.95
                    })
                    processed_personnel.add(name_key)

    # Enhanced role keywords and patterns
    role_keywords = {
        "Chief Engineer": ["chief engineer", "lead engineer", "senior engineer"],
        "Engineer": ["engineer", "engineering"],
        "Supervisor": ["supervisor", "lead", "manager"],
        "Technician": ["technician", "tech"],
        "Operator": ["operator", "specialist"],
        "Maintenance": ["maintenance", "mechanic"],
        "Safety Officer": ["safety officer", "safety coordinator"]
    }
    
    certification_patterns = [
        r"\b(PE\s+certified)\b",
        r"\b(OSHA\s*\d+\s+certified)\b",
        r"\b(ISO\s*\d+\s+certified)\b",
        r"\b([A-Z]{2,}\s+certified)\b",
        r"\b(certified\s+[A-Za-z\s]+)\b",
        r"\b(licensed\s+[A-Za-z\s]+)\b"
    ]

    # Look for NER entities and structured patterns
    for entity in entities:
        if entity['entity_group'] == 'PER':
            name = entity['word']
            name_key = name.lower()
            
            if name_key in processed_personnel:
                continue
            
            role = "Unknown"
            certifications = []
            confidence = entity['score']

            # Search for role and certifications in a larger window
            context_start = max(0, entity['start'] - 200)
            context_end = min(len(text), entity['end'] + 200)
            context_text = text[context_start:context_end]

            # Find role
            for role_name, keywords in role_keywords.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', context_text, re.IGNORECASE):
                        role = role_name
                        confidence = max(confidence, 0.85)
                        break
                if role != "Unknown":
                    break
            
            # Find certifications
            for cert_pattern in certification_patterns:
                cert_matches = re.findall(cert_pattern, context_text, re.IGNORECASE)
                for cert_match in cert_matches:
                    certifications.append(cert_match.strip())
                    confidence = max(confidence, 0.9)
            
            personnel_list.append({
                "name": name,
                "role": role,
                "responsibilities": None,
                "certifications": list(set(certifications)),
                "confidence": confidence
            })
            processed_personnel.add(name_key)
    
    return personnel_list

# --- Main Processing Functions ---

def process_document(file_path: str):
    """Simplified document processing with better error handling."""
    logger.info(f"Starting document processing: {file_path}")
    filename = os.path.basename(file_path)
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise ProcessingError(f"File not found: {file_path}", file_path=file_path, processing_stage="file_validation")

    try:
        # Extract text using simplified Tika approach
        extracted_text = _extract_text_with_tika(file_path)
        
        # Basic processing - simplified for efficiency
        document_sections = extract_sections(extracted_text)
        
        # AI processing with graceful fallbacks
        extracted_entities = _safe_extract_entities(extracted_text)
        classification_result = _safe_classify_document(extracted_text)
        key_phrases = _safe_extract_keyphrases(extracted_text)
        
        # Structured data extraction - simplified
        equipment_data = _extract_equipment_data(extracted_text, extracted_entities)
        procedure_data = _extract_procedure_data(extracted_text, extracted_entities)
        safety_info_data = _extract_safety_information(extracted_text, extracted_entities)
        technical_spec_data = _extract_technical_specifications(extracted_text, extracted_entities)
        personnel_data = _extract_personnel_data(extracted_text, extracted_entities)
        
        return {
            "filename": filename,
            "extracted_text": extracted_text,
            "metadata": {"source": "tika", "processing_method": "simplified"},
            "extracted_entities": extracted_entities,
            "classification": classification_result,
            "key_phrases": key_phrases,
            "equipment_data": equipment_data,
            "procedure_data": procedure_data,
            "safety_info_data": safety_info_data,
            "technical_spec_data": technical_spec_data,
            "personnel_data": personnel_data,
            "document_sections": document_sections,
            "status": "processed"
        }
        
    except Exception as e:
        logger.error(f"Document processing failed for {filename}: {str(e)}")
        raise ProcessingError(f"Failed to process {filename}: {str(e)}", file_path=file_path, processing_stage="document_processing") from e

def process_document_enhanced(file_path: str, enable_advanced_features: bool = True):
    """
    Enhanced document processing with integrated knowledge graph, tacit knowledge extraction,
    and relationship mapping capabilities.
    """
    logger.info(f"Starting enhanced document processing: {file_path}")
    filename = os.path.basename(file_path)
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise ProcessingError(f"File not found: {file_path}", file_path=file_path, processing_stage="file_validation")

    try:
        # Step 1: Basic text extraction and processing (same as before)
        log_processing_step(logger, "text_extraction", "started", extra_data={'filename': filename})
        extracted_text = _extract_text_with_tika(file_path)
        document_sections = extract_sections(extracted_text)
        log_processing_step(logger, "text_extraction", "completed", extra_data={'text_length': len(extracted_text)})
        
        # Step 2: Basic AI processing
        log_processing_step(logger, "basic_ai_processing", "started")
        extracted_entities = _safe_extract_entities(extracted_text)
        classification_result = _safe_classify_document(extracted_text)
        key_phrases = _safe_extract_keyphrases(extracted_text)
        log_processing_step(logger, "basic_ai_processing", "completed", 
                          extra_data={'entities_count': len(extracted_entities), 'keyphrases_count': len(key_phrases)})
        
        # Step 3: Structured data extraction
        log_processing_step(logger, "structured_extraction", "started")
        equipment_data = _extract_equipment_data(extracted_text, extracted_entities)
        procedure_data = _extract_procedure_data(extracted_text, extracted_entities)
        safety_info_data = _extract_safety_information(extracted_text, extracted_entities)
        technical_spec_data = _extract_technical_specifications(extracted_text, extracted_entities)
        personnel_data = _extract_personnel_data(extracted_text, extracted_entities)
        log_processing_step(logger, "structured_extraction", "completed")
        
        # Step 4: Advanced AI processing (if enabled)
        tacit_knowledge_data = {}
        relationship_data = {}
        knowledge_graph_data = {}
        
        if enable_advanced_features:
            try:
                # Tacit knowledge extraction
                log_processing_step(logger, "tacit_knowledge_extraction", "started")
                tacit_knowledge_data = knowledge_extraction_engine.extract_tacit_knowledge(
                    extracted_text, extracted_entities
                )
                log_processing_step(logger, "tacit_knowledge_extraction", "completed",
                                  extra_data={'workflow_deps': len(tacit_knowledge_data.get('workflow_dependencies', [])),
                                            'decision_patterns': len(tacit_knowledge_data.get('decision_patterns', [])),
                                            'optimization_patterns': len(tacit_knowledge_data.get('optimization_patterns', [])),
                                            'communication_flows': len(tacit_knowledge_data.get('communication_flows', []))})
                
                # Relationship mapping
                log_processing_step(logger, "relationship_mapping", "started")
                relationship_data = _extract_relationships_enhanced(
                    extracted_text, extracted_entities, equipment_data, procedure_data, personnel_data, safety_info_data
                )
                log_processing_step(logger, "relationship_mapping", "completed",
                                  extra_data={'total_relationships': sum(len(v) for v in relationship_data.values())})
                
                # Knowledge graph construction
                log_processing_step(logger, "knowledge_graph_construction", "started")
                knowledge_graph_data = _build_knowledge_graph(
                    equipment_data, procedure_data, personnel_data, 
                    relationship_data, tacit_knowledge_data
                )
                log_processing_step(logger, "knowledge_graph_construction", "completed",
                                  extra_data={'nodes_count': knowledge_graph_data.get('nodes_count', 0),
                                            'edges_count': knowledge_graph_data.get('edges_count', 0)})
                
            except Exception as e:
                logger.warning(f"Advanced processing failed for {filename}: {str(e)}")
                # Continue with basic processing if advanced features fail
        
        # Compile final result
        result = {
            "filename": filename,
            "extracted_text": extracted_text,
            "metadata": {
                "source": "tika", 
                "processing_method": "enhanced" if enable_advanced_features else "basic",
                "advanced_features_enabled": enable_advanced_features
            },
            "extracted_entities": extracted_entities,
            "classification": classification_result,
            "key_phrases": key_phrases,
            "equipment_data": equipment_data,
            "procedure_data": procedure_data,
            "safety_info_data": safety_info_data,
            "technical_spec_data": technical_spec_data,
            "personnel_data": personnel_data,
            "document_sections": document_sections,
            "status": "processed"
        }
        
        # Add advanced processing results if available
        if enable_advanced_features and tacit_knowledge_data:
            result.update({
                "tacit_knowledge": tacit_knowledge_data,
                "relationships": relationship_data,
                "knowledge_graph": knowledge_graph_data
            })
        
        logger.info(f"Enhanced document processing completed successfully: {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced document processing failed for {filename}: {str(e)}")
        raise ProcessingError(f"Failed to process {filename}: {str(e)}", file_path=file_path, processing_stage="enhanced_document_processing") from e

def _extract_relationships_enhanced(text: str, entities: list, equipment_data: list, 
                                  procedure_data: list, personnel_data: list, safety_info_data: list) -> dict:
    """Extract relationships using the advanced relationship mapping components."""
    relationships = {
        'process_dependencies': [],
        'equipment_maintenance_correlations': [],
        'skill_function_links': [],
        'compliance_procedure_connections': []
    }
    
    try:
        # Convert extracted data to database-like objects for compatibility
        # This is a simplified conversion - in production you'd want proper model instances
        mock_knowledge_items = []
        mock_procedures = []
        mock_equipment = []
        mock_personnel = []
        mock_safety_info = []
        
        # Initialize relationship mappers
        dependency_mapper = ProcessDependencyMapper()
        maintenance_correlator = EquipmentMaintenanceCorrelator()
        skill_linker = SkillFunctionLinker()
        compliance_connector = ComplianceProcedureConnector()
        
        # Extract process dependencies
        if procedure_data:
            relationships['process_dependencies'] = dependency_mapper.identify_process_dependencies(
                mock_knowledge_items, mock_procedures
            )
        
        # Extract equipment maintenance correlations
        if equipment_data and procedure_data:
            relationships['equipment_maintenance_correlations'] = maintenance_correlator.correlate_equipment_maintenance(
                mock_equipment, mock_procedures
            )
        
        # Extract skill-function links
        if personnel_data and procedure_data:
            relationships['skill_function_links'] = skill_linker.link_skills_functions(
                mock_personnel, mock_procedures
            )
        
        # Extract compliance connections
        if procedure_data and safety_info_data:
            relationships['compliance_procedure_connections'] = compliance_connector.connect_compliance_procedures(
                mock_procedures, mock_safety_info
            )
        
    except Exception as e:
        logger.warning(f"Relationship extraction failed: {str(e)}")
    
    return relationships

def _build_knowledge_graph(equipment_data: list, procedure_data: list, personnel_data: list,
                         relationship_data: dict, tacit_knowledge_data: dict) -> dict:
    """Build knowledge graph from extracted data and relationships."""
    graph_data = {
        'nodes_count': 0,
        'edges_count': 0,
        'graph_metadata': {},
        'visualization_data': None
    }
    
    try:
        # Initialize knowledge graph builder
        graph_builder = KnowledgeGraphBuilder()
        
        # Convert data to appropriate formats for graph building
        # This is a simplified version - in production you'd want proper model instances
        knowledge_items = []
        process_dependencies = relationship_data.get('process_dependencies', [])
        equipment_correlations = relationship_data.get('equipment_maintenance_correlations', [])
        skill_links = relationship_data.get('skill_function_links', [])
        compliance_connections = relationship_data.get('compliance_procedure_connections', [])
        
        # Build the graph
        graph = graph_builder.build_knowledge_graph(
            knowledge_items=knowledge_items,
            process_dependencies=process_dependencies,
            equipment_correlations=equipment_correlations,
            skill_links=skill_links,
            compliance_connections=compliance_connections,
            equipment_list=None,  # Would convert equipment_data to Equipment objects
            personnel_list=None   # Would convert personnel_data to Personnel objects
        )
        
        graph_data.update({
            'nodes_count': graph.number_of_nodes(),
            'edges_count': graph.number_of_edges(),
            'graph_metadata': {
                'is_connected': len(list(graph.nodes())) > 0,
                'has_cycles': not graph.number_of_edges() == 0  # Simplified check
            }
        })
        
    except Exception as e:
        logger.warning(f"Knowledge graph construction failed: {str(e)}")
    
    return graph_data

def _extract_text_with_tika(file_path: str) -> str:
    """Simplified Tika text extraction with better error handling."""
    try:
        with open(file_path, 'rb') as file:
            # Use simple text extraction endpoint
            response = requests.put(
                f"{TIKA_SERVER_URL}/tika", 
                data=file, 
                headers={'Accept': 'text/plain'},
                timeout=30  # Shorter timeout
            )
            response.raise_for_status()
            return response.text.strip()
            
    except requests.exceptions.ConnectionError:
        logger.warning("Tika server unavailable, using fallback text extraction")
        return _fallback_text_extraction(file_path)
    except requests.exceptions.Timeout:
        logger.warning("Tika server timeout, using fallback text extraction")
        return _fallback_text_extraction(file_path)
    except Exception as e:
        logger.warning(f"Tika extraction failed: {str(e)}, using fallback")
        return _fallback_text_extraction(file_path)

def _fallback_text_extraction(file_path: str) -> str:
    """Fallback text extraction for when Tika is unavailable."""
    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    elif ext == '.pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except:
            return f"Could not extract text from PDF: {filename}"
    else:
        return f"Text extraction not available for file type: {ext}. File: {filename}"

def _safe_extract_entities(text: str) -> list:
    """Safe entity extraction with fallback."""
    try:
        # Simple regex-based entity extraction as fallback
        import re
        entities = []
        
        # Find potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(person_pattern, text):
            entities.append({
                "word": match.group(),
                "entity_group": "PER",
                "score": 0.8,
                "start": match.start(),
                "end": match.end()
            })
        
        # Find potential organizations/equipment
        org_pattern = r'\b[A-Z][A-Z0-9]+\b'
        for match in re.finditer(org_pattern, text):
            if len(match.group()) > 2:  # Avoid single letters
                entities.append({
                    "word": match.group(),
                    "entity_group": "ORG",
                    "score": 0.7,
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities[:20]  # Limit to first 20 entities
        
    except Exception as e:
        logger.warning(f"Entity extraction failed: {str(e)}")
        return []

def _safe_classify_document(text: str) -> dict:
    """Safe document classification with fallback."""
    try:
        # Simple keyword-based classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['safety', 'hazard', 'ppe', 'danger', 'warning']):
            return {"category": "Safety Documentation", "score": 0.8}
        elif any(word in text_lower for word in ['procedure', 'step', 'instruction', 'process']):
            return {"category": "Operational Procedures", "score": 0.8}
        elif any(word in text_lower for word in ['training', 'manual', 'guide', 'learn']):
            return {"category": "Training Materials", "score": 0.8}
        elif any(word in text_lower for word in ['specification', 'technical', 'parameter', 'measurement']):
            return {"category": "Technical Specifications", "score": 0.8}
        elif any(word in text_lower for word in ['maintenance', 'repair', 'service', 'inspect']):
            return {"category": "Maintenance Guides", "score": 0.8}
        else:
            return {"category": "Technical Specifications", "score": 0.6}
            
    except Exception as e:
        logger.warning(f"Document classification failed: {str(e)}")
        return {"category": "unclassified", "score": 0.0}

def _safe_extract_keyphrases(text: str) -> list:
    """Safe keyphrase extraction with fallback."""
    try:
        # Simple keyword extraction based on common industrial terms
        import re
        
        keywords = []
        
        # Technical terms
        tech_patterns = [
            r'\b\d+\s*(HP|kW|W|V|A|PSI|GPM|RPM|°F|°C)\b',
            r'\b(motor|pump|valve|sensor|pressure|temperature|flow)\b',
            r'\b(safety|maintenance|procedure|equipment|specification)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
        
        # Remove duplicates and limit
        return list(set(keywords))[:15]
        
    except Exception as e:
        logger.warning(f"Keyphrase extraction failed: {str(e)}")
        return []

def process_video(file_path: str):
    print(f"Processing video with FFmpeg and Whisper: {file_path}")
    filename = os.path.basename(file_path)
    extracted_text = ""
    metadata = {"source": "video_processor", "original_path": file_path}
    status = "failed"
    extracted_entities = []
    classification_result = {"category": "unclassified", "score": 0.0}
    key_phrases = []
    audio_output_path = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(filename)[0]}.mp3")

    # Structured data placeholders
    equipment_data = []
    procedure_data = []
    safety_info_data = []
    technical_spec_data = []
    personnel_data = []
    document_sections = {}

    try:
        # 1. Extract audio using ffmpeg
        ffmpeg.input(file_path).output(audio_output_path, acodec='libmp3lame').run(overwrite_output=True)
        print(f"Audio extracted to: {audio_output_path}")

        # 2. Transcribe audio using Whisper
        model = whisper.load_model("base") 
        result = model.transcribe(audio_output_path)
        extracted_text = result["text"]
        status = "processed"

        # Extract document sections (from transcript)
        document_sections = extract_sections(extracted_text)

        # Perform NER on the extracted text
        extracted_entities = ner_extractor.extract_entities(extracted_text)

        # Classify the document
        classification_result = classifier.classify_document(extracted_text, CANDIDATE_LABELS)

        # Extract key phrases
        key_phrases = keyphrase_extractor.extract_keyphrases(extracted_text)

        # Extract structured data
        equipment_data = _extract_equipment_data(extracted_text, extracted_entities)
        procedure_data = _extract_procedure_data(extracted_text, extracted_entities)
        safety_info_data = _extract_safety_information(extracted_text, extracted_entities)
        technical_spec_data = _extract_technical_specifications(extracted_text, extracted_entities)
        personnel_data = _extract_personnel_data(extracted_text, extracted_entities)

    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        extracted_text = f"FFmpeg error processing {filename}: {e.stderr.decode()}"
        metadata["error"] = str(e)
    except Exception as e:
        print(f"Whisper or other error: {e}")
        extracted_text = f"Error transcribing {filename}: {e}"
        metadata["error"] = str(e)
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)
            print(f"Cleaned up temporary audio file: {audio_output_path}")

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "metadata": metadata,
        "extracted_entities": extracted_entities,
        "classification": classification_result,
        "key_phrases": key_phrases,
        "equipment_data": equipment_data,
        "procedure_data": procedure_data,
        "safety_info_data": safety_info_data,
        "technical_spec_data": technical_spec_data,
        "personnel_data": personnel_data,
        "document_sections": document_sections,
        "status": status
    }

def process_image(file_path: str) -> Dict:
    """
    Process standalone images with optimized OCR and AI integration.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing extracted text and structured data
    """
    logger.info(f"Processing image with optimized OCR: {file_path}")
    filename = os.path.basename(file_path)
    
    # Initialize result structure
    result = {
        "filename": filename,
        "extracted_text": "",
        "metadata": {
            "source": "optimized_image_processor",
            "original_path": file_path,
            "processor_version": "2.0"
        },
        "extracted_entities": [],
        "classification": {"category": "unclassified", "score": 0.0},
        "key_phrases": [],
        "equipment_data": [],
        "procedure_data": [],
        "safety_info_data": [],
        "technical_spec_data": [],
        "personnel_data": [],
        "document_sections": {},
        "status": "failed",
        "ocr_metadata": {}
    }
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise ProcessingError(
            f"Image file not found: {file_path}",
            file_path=file_path,
            processing_stage="file_validation"
        )
    
    # Check if image format is supported
    supported_formats = {fmt.value for fmt in ImageFormat}
    ext = Path(file_path).suffix.lower().lstrip('.')
    if ext not in supported_formats:
        raise ProcessingError(
            f"Unsupported image format: {ext}",
            file_path=file_path,
            processing_stage="format_validation"
        )
    
    try:
        # Extract text using optimized OCR
        log_processing_step(logger, "ocr_extraction", "started", extra_data={'filename': filename})
        ocr_result = ocr_processor.extract_text(file_path)
        log_processing_step(logger, "ocr_extraction", "completed", 
                          extra_data={'text_length': len(ocr_result.text), 'confidence': ocr_result.confidence})
        
        result["extracted_text"] = ocr_result.text
        result["ocr_metadata"] = {
            'confidence': ocr_result.confidence,
            'processing_time': ocr_result.processing_time,
            'image_quality_score': ocr_result.image_quality_score,
            'text_blocks_count': len(ocr_result.text_blocks),
            'ocr_engine': 'tesseract_optimized'
        }
        
        if not ocr_result.text.strip():
            logger.warning(f"No text extracted from image: {filename}")
            result["status"] = "no_text_found"
            return result
        
        result["status"] = "processed"
        
        # Extract document sections
        result["document_sections"] = extract_sections(ocr_result.text)
        
        # Perform AI processing with error handling
        try:
            log_processing_step(logger, "ner_extraction", "started")
            result["extracted_entities"] = ner_extractor.extract_entities(ocr_result.text)
            log_processing_step(logger, "ner_extraction", "completed", 
                              extra_data={'entities_count': len(result["extracted_entities"])})
        except Exception as e:
            log_error(logger, e, "NER extraction failed - continuing with empty entities")
            result["extracted_entities"] = []
        
        try:
            log_processing_step(logger, "document_classification", "started")
            result["classification"] = classifier.classify_document(ocr_result.text, CANDIDATE_LABELS)
            log_processing_step(logger, "document_classification", "completed", 
                              extra_data={'category': result["classification"].get('category')})
        except Exception as e:
            log_error(logger, e, "Document classification failed - using default")
            result["classification"] = {"category": "unclassified", "score": 0.0}
        
        try:
            log_processing_step(logger, "keyphrase_extraction", "started")
            result["key_phrases"] = keyphrase_extractor.extract_keyphrases(ocr_result.text)
            log_processing_step(logger, "keyphrase_extraction", "completed", 
                              extra_data={'keyphrases_count': len(result["key_phrases"])})
        except Exception as e:
            log_error(logger, e, "Keyphrase extraction failed - continuing with empty list")
            result["key_phrases"] = []
        
        # Extract structured data
        result["equipment_data"] = _extract_equipment_data(ocr_result.text, result["extracted_entities"])
        result["procedure_data"] = _extract_procedure_data(ocr_result.text, result["extracted_entities"])
        result["safety_info_data"] = _extract_safety_information(ocr_result.text, result["extracted_entities"])
        result["technical_spec_data"] = _extract_technical_specifications(ocr_result.text, result["extracted_entities"])
        result["personnel_data"] = _extract_personnel_data(ocr_result.text, result["extracted_entities"])
        
        logger.info(f"Image processing completed successfully: {filename}")
        
    except Exception as e:
        log_error(logger, e, f"Image processing failed for {filename}")
        result["status"] = "failed"
        result["metadata"]["error"] = str(e)
        raise ProcessingError(
            f"Image processing failed for {filename}",
            file_path=file_path,
            processing_stage="image_processing"
        ) from e
    
    return result