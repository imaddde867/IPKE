import os
import requests
import json
import ffmpeg
import whisper
import re
from src.ai.ner_extractor import ner_extractor
from src.ai.classifier import classifier
from src.ai.keyphrase_extractor import keyphrase_extractor

TIKA_SERVER_URL = "http://tika:9998"

# Define the candidate labels for document classification
CANDIDATE_LABELS = [
    "Operational Procedures",
    "Safety Documentation",
    "Training Materials",
    "Technical Specifications",
    "Maintenance Guides",
    "Quality Standards"
]

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
    # Look for common equipment names and associated specs using regex and NER
    equipment_patterns = {
        "Pump": r"\b(pump|centrifugal pump|gear pump)\b",
        "Motor": r"\b(motor|electric motor|induction motor)\b",
        "Valve": r"\b(valve|ball valve|gate valve|check valve)\b",
        "Sensor": r"\b(sensor|temperature sensor|pressure sensor)\b"
    }
    spec_patterns = {
        "power": r"(\d+\.?\d*)\s*(HP|kW|W)\b",
        "voltage": r"(\d+)\s*(V|volt)\b",
        "flow_rate": r"(\d+\.?\d*)\s*(gpm|l/s|m3/hr)\b",
        "pressure": r"(\d+\.?\d*)\s*(psi|bar|kPa)\b"
    }

    for eq_type, pattern in equipment_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(0).strip()
            specs = {}
            # Search for specs in a window around the equipment name
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context_text = text[context_start:context_end]

            for spec_name, spec_re in spec_patterns.items():
                spec_match = re.search(spec_re, context_text, re.IGNORECASE)
                if spec_match:
                    specs[spec_name] = spec_match.group(0).strip()
            
            equipment_list.append({"name": name, "type": eq_type, "specifications": specs, "location": None, "confidence": 0.8}) # Assign a default confidence

    # Refine with NER entities if available and relevant
    for entity in entities:
        if entity['entity_group'] == 'ORG' or entity['entity_group'] == 'PRODUCT': # Assuming these might be equipment names/models
            if any(kw in entity['word'].lower() for kw in equipment_patterns.keys()):
                # Avoid duplicates, merge specs if already found
                found = False
                for eq in equipment_list:
                    if eq["name"].lower() == entity['word'].lower():
                        found = True
                        break
                if not found:
                    equipment_list.append({"name": entity['word'], "type": "Generic", "specifications": {}, "location": None, "confidence": entity['score']}) # Use NER score as confidence

    return equipment_list

def _extract_procedure_data(text: str, entities: list) -> list[dict]:
    procedures = []
    # Look for common procedure titles and numbered/bulleted steps
    procedure_titles = re.findall(r'\b(Procedure|Operating Instructions|Maintenance Steps|Installation Guide)\b[\s\S]{0,50}\n', text, re.IGNORECASE)
    
    for title_match in procedure_titles:
        title = title_match.strip()
        steps = []
        # Attempt to find steps following the title
        # This is a very basic step extraction, needs improvement for complex documents
        step_matches = re.findall(r'^\s*(\d+\.\s*|\*\s*|\-\s*)(.*?)(?=\n\s*(\d+\.\s*|\*\s*|\-\s*)|$)', text[text.find(title):], re.MULTILINE | re.DOTALL)
        
        for i, step_tuple in enumerate(step_matches):
            step_text = step_tuple[1].strip()
            if step_text:
                steps.append({"step_number": i + 1, "description": step_text})
        
        if steps:
            procedures.append({"title": title, "steps": steps, "category": None, "confidence": 0.7}) # Assign a default confidence

    # If no specific titles, try to extract any numbered/bulleted lists as a generic procedure
    if not procedures:
        steps = []
        step_matches = re.findall(r'^\s*(\d+\.\s*|\*\s*|\-\s*)(.*?)(?=\n\s*(\d+\.\s*|\*\s*|\-\s*)|$)', text, re.MULTILINE | re.DOTALL)
        for i, step_tuple in enumerate(step_matches):
            step_text = step_tuple[1].strip()
            if step_text:
                steps.append({"step_number": i + 1, "description": step_text})
        if steps:
            procedures.append({"title": "General Procedure", "steps": steps, "category": None, "confidence": 0.6}) # Assign a lower confidence

    return procedures

def _extract_safety_information(text: str, entities: list) -> list[dict]:
    safety_info_list = []
    # Look for hazard/precaution keywords and associated PPE
    hazard_patterns = [
        r"\b(danger|warning|caution|hazard|risk)\b[\s\S]{0,100}?(?:\b(avoid|do not|ensure|wear)\b[\s\S]{0,100}?(?:\b(gloves|helmet|goggles|respirator|ear protection|safety glasses)\b)?)",
        r"\b(PPE required|Personal Protective Equipment)\b[\s\S]{0,100}?(?:\b(gloves|helmet|goggles|respirator|ear protection|safety glasses)\b)"
    ]
    ppe_keywords = ["gloves", "helmet", "goggles", "respirator", "ear protection", "safety glasses"]

    for pattern in hazard_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            hazard_text = match.group(0).strip()
            ppe_found = [ppe for ppe in ppe_keywords if ppe in hazard_text.lower()]
            safety_info_list.append({"hazard": hazard_text, "precaution": "See context", "ppe_required": ', '.join(ppe_found), "severity": None, "confidence": 0.8}) # Assign a default confidence
    
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
    # Look for NER entities classified as 'PERSON' and try to find roles/certifications nearby
    role_keywords = ["engineer", "technician", "operator", "manager", "supervisor", "specialist"]
    certification_patterns = [
        r"\b(OSHA\s*\d+)\b",
        r"\b(ISO\s*\d+)\b",
        r"\b(certified|licensed)\b[\s\S]{0,50}?(?:\b(welder|electrician|mechanic)\b)"
    ]

    for entity in entities:
        if entity['entity_group'] == 'PER':
            name = entity['word']
            role = "Unknown"
            certifications = []
            confidence = entity['score'] # Start with NER confidence

            # Search for role and certifications in a window around the person's name
            context_start = max(0, entity['start'] - 100)
            context_end = min(len(text), entity['end'] + 100)
            context_text = text[context_start:context_end]

            for r_kw in role_keywords:
                if re.search(r'\b' + re.escape(r_kw) + r'\b', context_text, re.IGNORECASE):
                    role = r_kw.capitalize()
                    confidence = max(confidence, 0.85) # Boost confidence if role found
                    break
            
            for cert_pattern in certification_patterns:
                for cert_match in re.finditer(cert_pattern, context_text, re.IGNORECASE):
                    certifications.append(cert_match.group(0).strip())
                    confidence = max(confidence, 0.9) # Boost confidence if certification found
            
            personnel_list.append({"name": name, "role": role, "responsibilities": None, "certifications": list(set(certifications)), "confidence": confidence})
    return personnel_list

# --- Main Processing Functions ---

def process_document(file_path: str):
    print(f"Processing document/image with Tika: {file_path}")
    filename = os.path.basename(file_path)
    extracted_text = ""
    metadata = {}
    status = "failed"
    extracted_entities = []
    classification_result = {"category": "unclassified", "score": 0.0}
    key_phrases = []

    # Structured data placeholders
    equipment_data = []
    procedure_data = []
    safety_info_data = []
    technical_spec_data = []
    personnel_data = []
    document_sections = {}

    try:
        with open(file_path, 'rb') as file:
            headers = {
                'Accept': 'application/json',  # Request JSON output
                'X-Tika-OCRTesseractPath': '/usr/bin/tesseract',  # Assuming Tesseract is installed in Tika container
                'X-Tika-OCRTimeout': '300',  # 5 minutes timeout for OCR
                'X-Tika-PDFextractInlineImages': 'true',  # Extract images from PDF
                'X-Tika-PDFOcrStrategy': 'ocr_and_text_extraction',  # OCR and text extraction
            }
            # Use /rmeta endpoint for rich metadata and content
            response = requests.put(f"{TIKA_SERVER_URL}/rmeta", data=file, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            tika_output = response.json()
            
            if tika_output and isinstance(tika_output, list) and len(tika_output) > 0:
                # Tika /rmeta returns a list of JSON objects, one for each embedded document/part
                main_content = tika_output[0]
                extracted_text = main_content.get('X-TIKA:content', '')
                metadata = main_content.get('metadata', {})
                status = "processed"

                # Extract document sections
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
                
            else:
                extracted_text = "Tika returned empty or unexpected output."
                status = "failed"

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Tika server: {e}")
        extracted_text = f"Error processing {filename} with Tika: {e}"
        metadata = {"source": "tika_processor", "original_path": file_path, "filename": filename, "error": str(e)}
        status = "failed"
    except json.JSONDecodeError as e:
        print(f"Error decoding Tika JSON response: {e}")
        extracted_text = f"Error decoding Tika JSON response for {filename}: {e}"
        metadata = {"source": "tika_processor", "original_path": file_path, "filename": filename, "error": str(e)}
        status = "failed"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        extracted_text = f"An unexpected error occurred while processing {filename}: {e}"
        metadata = {"source": "tika_processor", "original_path": file_path, "filename": filename, "error": str(e)}
        status = "failed"

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