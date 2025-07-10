import pytest
from unittest.mock import patch, MagicMock
from src.processors.document_processor import (
    get_file_type,
    _extract_equipment_data,
    _extract_procedure_data,
    _extract_safety_information,
    _extract_technical_specifications,
    _extract_personnel_data,
    extract_sections
)

# Mock external dependencies for isolated testing
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    with patch('src.ai.ner_extractor.ner_extractor') as mock_ner_extractor,
         patch('src.ai.classifier.classifier') as mock_classifier,
         patch('src.ai.keyphrase_extractor.keyphrase_extractor') as mock_keyphrase_extractor:
        mock_ner_extractor.extract_entities.return_value = []
        mock_classifier.classify_document.return_value = {"category": "unclassified", "score": 0.0}
        mock_keyphrase_extractor.extract_keyphrases.return_value = []
        yield

def test_get_file_type():
    assert get_file_type("document.pdf") == "document"
    assert get_file_type("image.JPG") == "image"
    assert get_file_type("video.mp4") == "video"
    assert get_file_type("archive.zip") == "other"

def test_extract_sections_no_sections():
    text = "This is a simple document with no clear sections."
    sections = extract_sections(text)
    assert "Full Document" in sections
    assert sections["Full Document"] == text

def test_extract_sections_with_sections():
    text = """
    1. Introduction
    This is the introduction.
    2. Procedure
    Step 1. Do this.
    Step 2. Do that.
    3. Conclusion
    This is the conclusion.
    """
    sections = extract_sections(text)
    assert "1. Introduction" in sections
    assert "2. Procedure" in sections
    assert "3. Conclusion" in sections
    assert "Introduction/Preamble" not in sections # No leading text before first section
    assert "This is the introduction." in sections["1. Introduction"]
    assert "Step 1. Do this.\n    Step 2. Do that." in sections["2. Procedure"]

def test_extract_equipment_data():
    text = "The centrifugal pump has a power of 10HP and operates at 480V."
    equipment = _extract_equipment_data(text, [])
    assert len(equipment) == 1
    assert equipment[0]["name"] == "pump"
    assert equipment[0]["type"] == "Pump"
    assert equipment[0]["specifications"]["power"] == "10HP"
    assert equipment[0]["specifications"]["voltage"] == "480V"

def test_extract_procedure_data():
    text = """
    Operating Instructions:
    1. Open valve A.
    2. Start pump B.
    """
    procedures = _extract_procedure_data(text, [])
    assert len(procedures) == 1
    assert procedures[0]["title"] == "Operating Instructions:"
    assert len(procedures[0]["steps"]) == 2
    assert procedures[0]["steps"][0]["description"] == "Open valve A."

def test_extract_safety_information():
    text = "WARNING: High voltage. Wear safety glasses and gloves."
    safety_info = _extract_safety_information(text, [])
    assert len(safety_info) == 1
    assert "High voltage" in safety_info[0]["hazard"]
    assert "safety glasses, gloves" in safety_info[0]["ppe_required"]

def test_extract_technical_specifications():
    text = "Temperature: 25C. Pressure: 100 psi. Dimension: 10.5mm +/- 0.1mm."
    tech_specs = _extract_technical_specifications(text, [])
    assert len(tech_specs) == 3
    assert tech_specs[0]["parameter"] == "Temperature"
    assert tech_specs[0]["value"] == "25"
    assert tech_specs[0]["unit"] == "C"
    assert tech_specs[1]["parameter"] == "Pressure"
    assert tech_specs[2]["tolerance"] == "+/- 0.1mm"

def test_extract_personnel_data():
    text = "John Doe, a certified engineer, is responsible."
    entities = [{'word': 'John Doe', 'entity_group': 'PER', 'score': 0.99, 'start': 0, 'end': 8}]
    personnel = _extract_personnel_data(text, entities)
    assert len(personnel) == 1
    assert personnel[0]["name"] == "John Doe"
    assert personnel[0]["role"] == "Engineer"
    assert "certified engineer" in personnel[0]["certifications"]
