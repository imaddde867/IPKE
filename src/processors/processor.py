"""
EXPLAINIUM - Consolidated Document Processor

A clean, professional implementation that consolidates all document processing
functionality with support for multiple formats and AI-powered knowledge extraction.
"""

import os
import io
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
from datetime import datetime

# Core processing libraries
import requests
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import pandas as pd
from pptx import Presentation
import PyPDF2
import fitz  # PyMuPDF
from docx import Document as DocxDocument

# AI and NLP libraries
import spacy
from transformers import pipeline
import torch

# Internal imports
from src.logging_config import get_logger, log_processing_step
from src.config import config_manager
from src.exceptions import ProcessingError, AIError
from src.ai.knowledge_extractor import KnowledgeExtractor

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Consolidated document processor supporting multiple formats with AI-powered knowledge extraction.
    
    Supported formats:
    - Text: PDF, DOC, DOCX, TXT, RTF
    - Images: JPG, PNG, GIF, BMP, TIFF (with OCR)
    - Spreadsheets: XLS, XLSX, CSV
    - Presentations: PPT, PPTX
    - Audio: MP3, WAV (with transcription)
    - Video: MP4, AVI (with audio extraction)
    """
    
    def __init__(self):
        self.tika_url = config_manager.get_tika_url()
        self.knowledge_extractor = KnowledgeExtractor()
        self.supported_formats = {
            'text': ['.pdf', '.doc', '.docx', '.txt', '.rtf'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            'spreadsheet': ['.xls', '.xlsx', '.csv'],
            'presentation': ['.ppt', '.pptx'],
            'audio': ['.mp3', '.wav', '.flac', '.aac'],
            'video': ['.mp4', '.avi', '.mov', '.mkv']
        }
        
        # Initialize OCR
        self._init_ocr()
        
        # Initialize audio processing
        self._init_audio_processing()
    
    def _init_ocr(self):
        """Initialize OCR capabilities"""
        try:
            # Test pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info("OCR initialized successfully")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
            self.ocr_available = False
    
    def _init_audio_processing(self):
        """Initialize audio processing capabilities"""
        try:
            # Check if whisper is available
            import whisper
            self.whisper_model = whisper.load_model("base")
            self.audio_processing_available = True
            logger.info("Audio processing initialized successfully")
        except Exception as e:
            logger.warning(f"Audio processing initialization failed: {e}")
            self.audio_processing_available = False
            self.whisper_model = None
    
    def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Main method to process any supported document format
        
        Args:
            file_path: Path to the document file
            document_id: Database ID of the document
            
        Returns:
            Dictionary containing extracted content and knowledge
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        file_type = self._get_file_type(file_extension)
        
        logger.info(f"Processing {file_type} document: {file_path.name}")
        
        try:
            # Extract content based on file type
            if file_type == 'text':
                content = self._process_text_document(file_path)
            elif file_type == 'image':
                content = self._process_image_document(file_path)
            elif file_type == 'spreadsheet':
                content = self._process_spreadsheet_document(file_path)
            elif file_type == 'presentation':
                content = self._process_presentation_document(file_path)
            elif file_type == 'audio':
                content = self._process_audio_document(file_path)
            elif file_type == 'video':
                content = self._process_video_document(file_path)
            else:
                raise ProcessingError(f"Unsupported file type: {file_extension}")
            
            # Extract knowledge from content
            knowledge = self.knowledge_extractor.extract_knowledge(
                content['text'], 
                document_metadata={
                    'filename': file_path.name,
                    'file_type': file_type,
                    'document_id': document_id
                }
            )
            
            result = {
                'document_id': document_id,
                'filename': file_path.name,
                'file_type': file_type,
                'processed_at': datetime.now().isoformat(),
                'content': content,
                'knowledge': knowledge,
                'processing_status': 'completed'
            }
            
            logger.info(f"Successfully processed document: {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path.name}: {e}")
            raise ProcessingError(f"Failed to process document: {str(e)}")
    
    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension"""
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        return 'unknown'
    
    def _process_text_document(self, file_path: Path) -> Dict[str, Any]:
        """Process text-based documents (PDF, DOC, DOCX, TXT)"""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf_content(file_path)
        elif extension in ['.doc', '.docx']:
            return self._extract_word_content(file_path)
        elif extension == '.txt':
            return self._extract_text_content(file_path)
        else:
            # Try Tika for other formats
            return self._extract_with_tika(file_path)
    
    def _extract_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from PDF files"""
        text_content = ""
        metadata = {}
        
        try:
            # Try PyMuPDF first (better for complex PDFs)
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            metadata = doc.metadata
            doc.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                    
                    if pdf_reader.metadata:
                        metadata = dict(pdf_reader.metadata)
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                raise ProcessingError(f"Failed to extract PDF content: {e2}")
        
        return {
            'text': text_content.strip(),
            'metadata': metadata,
            'page_count': len(text_content.split('\f')) if '\f' in text_content else 1
        }
    
    def _extract_word_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from Word documents"""
        try:
            doc = DocxDocument(file_path)
            text_content = ""
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            # Extract tables
            table_content = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                table_content.append(table_data)
            
            return {
                'text': text_content.strip(),
                'tables': table_content,
                'paragraph_count': len(doc.paragraphs),
                'table_count': len(doc.tables)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract Word content: {e}")
            raise ProcessingError(f"Failed to extract Word content: {e}")
    
    def _extract_text_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return {
                'text': content,
                'line_count': len(content.split('\n')),
                'character_count': len(content)
            }
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    return {
                        'text': content,
                        'encoding': encoding,
                        'line_count': len(content.split('\n')),
                        'character_count': len(content)
                    }
                except UnicodeDecodeError:
                    continue
            
            raise ProcessingError("Unable to decode text file with any supported encoding")
    
    def _process_image_document(self, file_path: Path) -> Dict[str, Any]:
        """Process image documents with OCR"""
        if not self.ocr_available:
            raise ProcessingError("OCR not available")
        
        try:
            # Load and preprocess image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ProcessingError("Unable to load image")
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising and sharpening
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Extract text using OCR
            text_content = pytesseract.image_to_string(denoised)
            
            # Get image metadata
            height, width = gray.shape
            
            return {
                'text': text_content.strip(),
                'image_width': width,
                'image_height': height,
                'ocr_confidence': self._calculate_ocr_confidence(text_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise ProcessingError(f"Failed to process image: {e}")
    
    def _process_spreadsheet_document(self, file_path: Path) -> Dict[str, Any]:
        """Process spreadsheet documents"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            
            if isinstance(df, dict):  # Multiple sheets
                text_content = ""
                sheet_info = {}
                for sheet_name, sheet_df in df.items():
                    sheet_text = self._dataframe_to_text(sheet_df)
                    text_content += f"\n\n=== Sheet: {sheet_name} ===\n{sheet_text}"
                    sheet_info[sheet_name] = {
                        'rows': len(sheet_df),
                        'columns': len(sheet_df.columns)
                    }
                
                return {
                    'text': text_content.strip(),
                    'sheets': sheet_info,
                    'total_sheets': len(df)
                }
            else:  # Single sheet
                text_content = self._dataframe_to_text(df)
                return {
                    'text': text_content,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                
        except Exception as e:
            logger.error(f"Failed to process spreadsheet: {e}")
            raise ProcessingError(f"Failed to process spreadsheet: {e}")
    
    def _process_presentation_document(self, file_path: Path) -> Dict[str, Any]:
        """Process presentation documents"""
        try:
            prs = Presentation(file_path)
            text_content = ""
            slide_count = 0
            
            for slide in prs.slides:
                slide_count += 1
                slide_text = f"\n\n=== Slide {slide_count} ===\n"
                
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                
                text_content += slide_text
            
            return {
                'text': text_content.strip(),
                'slide_count': slide_count
            }
            
        except Exception as e:
            logger.error(f"Failed to process presentation: {e}")
            raise ProcessingError(f"Failed to process presentation: {e}")
    
    def _process_audio_document(self, file_path: Path) -> Dict[str, Any]:
        """Process audio documents with transcription"""
        if not self.audio_processing_available:
            raise ProcessingError("Audio processing not available")
        
        try:
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(str(file_path))
            
            return {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            raise ProcessingError(f"Failed to process audio: {e}")
    
    def _process_video_document(self, file_path: Path) -> Dict[str, Any]:
        """Process video documents by extracting audio and transcribing"""
        if not self.audio_processing_available:
            raise ProcessingError("Audio processing not available for video")
        
        try:
            import ffmpeg
            
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Extract audio using ffmpeg
                (
                    ffmpeg
                    .input(str(file_path))
                    .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Transcribe extracted audio
                result = self.whisper_model.transcribe(temp_audio_path)
                
                return {
                    'text': result['text'],
                    'language': result.get('language', 'unknown'),
                    'segments': result.get('segments', []),
                    'source': 'video_audio_extraction'
                }
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise ProcessingError(f"Failed to process video: {e}")
    
    def _extract_with_tika(self, file_path: Path) -> Dict[str, Any]:
        """Extract content using Apache Tika server"""
        if not self.tika_url:
            raise ProcessingError("Tika server not configured")
        
        try:
            with open(file_path, 'rb') as file:
                response = requests.put(
                    f"{self.tika_url}/tika",
                    data=file,
                    headers={'Accept': 'text/plain'},
                    timeout=30
                )
                response.raise_for_status()
                
                return {
                    'text': response.text.strip(),
                    'extraction_method': 'tika'
                }
                
        except Exception as e:
            logger.error(f"Tika extraction failed: {e}")
            raise ProcessingError(f"Tika extraction failed: {e}")
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text"""
        text_parts = []
        
        # Add column headers
        text_parts.append("Columns: " + ", ".join(df.columns.astype(str)))
        
        # Add data rows (limit to prevent excessive text)
        max_rows = 100
        for idx, row in df.head(max_rows).iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(f"Row {idx + 1}: {row_text}")
        
        if len(df) > max_rows:
            text_parts.append(f"... and {len(df) - max_rows} more rows")
        
        return "\n".join(text_parts)
    
    def _calculate_ocr_confidence(self, text: str) -> float:
        """Calculate OCR confidence based on text characteristics"""
        if not text.strip():
            return 0.0
        
        # Simple heuristic based on text characteristics
        confidence_factors = []
        
        # Check for reasonable word length
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            confidence_factors.append(min(avg_word_length / 5.0, 1.0))
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        confidence_factors.append(alpha_ratio)
        
        # Check for sentence structure
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if len(text) > 100:
            sentence_factor = min(sentence_endings / (len(text) / 100), 1.0)
            confidence_factors.append(sentence_factor)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5