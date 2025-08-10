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
import torch

# Internal imports
from src.logging_config import get_logger, log_processing_step
from src.core.config import config as config_manager
from src.exceptions import ProcessingError, AIError
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine

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
    
    def __init__(self, db_session=None):
        self.tika_url = config_manager.get_tika_url()
        self.advanced_engine = AdvancedKnowledgeEngine(config_manager.ai, db_session)
        self.db_session = db_session
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
        
        # Initialize advanced knowledge engine
        self._init_knowledge_engine()
        
        # Set knowledge engine availability
        self.knowledge_engine_available = True
    
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
            # Prefer configured model name when available
            model_name = getattr(getattr(config_manager, 'ai', object()), 'whisper_model', 'base') or 'base'
            self.whisper_model = whisper.load_model(model_name)
            self.audio_processing_available = True
            logger.info("Audio processing initialized successfully")
        except Exception as e:
            logger.warning(f"Audio processing initialization failed: {e}")
            self.audio_processing_available = False
            self.whisper_model = None
    
    def _init_knowledge_engine(self):
        """Initialize the advanced knowledge engine"""
        try:
            import asyncio
            # If we're already inside an event loop (e.g., FastAPI with uvicorn reload), schedule task
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.advanced_engine.initialize())
                logger.info("Scheduled async initialization of advanced knowledge engine")
            except RuntimeError:
                # No running loop; safe to run synchronously
                asyncio.run(self.advanced_engine.initialize())
            self.knowledge_engine_available = True  # will be true once initialize completes
        except Exception as e:
            logger.warning(f"Advanced knowledge engine initialization failed: {e}")
            self.knowledge_engine_available = False
    
    async def process_document_with_context(self, document: Dict[str, Any], company_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document with contextual understanding using company knowledge base.
        
        This method implements intelligent document processing that:
        1. Identifies document type and purpose
        2. Extracts structured and unstructured data
        3. Applies domain-specific extraction templates
        4. Cross-references with company knowledge base
        5. Generates semantic embeddings for similarity search
        """
        try:
            if not self.advanced_engine:
                raise AIError("Advanced knowledge engine not initialized")
            
            # Extract document metadata and content
            doc_type = document.get('type', 'unknown')
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            # Step 1: Identify document type and purpose
            document_purpose = await self._identify_document_purpose(content, doc_type, company_context)
            
            # Step 2: Extract structured and unstructured data
            structured_data = await self._extract_structured_data(content, doc_type, document_purpose)
            unstructured_data = await self._extract_unstructured_data(content, doc_type, document_purpose)
            
            # Step 3: Apply domain-specific extraction templates
            domain_data = await self._apply_domain_templates(content, doc_type, company_context)
            
            # Step 4: Cross-reference with company knowledge base
            cross_references = await self._cross_reference_knowledge(content, company_context)
            
            # Step 5: Generate semantic embeddings
            embeddings = await self._generate_semantic_embeddings(content, structured_data)
            
            # Combine all extracted information
            enhanced_document = {
                'original_document': document,
                'document_purpose': document_purpose,
                'structured_data': structured_data,
                'unstructured_data': unstructured_data,
                'domain_data': domain_data,
                'cross_references': cross_references,
                'embeddings': embeddings,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_method': 'intelligent_contextual'
            }
            
            # Extract deep knowledge using the advanced engine
            knowledge_extraction = await self.advanced_engine.extract_deep_knowledge(enhanced_document)
            enhanced_document['knowledge_extraction'] = knowledge_extraction
            
            logger.info(f"Successfully processed document with context: {document.get('id', 'unknown')}")
            return enhanced_document
            
        except Exception as e:
            logger.error(f"Failed to process document with context: {e}")
            raise ProcessingError(f"Contextual processing failed: {str(e)}")
    
    async def extract_tacit_knowledge(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract tacit knowledge and patterns across multiple documents.
        
        This method implements pattern recognition that identifies:
        - Recurring themes and patterns
        - Implicit workflows from email chains
        - Organizational structures from documents
        - Policy changes over time
        - Informal communication networks
        """
        try:
            if not self.advanced_engine:
                raise AIError("Advanced knowledge engine not initialized")
            
            # Step 1: Analyze document patterns
            document_patterns = await self._analyze_document_patterns(documents)
            
            # Step 2: Extract implicit workflows
            implicit_workflows = await self._extract_implicit_workflows(documents)
            
            # Step 3: Identify organizational structures
            org_structures = await self._identify_organizational_structures(documents)
            
            # Step 4: Detect policy changes
            policy_changes = await self._detect_policy_changes(documents)
            
            # Step 5: Map communication networks
            communication_networks = await self._map_communication_networks(documents)
            
            # Combine all tacit knowledge insights
            tacit_knowledge = {
                'document_patterns': document_patterns,
                'implicit_workflows': implicit_workflows,
                'organizational_structures': org_structures,
                'policy_changes': policy_changes,
                'communication_networks': communication_networks,
                'extraction_timestamp': datetime.now().isoformat(),
                'total_documents_analyzed': len(documents)
            }
            
            # Use advanced engine to build knowledge graph from tacit knowledge
            knowledge_graph = await self.advanced_engine.build_knowledge_graph(tacit_knowledge)
            tacit_knowledge['knowledge_graph'] = knowledge_graph
            
            logger.info(f"Successfully extracted tacit knowledge from {len(documents)} documents")
            return tacit_knowledge
                
        except Exception as e:
            logger.error(f"Failed to extract tacit knowledge: {e}")
            raise ProcessingError(f"Tacit knowledge extraction failed: {str(e)}")
    
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
            
            # Extract knowledge from content using intelligent AI framework
            knowledge = {}
            if self.knowledge_engine_available:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    document_data = {
                        'id': document_id,
                        'content': content.get('text', ''),
                        'filename': file_path.name,
                        'metadata': {
                            'filename': file_path.name,
                            'file_type': file_type,
                            'document_id': document_id,
                            'sections': content.get('sections', [])
                        }
                    }
                    
                    # Use the new intelligent knowledge extraction framework
                    knowledge = loop.run_until_complete(
                        self.advanced_engine.extract_intelligent_knowledge(document_data)
                    )
                    loop.close()
                    
                except Exception as e:
                    logger.error(f"Intelligent knowledge extraction failed, falling back to legacy: {e}")
                    # Fallback to legacy extraction if intelligent framework fails
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        document_data = {
                            'id': document_id,
                            'content': content.get('text', ''),
                            'metadata': {
                                'filename': file_path.name,
                                'file_type': file_type,
                                'document_id': document_id
                            }
                        }
                        
                        knowledge = loop.run_until_complete(
                            self.advanced_engine.extract_deep_knowledge(document_data)
                        )
                        knowledge['processing_note'] = 'Used legacy extraction due to intelligent framework failure'
                        loop.close()
                        
                    except Exception as legacy_e:
                        logger.error(f"Legacy knowledge extraction also failed: {legacy_e}")
                        knowledge = {'error': f'Both intelligent and legacy extraction failed: {str(e)}, {str(legacy_e)}'}
            else:
                knowledge = {'error': 'Knowledge engine not available'}
            
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
        
        # Extract sections for intelligent processing
        sections = self._extract_document_sections(text_content)
        
        return {
            'text': text_content.strip(),
            'metadata': metadata,
            'page_count': len(text_content.split('\f')) if '\f' in text_content else 1,
            'sections': sections
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
        """Process video documents by extracting audio and/or on-screen text.

        Strategy:
        1) If audio transcription is available, extract audio with ffmpeg and transcribe via Whisper.
        2) Independently (or as a fallback), sample frames and OCR on-screen text if OCR is available.
        Returns combined text so videos without clear audio still yield useful content.
        """
        transcript_text = ""
        language_detected = "unknown"
        segments = []
        sources: List[str] = []

        # Attempt audio transcription first when available
        if self.audio_processing_available and self.whisper_model is not None:
            try:
                import ffmpeg  # ffmpeg-python wrapper; requires ffmpeg CLI installed

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name

                try:
                    (
                        ffmpeg
                        .input(str(file_path))
                        .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                        .overwrite_output()
                        .run(quiet=True)
                    )

                    result = self.whisper_model.transcribe(temp_audio_path)
                    transcript_text = (result.get('text') or '').strip()
                    language_detected = result.get('language', 'unknown')
                    segments = result.get('segments', []) or []
                    sources.append('video_audio_extraction')
                finally:
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
            except Exception as e:
                # Do not fail video processing outright; fall back to frame OCR
                logger.warning(f"Audio transcription from video failed, continuing with frame OCR: {e}")
        else:
            logger.info("Audio processing unavailable; attempting frame OCR for video")

        # Frame OCR (supplemental or fallback)
        ocr_text = ""
        frames_analyzed = 0
        if self.ocr_available:
            try:
                ocr_result = self._extract_text_from_video_frames(
                    file_path=file_path,
                    interval_seconds=getattr(getattr(config_manager, 'processing', object()), 'video_frame_interval_seconds', 5),
                    max_frames=30
                )
                ocr_text = ocr_result.get('text', '').strip()
                frames_analyzed = ocr_result.get('frames_analyzed', 0)
                if ocr_text:
                    sources.append('video_frame_ocr')
            except Exception as e:
                logger.warning(f"Frame OCR failed for video: {e}")

        combined_text_parts = [part for part in [transcript_text, ocr_text] if part]
        combined_text = "\n\n".join(combined_text_parts)

        if not combined_text:
            # Nothing extracted at all; return a graceful, informative result rather than throwing
            return {
                'text': '',
                'language': language_detected,
                'segments': segments,
                'source': ",".join(sources) if sources else 'video_processing_attempted',
                'frames_analyzed': frames_analyzed,
                'warning': 'No extractable audio or on-screen text detected from video'
            }

        return {
            'text': combined_text,
            'language': language_detected,
            'segments': segments,
            'source': ",".join(sources) if sources else 'video_processing',
            'frames_analyzed': frames_analyzed
        }

    def _extract_text_from_video_frames(self, file_path: Path, interval_seconds: int = 5, max_frames: int = 20) -> Dict[str, Any]:
        """Extract visible text from sampled video frames using OCR.

        Samples one frame approximately every `interval_seconds`, up to `max_frames` frames.
        Applies simple preprocessing to improve OCR accuracy.
        """
        capture = cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            raise ProcessingError("Unable to open video file for frame OCR")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frames_between_samples = max(1, int(round((fps or 1.0) * max(1, interval_seconds))))

            collected_lines: List[str] = []
            frames_analyzed = 0
            frame_index = 0

            while frame_index < total_frames and frames_analyzed < max_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Light denoising and thresholding to enhance text regions
                denoised = cv2.fastNlMeansDenoising(gray)
                _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                text = pytesseract.image_to_string(thresh) if self.ocr_available else ''
                if text and text.strip():
                    # Keep unique non-empty lines to reduce noise
                    for line in [l.strip() for l in text.splitlines() if l.strip()]:
                        if line not in collected_lines:
                            collected_lines.append(line)

                frames_analyzed += 1
                frame_index += frames_between_samples

            ocr_text = "\n".join(collected_lines)
            return {'text': ocr_text, 'frames_analyzed': frames_analyzed}
        finally:
            capture.release()
    
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
        
        # Simple confidence calculation based on text characteristics
        confidence = 1.0
        
        # Penalize for common OCR errors
        if any(char in text for char in '|[]{}()'):
            confidence -= 0.1
        
        # Penalize for excessive numbers (common OCR artifact)
        if sum(c.isdigit() for c in text) / len(text) > 0.3:
            confidence -= 0.2
        
        # Penalize for very short words (OCR artifacts)
        words = text.split()
        if words and any(len(word) == 1 for word in words):
            confidence -= 0.1
        
        return max(0.0, confidence)
    
    # Helper methods for intelligent document processing
    
    async def _identify_document_purpose(self, content: str, doc_type: str, company_context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the purpose and intent of a document"""
        try:
            # Use LLM to analyze document purpose
            purpose_prompt = f"""
            Analyze this {doc_type} document and identify its purpose:
            
            Content: {content[:1000]}...
            
            Company Context: {company_context.get('industry', 'Unknown')} industry
            
            Identify:
            1. Primary purpose (e.g., policy, procedure, report, communication)
            2. Target audience
            3. Business function
            4. Urgency level
            5. Compliance requirements
            """
            
            purpose_analysis = await self.advanced_engine.llm.analyze(purpose_prompt)
            
            return {
                'primary_purpose': purpose_analysis.get('primary_purpose', 'unknown'),
                'target_audience': purpose_analysis.get('target_audience', 'unknown'),
                'business_function': purpose_analysis.get('business_function', 'unknown'),
                'urgency_level': purpose_analysis.get('urgency_level', 'low'),
                'compliance_requirements': purpose_analysis.get('compliance_requirements', [])
            }
            
        except Exception as e:
            logger.warning(f"Failed to identify document purpose: {e}")
            return {'primary_purpose': 'unknown', 'error': str(e)}
    
    async def _extract_structured_data(self, content: str, doc_type: str, purpose: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data based on document type and purpose"""
        try:
            structured_data = {}
            
            if doc_type in ['spreadsheet', 'csv']:
                # Extract tables, formulas, and data relationships
                structured_data['tables'] = await self._extract_table_structures(content)
                structured_data['formulas'] = await self._extract_formulas(content)
                structured_data['data_relationships'] = await self._extract_data_relationships(content)
            
            elif doc_type in ['presentation', 'ppt', 'pptx']:
                # Extract slide structure, key points, and flow
                structured_data['slide_structure'] = await self._extract_slide_structure(content)
                structured_data['key_points'] = await self._extract_key_points(content)
                structured_data['presentation_flow'] = await self._extract_presentation_flow(content)
            
            elif doc_type in ['text', 'pdf', 'doc', 'docx']:
                # Extract sections, headings, and document structure
                structured_data['sections'] = await self._extract_document_sections(content)
                structured_data['headings'] = await self._extract_headings(content)
                structured_data['document_structure'] = await self._extract_document_structure(content)
            
            return structured_data
            
        except Exception as e:
            logger.warning(f"Failed to extract structured data: {e}")
            return {'error': str(e)}
    
    async def _extract_unstructured_data(self, content: str, doc_type: str, purpose: Dict[str, Any]) -> Dict[str, Any]:
        """Extract unstructured data and insights"""
        try:
            unstructured_data = {}
            
            # Extract key concepts and entities
            concepts = await self.advanced_engine.extract_concepts(content)
            unstructured_data['concepts'] = concepts
            
            # Extract sentiment and tone
            sentiment = await self.advanced_engine.analyze_sentiment(content)
            unstructured_data['sentiment'] = sentiment
            
            # Extract key phrases and topics
            topics = await self.advanced_engine.extract_topics(content)
            unstructured_data['topics'] = topics
            
            # Extract named entities
            entities = await self.advanced_engine.extract_entities(content)
            unstructured_data['entities'] = entities
            
            return unstructured_data
            
        except Exception as e:
            logger.warning(f"Failed to extract unstructured data: {e}")
            return {'error': str(e)}
    
    async def _apply_domain_templates(self, content: str, doc_type: str, company_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply domain-specific extraction templates"""
        try:
            domain_data = {}
            industry = company_context.get('industry', 'general')
            
            if industry == 'healthcare':
                domain_data.update(await self._extract_healthcare_entities(content))
            elif industry == 'finance':
                domain_data.update(await self._extract_finance_entities(content))
            elif industry == 'legal':
                domain_data.update(await self._extract_legal_entities(content))
            elif industry == 'manufacturing':
                domain_data.update(await self._extract_manufacturing_entities(content))
            else:
                domain_data.update(await self._extract_general_entities(content))
            
            return domain_data
            
        except Exception as e:
            logger.warning(f"Failed to apply domain templates: {e}")
            return {'error': str(e)}
    
    async def _cross_reference_knowledge(self, content: str, company_context: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference content with existing company knowledge base"""
        try:
            cross_references = {}
            
            # Find similar documents
            similar_docs = await self.advanced_engine.find_similar_documents(content)
            cross_references['similar_documents'] = similar_docs
            
            # Find related processes
            related_processes = await self.advanced_engine.find_related_processes(content)
            cross_references['related_processes'] = related_processes
            
            # Find related systems
            related_systems = await self.advanced_engine.find_related_systems(content)
            cross_references['related_systems'] = related_systems
            
            # Find related requirements
            related_requirements = await self.advanced_engine.find_related_requirements(content)
            cross_references['related_requirements'] = related_requirements
            
            return cross_references
            
        except Exception as e:
            logger.warning(f"Failed to cross-reference knowledge: {e}")
            return {'error': str(e)}
    
    async def _generate_semantic_embeddings(self, content: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate semantic embeddings for similarity search"""
        try:
            embeddings = {}
            
            # Generate content embedding
            content_embedding = await self.advanced_engine.generate_embedding(content)
            embeddings['content'] = content_embedding
            
            # Generate embeddings for structured data
            if 'key_points' in structured_data:
                key_points_embedding = await self.advanced_engine.generate_embedding(
                    ' '.join(structured_data['key_points'])
                )
                embeddings['key_points'] = key_points_embedding
            
            if 'topics' in structured_data:
                topics_embedding = await self.advanced_engine.generate_embedding(
                    ' '.join(structured_data['topics'])
                )
                embeddings['topics'] = topics_embedding
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return {'error': str(e)}
    
    async def _analyze_document_patterns(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across multiple documents"""
        try:
            patterns = {}
            
            # Analyze content patterns
            content_patterns = await self.advanced_engine.analyze_content_patterns(documents)
            patterns['content'] = content_patterns
            
            # Analyze metadata patterns
            metadata_patterns = await self.advanced_engine.analyze_metadata_patterns(documents)
            patterns['metadata'] = metadata_patterns
            
            # Analyze temporal patterns
            temporal_patterns = await self.advanced_engine.analyze_temporal_patterns(documents)
            patterns['temporal'] = temporal_patterns
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Failed to analyze document patterns: {e}")
            return {'error': str(e)}
    
    async def _extract_implicit_workflows(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract implicit workflows from document patterns"""
        try:
            workflows = {}
            
            # Identify process flows
            process_flows = await self.advanced_engine.identify_process_flows(documents)
            workflows['process_flows'] = process_flows
            
            # Identify decision points
            decision_points = await self.advanced_engine.identify_decision_points(documents)
            workflows['decision_points'] = decision_points
            
            # Identify handoffs
            handoffs = await self.advanced_engine.identify_handoffs(documents)
            workflows['handoffs'] = handoffs
            
            return workflows
            
        except Exception as e:
            logger.warning(f"Failed to extract implicit workflows: {e}")
            return {'error': str(e)}
    
    async def _identify_organizational_structures(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify organizational structures from documents"""
        try:
            org_structures = {}
            
            # Identify roles and responsibilities
            roles = await self.advanced_engine.identify_roles_responsibilities(documents)
            org_structures['roles'] = roles
            
            # Identify reporting relationships
            reporting = await self.advanced_engine.identify_reporting_relationships(documents)
            org_structures['reporting'] = reporting
            
            # Identify teams and departments
            teams = await self.advanced_engine.identify_teams_departments(documents)
            org_structures['teams'] = teams
            
            return org_structures
            
        except Exception as e:
            logger.warning(f"Failed to identify organizational structures: {e}")
            return {'error': str(e)}
    
    async def _detect_policy_changes(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect policy changes over time"""
        try:
            policy_changes = {}
            
            # Identify policy evolution
            evolution = await self.advanced_engine.analyze_policy_evolution(documents)
            policy_changes['evolution'] = evolution
            
            # Identify version differences
            versions = await self.advanced_engine.analyze_version_differences(documents)
            policy_changes['versions'] = versions
            
            # Identify compliance changes
            compliance = await self.advanced_engine.analyze_compliance_changes(documents)
            policy_changes['compliance'] = compliance
            
            return policy_changes
            
        except Exception as e:
            logger.warning(f"Failed to detect policy changes: {e}")
            return {'error': str(e)}
    
    async def _map_communication_networks(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map informal communication networks"""
        try:
            networks = {}
            
            # Identify communication patterns
            patterns = await self.advanced_engine.identify_communication_patterns(documents)
            networks['patterns'] = patterns
            
            # Identify key communicators
            communicators = await self.advanced_engine.identify_key_communicators(documents)
            networks['communicators'] = communicators
            
            # Identify information flow
            flow = await self.advanced_engine.identify_information_flow(documents)
            networks['flow'] = flow
            
            return networks
            
        except Exception as e:
            logger.warning(f"Failed to map communication networks: {e}")
            return {'error': str(e)}
    
    # Placeholder methods for structured data extraction
    async def _extract_table_structures(self, content: str) -> List[Dict[str, Any]]:
        """Extract table structures from content"""
        # Placeholder implementation
        return []
    
    async def _extract_formulas(self, content: str) -> List[str]:
        """Extract formulas from content"""
        # Placeholder implementation
        return []
    
    async def _extract_data_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract data relationships from content"""
        # Placeholder implementation
        return []
    
    async def _extract_slide_structure(self, content: str) -> Dict[str, Any]:
        """Extract slide structure from content"""
        # Placeholder implementation
        return {}
    
    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        # Placeholder implementation
        return []
    
    async def _extract_presentation_flow(self, content: str) -> List[Dict[str, Any]]:
        """Extract presentation flow from content"""
        # Placeholder implementation
        return []
    
    async def _extract_document_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract document sections from content"""
        # Placeholder implementation
        return []
    
    async def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from content"""
        # Placeholder implementation
        return []
    
    async def _extract_document_structure(self, content: str) -> Dict[str, Any]:
        """Extract document structure from content"""
        # Placeholder implementation
        return {}
    
    # Placeholder methods for domain-specific extraction
    async def _extract_healthcare_entities(self, content: str) -> Dict[str, Any]:
        """Extract healthcare-specific entities"""
        # Placeholder implementation
        return {}
    
    async def _extract_finance_entities(self, content: str) -> Dict[str, Any]:
        """Extract finance-specific entities"""
        # Placeholder implementation
        return {}
    
    async def _extract_legal_entities(self, content: str) -> Dict[str, Any]:
        """Extract legal-specific entities"""
        # Placeholder implementation
        return {}
    
    async def _extract_manufacturing_entities(self, content: str) -> Dict[str, Any]:
        """Extract manufacturing-specific entities"""
        # Placeholder implementation
        return {}
    
    async def _extract_general_entities(self, content: str) -> Dict[str, Any]:
        """Extract general entities"""
        # Placeholder implementation
        return {}
        
            # Note: The OCR confidence helper and its heuristic are defined above
    
    def _extract_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections for intelligent processing"""
        sections = []
        
        # Split text into logical sections
        section_patterns = [
            r'\n#+\s*(.*?)\n',  # Markdown headers
            r'\n(\d+\.?\s+.*?)\n',  # Numbered sections
            r'\n([A-Z][A-Z\s]{3,})\n',  # ALL CAPS headers
            r'\n(SECTION.*?)\n',  # Section headers
            r'\n(CHAPTER.*?)\n',  # Chapter headers
        ]
        
        current_section = ""
        current_title = "Introduction"
        section_id = 0
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            
            # Check if this line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, f'\n{line}\n'):
                    is_header = True
                    # Save previous section
                    if current_section.strip():
                        sections.append({
                            'id': section_id,
                            'title': current_title,
                            'content': current_section.strip(),
                            'word_count': len(current_section.split()),
                            'type': self._classify_section_type(current_title, current_section)
                        })
                        section_id += 1
                    
                    # Start new section
                    current_title = line_stripped
                    current_section = ""
                    break
            
            if not is_header:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append({
                'id': section_id,
                'title': current_title,
                'content': current_section.strip(),
                'word_count': len(current_section.split()),
                'type': self._classify_section_type(current_title, current_section)
            })
        
        return sections
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """Classify section type for intelligent processing"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(keyword in title_lower for keyword in ["process", "procedure", "step"]):
            return "process"
        elif any(keyword in title_lower for keyword in ["requirement", "compliance", "standard"]):
            return "compliance"
        elif any(keyword in title_lower for keyword in ["risk", "safety", "hazard"]):
            return "risk"
        elif any(keyword in title_lower for keyword in ["role", "responsibility", "organization"]):
            return "organizational"
        elif any(keyword in title_lower for keyword in ["definition", "glossary", "term"]):
            return "definition"
        elif any(keyword in content_lower for keyword in ["must", "shall", "mandatory", "required"]):
            return "requirement"
        elif any(keyword in content_lower for keyword in ["step", "procedure", "process"]):
            return "process"
        else:
            return "general"