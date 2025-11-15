"""
Test document processor
"""
import pytest
import tempfile
from pathlib import Path
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from src.exceptions import ProcessingError


@pytest.mark.integration
class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return StreamlinedDocumentProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly"""
        assert processor is not None
        assert processor.config is not None
        assert processor.knowledge_engine is not None
    
    def test_supported_formats(self, processor):
        """Test supported formats are defined"""
        assert len(processor.text_formats) > 0
        assert len(processor.image_formats) > 0
        assert len(processor.spreadsheet_formats) > 0
    
    def test_document_type_detection(self, processor):
        """Test document type detection"""
        assert processor._detect_document_type('.pdf') == 'manual'
        assert processor._detect_document_type('.docx') == 'document'
        assert processor._detect_document_type('.csv') == 'data'
        assert processor._detect_document_type('.jpg') == 'image'
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, processor):
        """Test processing non-existent file raises error"""
        with pytest.raises(ProcessingError):
            await processor.process_document('/nonexistent/file.pdf')
    
    @pytest.mark.asyncio
    async def test_process_plain_text(self, processor):
        """Test processing plain text file"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document with some content.\n")
            f.write("It contains multiple lines of text.\n")
            f.write("Safety procedure: Wear protective equipment.\n")
            temp_path = f.name
        
        try:
            result = await processor.process_document(temp_path)
            assert result is not None
            assert result.document_id is not None
            assert len(result.content_extracted) > 0
            assert result.processing_time > 0
        finally:
            Path(temp_path).unlink()
    
    def test_stats_tracking(self, processor):
        """Test statistics tracking"""
        stats = processor.get_processing_stats()
        assert 'documents_processed' in stats
        assert 'total_processing_time' in stats
        assert 'format_counts' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
