"""
Unit tests for DocumentProcessor service
"""
import pytest
from pathlib import Path
from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Fixture to create a DocumentProcessor instance"""
    return DocumentProcessor()


def test_valid_pdf_file_passes(processor):
    """Valid PDF file under size limit should pass"""
    processor.validate_file("test.pdf", 5 * 1024 * 1024)  # 5MB


def test_invalid_extension_raises_error(processor):
    """Unsupported file extension should raise ValueError"""
    with pytest.raises(ValueError, match="File type not allowed"):
        processor.validate_file("test.exe", 5 * 1024 * 1024)


def test_file_too_large_raises_error(processor):
    """File exceeding size limit should raise ValueError"""
    with pytest.raises(ValueError, match="File too large"):
        processor.validate_file("test.pdf", 20 * 1024 * 1024)  # 20MB


def test_generate_document_id_is_consistent(processor):
    """Document ID should be deterministic and space-free"""
    filename = "test document.pdf"
    content = b"test content"
    doc_id = processor.generate_document_id(filename, content)

    assert isinstance(doc_id, str)
    assert len(doc_id) > 0
    assert " " not in doc_id

    # Same input should yield same ID
    doc_id2 = processor.generate_document_id(filename, content)
    assert doc_id == doc_id2


def test_chunk_text_returns_chunks(processor):
    """Chunking long text should return structured chunks"""
    text = "This is a test. " * 200
    metadata = {"document_id": "test123", "filename": "test.txt"}

    chunks = processor.chunk_text(text, metadata)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["document_id"] == "test123"


def test_extract_page_number_with_marker(processor):
    """Should extract page number from marker"""
    text = "[Page 5]
Some content here"
    page_num = processor._extract_page_number(text)
    assert page_num == 5


def test_extract_page_number_without_marker(processor):
    """Should default to page 1 if no marker"""
    text = "Some content without page marker"
    page_num = processor._extract_page_number(text)
    assert page_num == 1


def test_chunk_text_empty_string_raises_error(processor):
    """Empty text should raise ValueError"""
    metadata = {"document_id": "test123"}
    with pytest.raises(ValueError, match="Text is empty"):
        processor.chunk_text("", metadata)
