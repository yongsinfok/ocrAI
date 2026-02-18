# tests/test_ocr_engine.py
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.ocr_engine import OCREngine, OCRResult


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache" / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    manager.get_model = Mock(return_value=Mock())
    return manager


@pytest.fixture
def ocr_engine(temp_cache_dir, mock_model_manager):
    """Create an OCR engine for testing."""
    return OCREngine(mock_model_manager, cache_dir=temp_cache_dir)


# Test 1: Cache functionality
def test_save_to_cache(ocr_engine, temp_cache_dir):
    """Test saving OCR results to cache."""
    # Create a sample OCR result
    result = OCRResult(
        text="Sample text from OCR",
        tables=[[["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]],
        confidence=0.95,
        raw_output={"key": "value"}
    )

    # Create sample image bytes
    image_bytes = b"sample_image_data"

    # Save to cache
    ocr_engine._save_to_cache(image_bytes, result)

    # Verify cache file was created
    cache_path = ocr_engine._get_cache_path(image_bytes)
    assert cache_path.exists()

    # Verify cache content
    with open(cache_path, 'r') as f:
        cached_data = json.load(f)

    assert cached_data['text'] == "Sample text from OCR"
    assert cached_data['tables'] == [[["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]]
    assert cached_data['confidence'] == 0.95
    assert cached_data['raw_output'] == {"key": "value"}


def test_load_from_cache_hit(ocr_engine, temp_cache_dir):
    """Test loading a cached OCR result."""
    # Create a sample OCR result
    result = OCRResult(
        text="Cached text",
        tables=[],
        confidence=0.85,
        raw_output={}
    )

    # Create sample image bytes
    image_bytes = b"another_image_data"

    # Save to cache first
    ocr_engine._save_to_cache(image_bytes, result)

    # Load from cache
    loaded_result = ocr_engine._load_from_cache(image_bytes)

    assert loaded_result is not None
    assert loaded_result.text == "Cached text"
    assert loaded_result.confidence == 0.85
    assert loaded_result.tables == []


def test_load_from_cache_miss(ocr_engine):
    """Test loading when cache miss occurs."""
    # Use image bytes that haven't been cached
    image_bytes = b"non_cached_image"

    loaded_result = ocr_engine._load_from_cache(image_bytes)

    assert loaded_result is None


def test_cache_path_generation(ocr_engine):
    """Test that cache paths are generated consistently."""
    image_bytes = b"test_image"

    path1 = ocr_engine._get_cache_path(image_bytes)
    path2 = ocr_engine._get_cache_path(image_bytes)

    assert path1 == path2
    assert str(path1).endswith(".json")


# Test 2: _extract_tables() method
def test_extract_tables_with_tabs(ocr_engine):
    """Test extracting tables from tab-separated text."""
    text = """Header 1\tHeader 2\tHeader 3
Row 1 Col 1\tRow 1 Col 2\tRow 1 Col 3
Row 2 Col 1\tRow 2 Col 2\tRow 2 Col 3

Some regular text here."""

    tables = ocr_engine._extract_tables(text)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][0] == ["Header 1", "Header 2", "Header 3"]
    assert tables[0][1] == ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"]
    assert tables[0][2] == ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]


def test_extract_tables_with_pipes(ocr_engine):
    """Test extracting tables from pipe-separated text."""
    text = """| Name | Age | City |
| John | 25 | NYC |
| Jane | 30 | LA |"""

    tables = ocr_engine._extract_tables(text)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][0] == ["Name", "Age", "City"]
    assert tables[0][1] == ["John", "25", "NYC"]


def test_extract_tables_multiple_tables(ocr_engine):
    """Test extracting multiple tables from text."""
    text = """Table 1:
A\tB
1\t2

Some text between tables

Table 2:
X\tY
3\t4"""

    tables = ocr_engine._extract_tables(text)

    assert len(tables) == 2
    assert tables[0] == [["A", "B"], ["1", "2"]]
    assert tables[1] == [["X", "Y"], ["3", "4"]]


def test_extract_tables_no_tables(ocr_engine):
    """Test text with no table structures."""
    text = """This is just regular text.
It has multiple lines.
But no tables here."""

    tables = ocr_engine._extract_tables(text)

    assert len(tables) == 0


def test_extract_tables_empty_cells_filtered(ocr_engine):
    """Test that empty cells are filtered out."""
    text = """A\t\tB\t
1\t\t2\t"""

    tables = ocr_engine._extract_tables(text)

    assert len(tables) == 1
    # Empty cells should be filtered
    assert tables[0][0] == ["A", "B"]
    assert tables[0][1] == ["1", "2"]


# Test 3: process_image() with mocked model
def test_process_image_with_mock_model(ocr_engine):
    """Test process_image with a mocked model."""
    # Setup mock model
    mock_model = Mock()
    mock_model.return_value = {
        'choices': [{'text': 'Sample OCR text result'}]
    }
    ocr_engine._model = mock_model

    image_bytes = b"test_image_bytes"

    result = ocr_engine.process_image(image_bytes, use_cache=False)

    assert result.text == "Sample OCR text result"
    assert result.confidence == 0.9
    assert result.raw_output == {'choices': [{'text': 'Sample OCR text result'}]}

    # Verify model was called with correct parameters
    mock_model.assert_called_once()
    call_args = mock_model.call_args
    assert call_args[1]['max_tokens'] == 2048
    assert call_args[1]['temperature'] == 0.1


def test_process_image_uses_cache(ocr_engine):
    """Test that process_image uses cache when available."""
    # Pre-populate cache
    cached_result = OCRResult(
        text="Cached result",
        tables=[],
        confidence=0.88,
        raw_output={}
    )
    image_bytes = b"cached_image"
    ocr_engine._save_to_cache(image_bytes, cached_result)

    # Process with cache enabled
    result = ocr_engine.process_image(image_bytes, use_cache=True)

    assert result.text == "Cached result"
    assert result.confidence == 0.88


def test_process_image_saves_to_cache(ocr_engine, temp_cache_dir):
    """Test that process_image saves results to cache."""
    mock_model = Mock()
    mock_model.return_value = {
        'choices': [{'text': 'New OCR result'}]
    }
    ocr_engine._model = mock_model

    image_bytes = b"new_image"
    result = ocr_engine.process_image(image_bytes, use_cache=True)

    # Check cache was saved
    cache_path = ocr_engine._get_cache_path(image_bytes)
    assert cache_path.exists()

    # Verify can load from cache
    loaded = ocr_engine._load_from_cache(image_bytes)
    assert loaded.text == "New OCR result"


def test_process_image_with_tables_in_response(ocr_engine):
    """Test process_image with table data in response."""
    mock_model = Mock()
    mock_model.return_value = {
        'choices': [{'text': 'Header1\tHeader2\nRow1\tRow2'}]
    }
    ocr_engine._model = mock_model

    image_bytes = b"table_image"
    result = ocr_engine.process_image(image_bytes, use_cache=False)

    assert len(result.tables) == 1
    assert result.tables[0] == [["Header1", "Header2"], ["Row1", "Row2"]]


# Test 4: process_page_from_pdf() with mocked model
@patch('fitz.open')
def test_process_page_from_pdf(mock_fitz_open, ocr_engine):
    """Test processing a PDF page."""
    # Setup mock PDF document
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_pix = MagicMock()

    mock_pix.tobytes.return_value = b"pdf_page_image_bytes"
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__len__.return_value = 10
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__enter__ = Mock(return_value=mock_doc)
    mock_doc.__exit__ = Mock(return_value=False)
    mock_doc.close = Mock()
    mock_fitz_open.return_value = mock_doc

    # Setup mock model
    mock_model = Mock()
    mock_model.return_value = {
        'choices': [{'text': 'PDF Page Text'}]
    }
    ocr_engine._model = mock_model

    pdf_path = Path("/fake/path/test.pdf")
    result = ocr_engine.process_page_from_pdf(pdf_path, page_num=0, use_cache=False)

    assert result.text == "PDF Page Text"
    mock_fitz_open.assert_called_once_with("/fake/path/test.pdf")


@patch('fitz.open')
def test_process_page_from_pdf_invalid_page(mock_fitz_open, ocr_engine):
    """Test processing a PDF with invalid page number."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 5
    mock_doc.__enter__ = Mock(return_value=mock_doc)
    mock_doc.__exit__ = Mock(return_value=False)
    mock_doc.close = Mock()
    mock_fitz_open.return_value = mock_doc

    pdf_path = Path("/fake/path/test.pdf")

    with pytest.raises(ValueError, match="Page 10 not found in PDF"):
        ocr_engine.process_page_from_pdf(pdf_path, page_num=10, use_cache=False)


# Original initialization test
def test_ocr_engine_initialization():
    """Test basic OCR engine initialization."""
    engine = OCREngine(None)
    assert engine is not None
    assert engine.cache_dir == Path("cache/ocr")
    assert engine._model is None
