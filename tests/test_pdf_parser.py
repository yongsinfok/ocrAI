# tests/test_pdf_parser.py
import pytest
from pathlib import Path
from src.pdf_parser import PDFParser, PageType, PageInfo

def test_parser_initialization():
    parser = PDFParser()
    assert parser is not None
    assert parser.docs == {}

def test_structure_index():
    """Test structure indexing with sample text patterns."""
    parser = PDFParser()

    # Create mock page info with structured content
    page_0 = PageInfo(
        page_num=0,
        page_type=PageType.TEXT,
        text="Introduction\nThis is the intro.",
        image_count=0,
        text_density=0.01
    )

    page_1 = PageInfo(
        page_num=1,
        page_type=PageType.TEXT,
        text="1.1 Getting Started\nThis section covers basics.\n\n2.0 Advanced Topics\nMore complex stuff.",
        image_count=0,
        text_density=0.02
    )

    page_2 = PageInfo(
        page_num=2,
        page_type=PageType.TEXT,
        text="Table 3.1 Data Summary\nSome table data.",
        image_count=0,
        text_density=0.015
    )

    index = {}

    parser._index_structure(page_0, index)
    parser._index_structure(page_1, index)
    parser._index_structure(page_2, index)

    # Verify that numbered sections are detected
    assert "1.1 Getting Started" in index
    assert index["1.1 Getting Started"] == 1

    assert "2.0 Advanced Topics" in index
    assert index["2.0 Advanced Topics"] == 1

    # Verify that table patterns are detected
    assert "Table 3.1 Data Summary" in index
    assert index["Table 3.1 Data Summary"] == 2

    # Verify section patterns are defined
    assert len(parser.SECTION_PATTERNS) == 3

