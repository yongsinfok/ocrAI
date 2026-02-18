# tests/test_index_manager.py
import pytest
from pathlib import Path
from src.index_manager import IndexManager

def test_index_manager_initialization():
    manager = IndexManager()
    assert manager is not None
    assert manager.index is not None
    assert manager.index_dir.exists()

def test_index_and_search():
    manager = IndexManager()

    # Index a document with sample pages
    doc_id = "test_doc_1"
    pages = [
        (1, "The quick brown fox jumps over the lazy dog."),
        (2, "Python is a great programming language for data science."),
        (3, "OCR technology can extract text from images and PDFs."),
    ]
    manager.index_document(doc_id, pages)

    # Search for content
    results = manager.search("programming")
    assert len(results) > 0
    assert results[0].page_num == 2
    assert "programming" in results[0].content.lower()
    assert results[0].score > 0

    # Search with doc_id filter
    results_filtered = manager.search("programming", doc_id=doc_id)
    assert len(results_filtered) > 0
    assert results_filtered[0].page_num == 2

    # Search for non-existent content
    empty_results = manager.search("nonexistent_term_xyz")
    assert len(empty_results) == 0

    # Test clear_document
    manager.clear_document(doc_id)
    results_after_clear = manager.search("programming", doc_id=doc_id)
    assert len(results_after_clear) == 0
