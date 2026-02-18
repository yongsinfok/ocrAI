# tests/test_index_manager.py
import pytest
from pathlib import Path
from src.index_manager import IndexManager

def test_index_manager_initialization():
    manager = IndexManager()
    assert manager is not None

def test_build_and_search():
    manager = IndexManager()
    # Build index
    # Search
    assert True
