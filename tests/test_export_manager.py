# tests/test_export_manager.py
import pytest
from src.export_manager import ExportManager

def test_export_manager_initialization():
    manager = ExportManager()
    assert manager is not None

def test_to_excel():
    manager = ExportManager()
    data = [["A", "B"], [1, 2]]
    result = manager.to_excel(data, "test")
    assert result is not None
