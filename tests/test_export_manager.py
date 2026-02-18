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

def test_to_csv_with_list_of_lists():
    """Test to_csv() with list of lists data."""
    manager = ExportManager()
    data = [["Name", "Age"], ["Alice", 30], ["Bob", 25]]
    result = manager.to_csv(data)
    assert "Name,Age" in result
    assert "Alice,30" in result
    assert "Bob,25" in result

def test_to_json_with_dict():
    """Test to_json() with dict data."""
    manager = ExportManager()
    data = {"Name": "Alice", "Age": 30}
    result = manager.to_json(data)
    assert "Name" in result
    assert "Alice" in result
    assert "Age" in result
    assert "30" in result

def test_format_by_template_with_template_and_data():
    """Test format_by_template() with template and data."""
    manager = ExportManager()
    data = {"Name": "Alice", "Age": 30}
    template = "Name: {Name}, Age: {Age}"
    result = manager.format_by_template(data, template)
    assert result == "Name: Alice, Age: 30"
