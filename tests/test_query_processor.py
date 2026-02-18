"""Tests for query processor."""
import pytest
from src.query_processor import QueryProcessor, QueryType

def test_query_processor_initialization():
    processor = QueryProcessor(None)
    assert processor is not None

def test_analyze_simple_extract():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("从 Table 1.2.1 提取数据")
    assert result.query_type == QueryType.SIMPLE_EXTRACT

def test_analyze_format_convert():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("把Table 1.2.1转换成JSON格式")
    assert result.query_type == QueryType.FORMAT_CONVERT
    assert result.output_format == "json"

def test_analyze_compute():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("计算Table 1.2.1的总和")
    assert result.query_type == QueryType.COMPUTE

def test_analyze_multi_step():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("从Table 1.2.1提取数据，然后计算总和")
    assert result.query_type == QueryType.MULTI_STEP

def test_detect_target_table():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("从 Table 3.4.5 提取数据")
    assert result.target == "Table 3.4.5"

def test_detect_csv_format():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("导出为csv格式")
    assert result.output_format == "csv"
