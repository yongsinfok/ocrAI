# tests/test_ocr_engine.py
import pytest
from src.ocr_engine import OCREngine

def test_ocr_engine_initialization():
    engine = OCREngine(None)
    assert engine is not None
