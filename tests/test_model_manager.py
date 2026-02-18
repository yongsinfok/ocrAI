# tests/test_model_manager.py
import pytest
from pathlib import Path
from src.model_manager import ModelManager
from config import config

def test_model_manager_initialization():
    manager = ModelManager(config)
    assert manager is not None
    assert manager.models_dir == config.models_dir

def test_model_path_resolution():
    manager = ModelManager(config)
    path = manager.get_model_path("glm_ocr")
    assert path.is_absolute()

def test_model_exists_check():
    manager = ModelManager(config)
    # Models not downloaded yet
    assert not manager.model_exists("glm_ocr")
