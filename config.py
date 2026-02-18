"""Application configuration."""
from pathlib import Path
from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration."""
    repo_id: str
    filename: str
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    verbose: bool = False

@dataclass
class GLMOCRConfig:
    """GLM-OCR configuration."""
    # Enable/disable GLM-OCR
    enabled: bool = True

    # Mode: "local" (vLLM/SGLang/Ollama) or "maas" (Zhipu MaaS API)
    mode: str = "local"

    # Local vLLM/SGLang configuration
    api_host: str = "localhost"
    api_port: int = 8080

    # MaaS API configuration (cloud mode)
    api_key: Optional[str] = None

    # Request timeouts
    connect_timeout: int = 300
    request_timeout: int = 300

    # Image preprocessing settings
    max_pixels: int = 71372800
    min_pixels: int = 12544
    image_format: str = "JPEG"

    # Output format: "json", "markdown", or "both"
    output_format: str = "both"

    # Enable layout detection (slower but better for complex documents)
    enable_layout: bool = False

    def __post_init__(self):
        """Load API key from environment if not set."""
        if self.api_key is None:
            self.api_key = os.getenv("GLM_OCR_API_KEY") or os.getenv("ZHIPU_API_KEY")

@dataclass
class AppConfig:
    """Application configuration."""
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    index_dir: Path = field(init=False)

    # Model configurations
    # OCR: GLM-OCR (preferred) or Tesseract (fallback)
    # LLM: Llama-3.1 for query understanding and text processing
    ocr_engine: str = "glm-ocr"  # Use GLM-OCR by default, falls back to tesseract
    glm_ocr: GLMOCRConfig = field(init=False)
    llama_3_1: ModelConfig = field(init=False)

    # Processing
    max_file_size_mb: int = 100
    supported_formats: tuple = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.index_dir = self.cache_dir / "index"

        # GLM-OCR configuration
        self.glm_ocr = GLMOCRConfig(
            enabled=os.getenv("GLM_OCR_ENABLED", "true").lower() == "true",
            mode=os.getenv("GLM_OCR_MODE", "local"),  # "local" or "maas"
            api_host=os.getenv("GLM_OCR_HOST", "localhost"),
            api_port=int(os.getenv("GLM_OCR_PORT", "8080")),
            api_key=os.getenv("GLM_OCR_API_KEY") or os.getenv("ZHIPU_API_KEY"),
        )

        # Llama-3.1 configuration for query processing
        self.llama_3_1 = ModelConfig(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=8192,
        )

        # Legacy GLM-OCR config (kept for backward compatibility)
        self.glm_ocr_legacy = ModelConfig(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=4096,
        )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
