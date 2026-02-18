"""Application configuration."""
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Model configuration."""
    repo_id: str
    filename: str
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    verbose: bool = False

@dataclass
class AppConfig:
    """Application configuration."""
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    index_dir: Path = field(init=False)

    # Model configurations
    # OCR: Tesseract (local OCR engine)
    # LLM: Llama-3.1 for query understanding and text processing
    ocr_engine: str = "tesseract"  # Use Tesseract OCR
    llama_3_1: ModelConfig = field(init=False)

    # Processing
    max_file_size_mb: int = 100
    supported_formats: tuple = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.index_dir = self.cache_dir / "index"

        # GLM-OCR doesn't have a GGUF version on HuggingFace yet
        # Using a small multimodal model alternative
        self.glm_ocr = ModelConfig(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=4096,
        )
        self.llama_3_1 = ModelConfig(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_ctx=8192,
        )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
