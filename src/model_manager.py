"""Model download and management."""
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages GGUF model download and loading."""

    def __init__(self, config):
        """Initialize model manager."""
        self.config = config
        self.models_dir = config.models_dir
        self._loaded_models: Dict[str, Any] = {}

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        model_config = getattr(self.config, model_name)
        return self.models_dir / model_config.filename

    def model_exists(self, model_name: str) -> bool:
        """Check if model file exists locally."""
        return self.get_model_path(model_name).exists()

    def download_model(self, model_name: str, progress_callback=None) -> Path:
        """Download model from HuggingFace."""
        if self.model_exists(model_name):
            logger.info(f"Model {model_name} already exists")
            return self.get_model_path(model_name)

        model_config = getattr(self.config, model_name)

        logger.info(f"Downloading {model_name} from {model_config.repo_id}...")

        try:
            path = hf_hub_download(
                repo_id=model_config.repo_id,
                filename=model_config.filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded to {path}")
            return Path(path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def ensure_model(self, model_name: str) -> Path:
        """Ensure model is available, download if necessary."""
        if not self.model_exists(model_name):
            return self.download_model(model_name)
        return self.get_model_path(model_name)

    def load_llama_model(self, model_name: str):
        """Load a GGUF model with llama-cpp-python."""
        from llama_cpp import Llama

        model_path = self.ensure_model(model_name)
        model_config = getattr(self.config, model_name)

        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        logger.info(f"Loading model {model_name} from {model_path}")

        model = Llama(
            model_path=str(model_path),
            n_gpu_layers=model_config.n_gpu_layers,
            n_ctx=model_config.n_ctx,
            verbose=model_config.verbose,
        )

        self._loaded_models[model_name] = model
        return model

    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            import gc
            gc.collect()

    def get_model(self, model_name: str):
        """Get or load a model."""
        if model_name not in self._loaded_models:
            return self.load_llama_model(model_name)
        return self._loaded_models[model_name]
