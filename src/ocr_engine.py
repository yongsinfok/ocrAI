"""OCR processing using GLM-OCR SDK with Tesseract fallback."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
import io
import logging
import hashlib
import json
import base64

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    tables: List[List[List[str]]]  # List of tables
    confidence: float
    raw_output: Dict[str, Any]


class GLMOCRClient:
    """GLM-OCR SDK client wrapper."""

    def __init__(self, config):
        """Initialize GLM-OCR client.

        Args:
            config: GLMOCRConfig instance
        """
        self.config = config
        self._parser = None
        self._available = None
        self._error_message = None

    @property
    def available(self) -> bool:
        """Check if GLM-OCR SDK is available."""
        if self._available is not None:
            return self._available

        try:
            from glmocr import GlmOcr
            self._available = True
            return True
        except ImportError:
            self._error_message = (
                "GLM-OCR SDK not installed. Install with:\n"
                "  pip install git+https://github.com/zai-org/GLM-OCR.git\n"
                "Or clone and install in editable mode."
            )
            self._available = False
            return False
        except Exception as e:
            self._error_message = f"GLM-OCR SDK error: {e}"
            self._available = False
            return False

    @property
    def connected(self) -> bool:
        """Check if GLM-OCR service is connected/reachable."""
        if not self.available:
            return False

        try:
            # Try to create a parser (this will validate the config)
            if self._parser is None:
                self._get_parser()
            return True
        except Exception as e:
            self._error_message = str(e)
            return False

    def _get_parser(self):
        """Get or create GLM-OCR parser instance."""
        if self._parser is not None:
            return self._parser

        try:
            from glmocr import GlmOcr
        except ImportError as e:
            raise ImportError(
                "GLM-OCR SDK not installed. Install with:\n"
                "  pip install git+https://github.com/zai-org/GLM-OCR.git"
            ) from e

        # Prepare config for GLM-OCR SDK
        sdk_config = {
            "pipeline": {
                "ocr_api": {
                    "api_host": self.config.api_host,
                    "api_port": self.config.api_port,
                    "connect_timeout": self.config.connect_timeout,
                    "request_timeout": self.config.request_timeout,
                },
                "page_loader": {
                    "max_pixels": self.config.max_pixels,
                    "min_pixels": self.config.min_pixels,
                    "image_format": self.config.image_format,
                },
                "result_formatter": {
                    "output_format": self.config.output_format,
                },
                "enable_layout": self.config.enable_layout,
            }
        }

        # Add MaaS config if enabled
        if self.config.mode == "maas":
            if not self.config.api_key:
                raise ValueError(
                    "GLM-OCR MaaS mode requires API key. Set GLM_OCR_API_KEY "
                    "or ZHIPU_API_KEY environment variable, or get one from "
                    "https://open.bigmodel.cn"
                )
            sdk_config["pipeline"]["maas"] = {
                "enabled": True,
                "api_key": self.config.api_key,
            }

        # Create parser with config
        # Note: GLM-OCR SDK loads config from file, so we create a temp config
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sdk_config, f)
            config_path = f.name

        try:
            self._parser = GlmOcr(config_path=config_path)
            return self._parser
        except Exception as e:
            # Provide helpful error messages
            if self.config.mode == "local":
                raise ConnectionError(
                    f"Cannot connect to GLM-OCR local server at {self.config.api_host}:{self.config.api_port}.\n"
                    f"Make sure vLLM/SGLang is running:\n"
                    f"  vllm serve zai-org/GLM-OCR --port {self.config.api_port} --allowed-local-media-path /\n"
                    f"Or use MaaS mode (set GLM_OCR_MODE=maas and provide API key).\n"
                    f"Original error: {e}"
                ) from e
            elif self.config.mode == "maas":
                if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                    raise ValueError(
                        f"GLM-OCR MaaS API key invalid or not set.\n"
                        f"Get an API key from https://open.bigmodel.cn\n"
                        f"Set GLM_OCR_API_KEY environment variable.\n"
                        f"Original error: {e}"
                    ) from e
                raise ConnectionError(
                    f"Cannot connect to GLM-OCR MaaS API.\n"
                    f"Check your API key and network connection.\n"
                    f"Original error: {e}"
                ) from e
            raise
        finally:
            # Clean up temp config
            try:
                Path(config_path).unlink()
            except:
                pass

    def parse(self, image_input: Any) -> "GLMOCRResult":
        """Parse an image with GLM-OCR.

        Args:
            image_input: File path, bytes, or PIL Image

        Returns:
            GLMOCRResult object
        """
        parser = self._get_parser()
        return parser.parse(image_input)

    @property
    def error_message(self) -> Optional[str]:
        """Get the last error message."""
        return self._error_message


class OCREngine:
    """OCR processing engine using GLM-OCR with Tesseract fallback."""

    def __init__(self, model_manager: Optional['ModelManager'] = None, cache_dir: Optional[Path] = None, config=None):
        """Initialize OCR engine.

        Args:
            model_manager: Not used for GLM-OCR (kept for compatibility)
            cache_dir: Directory for caching OCR results
            config: AppConfig instance (optional, will use default if not provided)
        """
        from config import config as default_config

        self.config = config or default_config
        self.cache_dir = cache_dir or self.config.cache_dir / "ocr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GLM-OCR client if enabled
        self.glm_ocr_client: Optional[GLMOCRClient] = None
        self.use_glm_ocr = False

        if self.config.glm_ocr.enabled:
            self.glm_ocr_client = GLMOCRClient(self.config.glm_ocr)
            if self.glm_ocr_client.available:
                self.use_glm_ocr = True
                logger.info("GLM-OCR enabled and SDK available")
            else:
                logger.warning(f"GLM-OCR enabled but not available: {self.glm_ocr_client.error_message}")

    @property
    def glm_ocr_status(self) -> Dict[str, Any]:
        """Get GLM-OCR status information."""
        if not self.config.glm_ocr.enabled:
            return {
                "enabled": False,
                "mode": None,
                "available": False,
                "connected": False,
                "error": "GLM-OCR disabled in configuration"
            }

        if not self.glm_ocr_client:
            return {
                "enabled": True,
                "mode": self.config.glm_ocr.mode,
                "available": False,
                "connected": False,
                "error": "GLM-OCR client not initialized"
            }

        return {
            "enabled": True,
            "mode": self.config.glm_ocr.mode,
            "available": self.glm_ocr_client.available,
            "connected": self.glm_ocr_client.connected if self.glm_ocr_client.available else False,
            "error": self.glm_ocr_client.error_message if not self.glm_ocr_client.connected else None
        }

    def _get_cache_path(self, image_bytes: bytes, engine: str) -> Path:
        """Get cache path for an image."""
        h = hashlib.md5(image_bytes, usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{h}_{engine}.json"

    def _load_from_cache(self, image_bytes: bytes, engine: str) -> Optional[OCRResult]:
        """Load OCR result from cache."""
        cache_path = self._get_cache_path(image_bytes, engine)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return OCRResult(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, image_bytes: bytes, result: OCRResult, engine: str):
        """Save OCR result to cache."""
        cache_path = self._get_cache_path(image_bytes, engine)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'text': result.text,
                    'tables': result.tables,
                    'confidence': result.confidence,
                    'raw_output': result.raw_output
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _process_with_glm_ocr(self, image_bytes: bytes, use_cache: bool = True) -> OCRResult:
        """Process image with GLM-OCR.

        Args:
            image_bytes: Image data as bytes
            use_cache: Whether to use cached results

        Returns:
            OCRResult
        """
        if use_cache:
            cached = self._load_from_cache(image_bytes, "glm_ocr")
            if cached:
                logger.info("Using cached GLM-OCR result")
                return cached

        if not self.glm_ocr_client or not self.glm_ocr_client.available:
            raise RuntimeError("GLM-OCR not available")

        try:
            # Create temp file for GLM-OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(image_bytes)
                temp_path = f.name

            try:
                result_obj = self.glm_ocr_client.parse(temp_path)

                # Extract text from result
                text = result_obj.markdown if hasattr(result_obj, 'markdown') else ""
                json_result = result_obj.json_result if hasattr(result_obj, 'json_result') else []

                # Extract tables from JSON result
                tables = self._extract_tables_from_glm_result(json_result, text)

                result = OCRResult(
                    text=text.strip(),
                    tables=tables,
                    confidence=0.95,  # GLM-OCR typically has high confidence
                    raw_output={'glm_ocr_result': json_result, 'markdown': text}
                )

                if use_cache:
                    self._save_to_cache(image_bytes, result, "glm_ocr")

                return result

            finally:
                # Clean up temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass

        except Exception as e:
            logger.error(f"GLM-OCR processing failed: {e}")
            raise

    def _process_with_tesseract(self, image_bytes: bytes, use_cache: bool = True) -> OCRResult:
        """Process image with Tesseract (fallback).

        Args:
            image_bytes: Image data as bytes
            use_cache: Whether to use cached results

        Returns:
            OCRResult
        """
        if use_cache:
            cached = self._load_from_cache(image_bytes, "tesseract")
            if cached:
                logger.info("Using cached Tesseract result")
                return cached

        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            raise ImportError(
                "pytesseract is required for Tesseract OCR. Install it with: pip install pytesseract\n"
                "Also make sure Tesseract OCR is installed on your system:\n"
                "- Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "- Linux: sudo apt install tesseract-ocr\n"
                "- macOS: brew install tesseract"
            )

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Get OCR data with confidence scores
            data = pytesseract.image_to_data(image, output_type=Output.DICT)

            # Extract text
            text = pytesseract.image_to_string(image)

            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if str(conf).isdigit()]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5

            # Parse tables from OCR text
            tables = self._extract_tables(text)

            result = OCRResult(
                text=text.strip(),
                tables=tables,
                confidence=avg_confidence,
                raw_output={'data': data}
            )

            if use_cache:
                self._save_to_cache(image_bytes, result, "tesseract")

            return result

        except Exception as e:
            logger.error(f"Tesseract OCR processing failed: {e}")
            raise

    def process_image(self, image_bytes: bytes, use_cache: bool = True, force_tesseract: bool = False) -> OCRResult:
        """Process an image with OCR.

        Args:
            image_bytes: Image data as bytes
            use_cache: Whether to use cached results
            force_tesseract: Force use of Tesseract instead of GLM-OCR

        Returns:
            OCRResult

        Raises:
            RuntimeError: If both GLM-OCR and Tesseract are unavailable
        """
        # Try GLM-OCR first if enabled and not forced to use Tesseract
        if self.use_glm_ocr and not force_tesseract:
            try:
                return self._process_with_glm_ocr(image_bytes, use_cache)
            except Exception as e:
                logger.warning(f"GLM-OCR failed, falling back to Tesseract: {e}")
                # Fall through to Tesseract

        # Use Tesseract as fallback
        try:
            return self._process_with_tesseract(image_bytes, use_cache)
        except Exception as e:
            raise RuntimeError(
                f"OCR processing failed. Both GLM-OCR and Tesseract are unavailable.\n"
                f"GLM-OCR error: {self.glm_ocr_client.error_message if self.glm_ocr_client else 'Not configured'}\n"
                f"Tesseract error: {e}"
            ) from e

    def _extract_tables_from_glm_result(self, json_result: List, text: str) -> List[List[List[str]]]:
        """Extract tables from GLM-OCR JSON result.

        Args:
            json_result: GLM-OCR JSON output
            text: Markdown text output

        Returns:
            List of tables
        """
        tables = []

        # Try to parse tables from JSON result
        if json_result and isinstance(json_result, list):
            for page_result in json_result:
                if isinstance(page_result, list):
                    # Group table cells together
                    table_rows = {}
                    for item in page_result:
                        if isinstance(item, dict):
                            label = item.get('label', '')
                            content = item.get('content', '')
                            bbox = item.get('bbox_2d')

                            if label == 'table' or 'table' in label.lower():
                                # This is a table region, try to parse it
                                if content:
                                    # Simple line-based table parsing
                                    lines = content.split('\n')
                                    table = []
                                    for line in lines:
                                        if line.strip():
                                            # Try to split by common table separators
                                            cells = [c.strip() for c in line.replace('|', '\t').split('\t')]
                                            cells = [c for c in cells if c]
                                            if cells:
                                                table.append(cells)
                                    if table:
                                        tables.append(table)

        # If no tables found in JSON, try parsing from markdown
        if not tables:
            tables = self._extract_tables_from_markdown(text)

        return tables

    def _extract_tables_from_markdown(self, text: str) -> List[List[List[str]]]:
        """Extract tables from markdown text.

        Args:
            text: Markdown text with tables

        Returns:
            List of tables
        """
        tables = []
        lines = text.split('\n')
        current_table = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Check for markdown table row
            if '|' in stripped and stripped.startswith('|'):
                if not in_table:
                    in_table = True
                    current_table = []
                # Skip separator lines
                if not set(stripped) <= {'|', '-', ' ', ':'}:
                    cells = [cell.strip() for cell in stripped.split('|')]
                    cells = [c for c in cells if c]
                    if cells:
                        current_table.append(cells)
            else:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                in_table = False

        if current_table:
            tables.append(current_table)

        return tables

    def _extract_tables(self, text: str) -> List[List[List[str]]]:
        """Extract table structures from OCR text.

        This is a simplified implementation. For better table extraction,
        consider using pdfplumber or the LLM to parse the OCR text into tables.
        """
        # First try markdown table extraction
        tables = self._extract_tables_from_markdown(text)

        if tables:
            return tables

        # Basic table detection - look for tab-separated data
        tables = []
        lines = text.split('\n')

        current_table = []
        for line in lines:
            if '\t' in line or '|' in line:
                # Likely a table row - replace pipes with tabs, split, and filter empty cells
                cells = [cell.strip() for cell in line.replace('|', '\t').split('\t')]
                # Filter out empty cells
                cells = [cell for cell in cells if cell]
                if cells:
                    current_table.append(cells)
            else:
                if current_table:
                    tables.append(current_table)
                    current_table = []

        if current_table:
            tables.append(current_table)

        return tables

    def process_page_from_pdf(self, pdf_path: Path, page_num: int, use_cache: bool = True) -> OCRResult:
        """Process a PDF page by rendering it as an image first.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            use_cache: Whether to use cached results

        Returns:
            OCRResult
        """
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} not found in PDF")

            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            return self.process_image(img_bytes, use_cache=use_cache)
        finally:
            doc.close()
