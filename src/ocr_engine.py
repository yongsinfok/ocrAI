"""OCR processing using GLM-OCR."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
import io
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    tables: List[List[List[str]]]  # List of tables
    confidence: float
    raw_output: Dict[str, Any]

class OCREngine:
    """OCR processing engine using GLM-OCR."""

    def __init__(self, model_manager: 'ModelManager', cache_dir: Optional[Path] = None):
        """Initialize OCR engine.

        Args:
            model_manager: ModelManager instance
            cache_dir: Directory for caching OCR results
        """
        self.model_manager = model_manager
        self.cache_dir = cache_dir or Path("cache/ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _get_model(self):
        """Get or load GLM-OCR model."""
        if self._model is None:
            self._model = self.model_manager.get_model("glm_ocr")
        return self._model

    def _get_cache_path(self, image_bytes: bytes) -> Path:
        """Get cache path for an image."""
        h = hashlib.md5(image_bytes, usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _load_from_cache(self, image_bytes: bytes) -> Optional[OCRResult]:
        """Load OCR result from cache."""
        cache_path = self._get_cache_path(image_bytes)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return OCRResult(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, image_bytes: bytes, result: OCRResult):
        """Save OCR result to cache."""
        cache_path = self._get_cache_path(image_bytes)
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

    def process_image(self, image_bytes: bytes, use_cache: bool = True) -> OCRResult:
        """Process an image with OCR.

        Args:
            image_bytes: Image data as bytes
            use_cache: Whether to use cached results

        Returns:
            OCRResult
        """
        if use_cache:
            cached = self._load_from_cache(image_bytes)
            if cached:
                logger.info("Using cached OCR result")
                return cached

        model = self._get_model()

        # Convert image to base64 for model input
        import base64
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare prompt for OCR
        prompt = """Extract all text and tables from this image. For tables, return them in a structured format.
        If there are tables, identify them clearly with their headers and row data."""

        try:
            response = model(
                prompt,
                images=[img_b64],
                max_tokens=2048,
                temperature=0.1,
            )

            text = response['choices'][0]['text']

            # Parse tables from response (basic implementation)
            tables = self._extract_tables(text)

            result = OCRResult(
                text=text,
                tables=tables,
                confidence=0.9,  # Placeholder
                raw_output=response
            )

            if use_cache:
                self._save_to_cache(image_bytes, result)

            return result

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise

    def _extract_tables(self, text: str) -> List[List[List[str]]]:
        """Extract table structures from OCR text.

        This is a simplified implementation. GLM-OCR should return
        structured table data directly.
        """
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
