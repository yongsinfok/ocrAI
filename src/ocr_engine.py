"""OCR processing using Tesseract."""
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
    """OCR processing engine using Tesseract."""

    def __init__(self, model_manager: Optional['ModelManager'] = None, cache_dir: Optional[Path] = None):
        """Initialize OCR engine.

        Args:
            model_manager: Not used for Tesseract (kept for compatibility)
            cache_dir: Directory for caching OCR results
        """
        self.cache_dir = cache_dir or Path("cache/ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        """Process an image with OCR using Tesseract.

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

        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            raise ImportError(
                "pytesseract is required for OCR. Install it with: pip install pytesseract\n"
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
                self._save_to_cache(image_bytes, result)

            return result

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise

    def _extract_tables(self, text: str) -> List[List[List[str]]]:
        """Extract table structures from OCR text.

        This is a simplified implementation. For better table extraction,
        consider using pdfplumber or the LLM to parse the OCR text into tables.
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
