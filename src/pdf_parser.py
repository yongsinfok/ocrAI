# src/pdf_parser.py
"""PDF parsing and structure extraction."""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class PageType(Enum):
    """Page content type."""
    TEXT = "text"         # Digital text PDF
    IMAGE = "image"       # Scanned/image-based
    MIXED = "mixed"       # Both text and images

@dataclass
class PageInfo:
    """Information about a page."""
    page_num: int
    page_type: PageType
    text: str
    image_count: int
    text_density: float  # Characters per page area

@dataclass
class DocumentInfo:
    """Information about a parsed document."""
    path: Path
    total_pages: int
    pages: List[PageInfo]
    structure_index: Dict[str, int]  # Section/title -> page number

class PDFParser:
    """Parse PDF documents and extract structure."""

    # Regex patterns for structure detection
    SECTION_PATTERNS = [
        r'^Table\s+(\d+\.?\d*)',  # Table 1.2.1
        r'^(\d+\.?\d*)\s+(.+)',   # 1.2.1 Section Title
        r'^第[一二三四五六七八九十\d]+[章节]',  # Chinese chapters
    ]

    def __init__(self):
        """Initialize parser."""
        self.docs: Dict[Path, DocumentInfo] = {}

    def parse(self, pdf_path: Path) -> DocumentInfo:
        """Parse a PDF document."""
        logger.info(f"Parsing {pdf_path}")

        doc = fitz.open(str(pdf_path))
        try:
            pages = []
            structure_index = {}

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_info = self._parse_page(page, page_num)
                pages.append(page_info)

                # Build structure index from this page
                self._index_structure(page_info, structure_index)

            doc_info = DocumentInfo(
                path=pdf_path,
                total_pages=len(doc),
                pages=pages,
                structure_index=structure_index
            )

            self.docs[pdf_path] = doc_info
            logger.info(f"Parsed {len(pages)} pages, found {len(structure_index)} sections")
            return doc_info
        finally:
            doc.close()

    def _parse_page(self, page, page_num: int) -> PageInfo:
        """Parse a single page."""
        text = page.get_text()
        images = page.get_images()
        image_count = len(images)

        # Calculate text density
        rect = page.rect
        area = rect.width * rect.height
        text_density = len(text.strip()) / area if area > 0 else 0

        # Detect page type
        if image_count > 0 and len(text.strip()) < 100:
            page_type = PageType.IMAGE
        elif image_count > 0:
            page_type = PageType.MIXED
        else:
            page_type = PageType.TEXT

        return PageInfo(
            page_num=page_num,
            page_type=page_type,
            text=text,
            image_count=image_count,
            text_density=text_density
        )

    def _index_structure(self, page_info: PageInfo, index: Dict[str, int]):
        """Extract structural elements from page text."""
        lines = page_info.text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    # Use the line as key, store page number
                    index[line] = page_info.page_num
                    break

    def get_page_info(self, pdf_path: Path, page_num: int) -> Optional[PageInfo]:
        """Get information about a specific page."""
        if pdf_path not in self.docs:
            self.parse(pdf_path)

        doc_info = self.docs[pdf_path]
        if 0 <= page_num < len(doc_info.pages):
            return doc_info.pages[page_num]
        return None

    def search_structure(self, pdf_path: Path, query: str) -> List[Tuple[str, int]]:
        """Search structure index for matching entries."""
        if pdf_path not in self.docs:
            self.parse(pdf_path)

        doc_info = self.docs[pdf_path]
        results = []

        for key, page_num in doc_info.structure_index.items():
            if query.lower() in key.lower():
                results.append((key, page_num))

        return sorted(results, key=lambda x: x[1])

    def extract_page_text(self, pdf_path: Path, page_num: int) -> str:
        """Extract text from a specific page."""
        if pdf_path not in self.docs:
            self.parse(pdf_path)

        doc_info = self.docs[pdf_path]
        if 0 <= page_num < len(doc_info.pages):
            return doc_info.pages[page_num].text
        return ""

    def get_page_image(self, pdf_path: Path, page_num: int) -> Optional[bytes]:
        """Render page as image for OCR."""
        doc = fitz.open(str(pdf_path))
        try:
            if page_num < len(doc):
                page = doc[page_num]
                # Render at 2x resolution for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                return img_bytes
            return None
        finally:
            doc.close()
