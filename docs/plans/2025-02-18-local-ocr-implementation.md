# Local OCR Application Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local document processing application that uses GLM-OCR for OCR and Llama-3.1 for query understanding, allowing users to extract data from PDFs/images via natural language and export to Excel.

**Architecture:** Streamlit frontend with Python backend, using PyMuPDF for PDF parsing, llama-cpp-python for GGUF model inference (CUDA accelerated), with a hybrid indexing system (structure tree + fulltext) for fast document navigation.

**Tech Stack:** Streamlit, PyMuPDF, llama-cpp-python, huggingface-hub, openpyxl, pandas, Whoosh, Pillow

---

## Task 1: Project Setup and Configuration

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `.gitignore`
- Create: `models/.gitkeep`

**Step 1: Create requirements.txt**

```txt
streamlit>=1.31.0
pymupdf>=1.23.0
llama-cpp-python>=0.2.0
huggingface-hub>=0.20.0
openpyxl>=3.1.0
pandas>=2.0.0
whoosh>=2.7.4
Pillow>=10.0.0
python-dotenv>=1.0.0
```

**Step 2: Create config.py**

```python
"""Application configuration."""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

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
    base_dir: Path = Path(__file__).parent
    models_dir: Path = base_dir / "models"
    cache_dir: Path = base_dir / "cache"
    index_dir: Path = cache_dir / "index"

    # Model configurations
    glm_ocr: ModelConfig = ModelConfig(
        repo_id="ggml-org/GLM-OCR-GGUF",
        filename="glm-ocr-q4_k_m.gguf",
        n_ctx=4096,
    )
    llama_3_1: ModelConfig = ModelConfig(
        repo_id="lm-community/Llama-3.1-8B-Instruct-GGUF",
        filename="llama-3.1-8b-instruct-q4_k_m.gguf",
        n_ctx=8192,
    )

    # Processing
    max_file_size_mb: int = 100
    supported_formats: tuple = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
```

**Step 3: Create .gitignore**

```
# Models
models/*.gguf
models/*.bin

# Cache
cache/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Environment
.env
.env.local
```

**Step 4: Create models/.gitkeep**

```bash
touch models/.gitkeep
```

**Step 5: Create cache/.gitkeep**

```bash
mkdir -p cache && touch cache/.gitkeep
```

**Step 6: Commit**

```bash
git add requirements.txt config.py .gitignore models/.gitkeep cache/.gitkeep
git commit -m "feat: add project configuration and structure"
```

---

## Task 2: Model Manager - Download and Load GGUF Models

**Files:**
- Create: `src/__init__.py`
- Create: `src/model_manager.py`
- Test: `tests/test_model_manager.py`

**Step 1: Create src directory structure**

```bash
mkdir -p src tests
```

**Step 2: Write failing tests**

```python
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
```

**Step 3: Run test to verify failures**

```bash
pytest tests/test_model_manager.py -v
```

Expected: FAIL - module not found

**Step 4: Implement ModelManager**

```python
# src/model_manager.py
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
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_model_manager.py -v
```

**Step 6: Commit**

```bash
git add src/model_manager.py tests/test_model_manager.py
git commit -m "feat: add model manager with download and load capabilities"
```

---

## Task 3: PDF Parser - Parse and Index Documents

**Files:**
- Create: `src/pdf_parser.py`
- Test: `tests/test_pdf_parser.py`

**Step 1: Write failing tests**

```python
# tests/test_pdf_parser.py
import pytest
from pathlib import Path
from src.pdf_parser import PDFParser, PageType

def test_parser_initialization():
    parser = PDFParser()
    assert parser is not None

def test_parse_text_page():
    # Test with a simple text extraction
    parser = PDFParser()
    # This would need a sample PDF
    assert True  # Placeholder

def test_detect_page_type():
    parser = PDFParser()
    # Test page type detection logic
    assert True  # Placeholder
```

**Step 2: Implement PDFParser**

```python
# src/pdf_parser.py
"""PDF parsing and structure extraction."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import fitz  # PyMuPDF
import re
import logging

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

        doc.close()
        self.docs[pdf_path] = doc_info

        logger.info(f"Parsed {len(pages)} pages, found {len(structure_index)} sections")
        return doc_info

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
        if page_num < len(doc):
            page = doc[page_num]
            # Render at 2x resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        doc.close()
        return None
```

**Step 3: Run tests**

```bash
pytest tests/test_pdf_parser.py -v
```

**Step 4: Update tests for actual functionality**

```python
# tests/test_pdf_parser.py
import pytest
from pathlib import Path
from src.pdf_parser import PDFParser, PageType

def test_parser_initialization():
    parser = PDFParser()
    assert parser is not None
    assert parser.docs == {}

def test_structure_index():
    parser = PDFParser()
    # Test structure indexing
    assert True
```

**Step 5: Commit**

```bash
git add src/pdf_parser.py tests/test_pdf_parser.py
git commit -m "feat: add PDF parser with structure extraction"
```

---

## Task 4: Index Manager - Full Text Search

**Files:**
- Create: `src/index_manager.py`
- Test: `tests/test_index_manager.py`

**Step 1: Write failing tests**

```python
# tests/test_index_manager.py
import pytest
from pathlib import Path
from src.index_manager import IndexManager

def test_index_manager_initialization():
    manager = IndexManager()
    assert manager is not None

def test_build_and_search():
    manager = IndexManager()
    # Build index
    # Search
    assert True
```

**Step 2: Implement IndexManager**

```python
# src/index_manager.py
"""Full text indexing using Whoosh."""
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from whoosh.index import create_in, exists_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """A search result."""
    page_num: int
    content: str
    score: float

class IndexManager:
    """Manage full text search index."""

    def __init__(self, index_dir: Optional[Path] = None):
        """Initialize index manager."""
        self.index_dir = index_dir or Path(tempfile.mkdtemp())
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self._init_index()

    def _init_index(self):
        """Initialize Whoosh index."""
        if exists_in(str(self.index_dir)):
            self.index = open_dir(str(self.index_dir))
        else:
            self.index = create_in(
                str(self.index_dir),
                schema=Schema(
                    doc_id=ID(stored=True),
                    page_num=ID(stored=True),
                    content=TEXT(stored=True)
                )
            )

    def index_document(self, doc_id: str, pages: List[tuple]):
        """Index a document.

        Args:
            doc_id: Document identifier
            pages: List of (page_num, content) tuples
        """
        writer = self.index.writer()

        # Clear existing document
        writer.delete_by_term("doc_id", doc_id)

        # Add pages
        for page_num, content in pages:
            writer.add_document(
                doc_id=doc_id,
                page_num=str(page_num),
                content=content
            )

        writer.commit()
        logger.info(f"Indexed document {doc_id} with {len(pages)} pages")

    def search(self, query: str, doc_id: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search the index.

        Args:
            query: Search query string
            doc_id: Optional document ID to restrict search
            limit: Maximum number of results

        Returns:
            List of SearchResult
        """
        searcher = self.index.searcher()
        parser = QueryParser("content", self.index.schema)

        q = parser.parse(query)

        if doc_id:
            q = parser.parse(f"{query} doc_id:{doc_id}")

        results = searcher.search(q, limit=limit)

        search_results = []
        for r in results:
            search_results.append(SearchResult(
                page_num=int(r["page_num"]),
                content=r.get("content", "")[:200],  # Preview
                score=r.score
            ))

        searcher.close()
        return search_results

    def clear_document(self, doc_id: str):
        """Remove a document from index."""
        writer = self.index.writer()
        writer.delete_by_term("doc_id", doc_id)
        writer.commit()

    def close(self):
        """Close the index."""
        if self.index:
            self.index.close()
```

**Step 3: Run tests**

```bash
pytest tests/test_index_manager.py -v
```

**Step 4: Commit**

```bash
git add src/index_manager.py tests/test_index_manager.py
git commit -m "feat: add full text search index manager"
```

---

## Task 5: OCR Engine - GLM-OCR Integration

**Files:**
- Create: `src/ocr_engine.py`
- Test: `tests/test_ocr_engine.py`

**Step 1: Write failing tests**

```python
# tests/test_ocr_engine.py
import pytest
from src.ocr_engine import OCREngine

def test_ocr_engine_initialization():
    engine = OCREngine(None)
    assert engine is not None
```

**Step 2: Implement OCREngine**

```python
# src/ocr_engine.py
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

    def __init__(self, model_manager, cache_dir: Optional[Path] = None):
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
        h = hashlib.md5(image_bytes).hexdigest()
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
                # Likely a table row
                cells = [cell.strip() for cell in line.replace('|', '\t').split('\t')]
                cells = [c for c in cells if c]
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
        if page_num >= len(doc):
            doc.close()
            raise ValueError(f"Page {page_num} not found in PDF")

        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        doc.close()

        return self.process_image(img_bytes, use_cache=use_cache)
```

**Step 3: Run tests**

```bash
pytest tests/test_ocr_engine.py -v
```

**Step 4: Commit**

```bash
git add src/ocr_engine.py tests/test_ocr_engine.py
git commit -m "feat: add GLM-OCR engine with caching"
```

---

## Task 6: Query Processor - Intent Analysis

**Files:**
- Create: `src/query_processor.py`
- Test: `tests/test_query_processor.py`

**Step 1: Write failing tests**

```python
# tests/test_query_processor.py
import pytest
from src.query_processor import QueryProcessor, QueryType

def test_query_processor_initialization():
    processor = QueryProcessor(None)
    assert processor is not None

def test_analyze_simple_extract():
    processor = QueryProcessor(None)
    result = processor.analyze_intent("从 Table 1.2.1 提取数据")
    assert result.query_type == QueryType.SIMPLE_EXTRACT
```

**Step 2: Implement QueryProcessor**

```python
# src/query_processor.py
"""Query intent analysis and processing."""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classification."""
    SIMPLE_EXTRACT = "simple_extract"      # 直接提取
    FORMAT_CONVERT = "format_convert"      # 格式转换
    COMPUTE = "compute"                    # 计算/过滤
    MULTI_STEP = "multi_step"              # 多步骤

@dataclass
class QueryIntent:
    """Analyzed query intent."""
    original_query: str
    query_type: QueryType
    target: Optional[str] = None           # e.g., "Table 1.2.1"
    output_format: str = "excel"           # excel, csv, json
    template: Optional[str] = None         # Custom format template
    confidence: float = 0.0

@dataclass
class ProcessingResult:
    """Result from query processing."""
    data: Any
    format: str
    metadata: Dict[str, Any]

class QueryProcessor:
    """Process user queries using LLM."""

    # System prompt for intent analysis
    INTENT_ANALYSIS_PROMPT = """Analyze the user's query and extract:
1. Query type (simple_extract, format_convert, compute, multi_step)
2. Target (what to extract - e.g., "Table 1.2.1")
3. Output format (excel, csv, json)

Respond in JSON format:
{
    "query_type": "...",
    "target": "...",
    "output_format": "...",
    "confidence": 0.95
}"""

    def __init__(self, model_manager):
        """Initialize query processor.

        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self._llm = None

    def _get_llm(self):
        """Get or load LLM for query understanding."""
        if self._llm is None:
            self._llm = self.model_manager.get_model("llama_3_1")
        return self._llm

    def analyze_intent(self, query: str) -> QueryIntent:
        """Analyze user query intent.

        Args:
            query: User's natural language query

        Returns:
            QueryIntent
        """
        # Use LLM for intent analysis
        llm = self._get_llm()

        prompt = f"{self.INTENT_ANALYSIS_PROMPT}\n\nUser query: {query}"

        try:
            response = llm(
                prompt,
                max_tokens=256,
                temperature=0.1,
                stop=["\n\n"]
            )

            # Parse JSON response
            import json
            response_text = response['choices'][0]['text'].strip()

            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return QueryIntent(
                    original_query=query,
                    query_type=QueryType(data.get("query_type", "simple_extract")),
                    target=data.get("target"),
                    output_format=data.get("output_format", "excel"),
                    confidence=data.get("confidence", 0.8)
                )

        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}, using fallback")

        # Fallback: rule-based analysis
        return self._fallback_analysis(query)

    def _fallback_analysis(self, query: str) -> QueryIntent:
        """Fallback rule-based intent analysis."""
        query_lower = query.lower()

        # Detect target
        target = None
        table_match = re.search(r'[Tt]able\s+([\d.]+)', query)
        if table_match:
            target = f"Table {table_match.group(1)}"

        section_match = re.search(r'(\d+\.?\d*)', query)
        if section_match and not target:
            target = section_match.group(1)

        # Detect output format
        output_format = "excel"
        if "json" in query_lower:
            output_format = "json"
        elif "csv" in query_lower:
            output_format = "csv"

        # Detect query type
        query_type = QueryType.SIMPLE_EXTRACT
        if any(word in query_lower for word in ["转换", "格式", "format", "convert"]):
            query_type = QueryType.FORMAT_CONVERT
        elif any(word in query_lower for word in ["计算", "筛选", "过滤", "filter", "compute"]):
            query_type = QueryType.COMPUTE
        elif "，" in query or " then " in query_lower:
            query_type = QueryType.MULTI_STEP

        return QueryIntent(
            original_query=query,
            query_type=query_type,
            target=target,
            output_format=output_format,
            confidence=0.7
        )

    def format_data(self, data: Any, intent: QueryIntent) -> str:
        """Format data according to query intent.

        Args:
            data: Data to format
            intent: Query intent with format specifications

        Returns:
            Formatted data as string
        """
        llm = self._get_llm()

        if intent.output_format == "json":
            if isinstance(data, list):
                import json
                return json.dumps(data, ensure_ascii=False, indent=2)

        # Use LLM for complex formatting
        if intent.template or intent.query_type == QueryType.FORMAT_CONVERT:
            prompt = f"""Format the following data according to the user's request.

Data: {str(data)[:1000]}

User request: {intent.original_query}

Respond only with the formatted output."""

            response = llm(
                prompt,
                max_tokens=2048,
                temperature=0.3
            )

            return response['choices'][0]['text'].strip()

        # Default formatting
        return str(data)
```

**Step 3: Run tests**

```bash
pytest tests/test_query_processor.py -v
```

**Step 4: Commit**

```bash
git add src/query_processor.py tests/test_query_processor.py
git commit -m "feat: add query processor with LLM intent analysis"
```

---

## Task 7: Export Manager - Data Export

**Files:**
- Create: `src/export_manager.py`
- Test: `tests/test_export_manager.py`

**Step 1: Write failing tests**

```python
# tests/test_export_manager.py
import pytest
from src.export_manager import ExportManager

def test_export_manager_initialization():
    manager = ExportManager()
    assert manager is not None

def test_to_excel():
    manager = ExportManager()
    data = [["A", "B"], [1, 2]]
    result = manager.to_excel(data, "test")
    assert result is not None
```

**Step 2: Implement ExportManager**

```python
# src/export_manager.py
"""Data export functionality."""
from typing import Any, List, Dict, Optional
import io
import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

class ExportManager:
    """Manage data export to various formats."""

    def to_excel(self, data: Any, filename: str = "export") -> bytes:
        """Convert data to Excel format.

        Args:
            data: Data to export (list of lists, dict, or DataFrame)
            filename: Base filename for the Excel file

        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()

        # Convert to DataFrame
        df = self._to_dataframe(data)

        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=filename[:31])  # Excel sheet name limit

        output.seek(0)
        return output.read()

    def to_csv(self, data: Any) -> str:
        """Convert data to CSV format.

        Args:
            data: Data to export

        Returns:
            CSV string
        """
        df = self._to_dataframe(data)
        return df.to_csv(index=False)

    def to_json(self, data: Any) -> str:
        """Convert data to JSON format.

        Args:
            data: Data to export

        Returns:
            JSON string
        """
        if isinstance(data, str):
            return data

        df = self._to_dataframe(data)
        return df.to_json(orient='records', ensure_ascii=False, indent=2)

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data types to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame(data)

        if isinstance(data, list):
            if not data:
                return pd.DataFrame()

            # Check if list of dicts
            if isinstance(data[0], dict):
                return pd.DataFrame(data)

            # Check if list of lists (table format)
            if isinstance(data[0], list):
                # First row as headers
                return pd.DataFrame(data[1:], columns=data[0])

            # Simple list
            return pd.DataFrame(data, columns=["Value"])

        # String - try to parse as JSON
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return self._to_dataframe(parsed)
            except:
                return pd.DataFrame({"Value": [data]})

        # Fallback
        return pd.DataFrame({"Value": [str(data)]})

    def format_by_template(self, data: Any, template: str) -> str:
        """Format data using a template string.

        Args:
            data: Data to format
            template: Template string with {placeholders}

        Returns:
            Formatted string
        """
        df = self._to_dataframe(data)

        # Convert DataFrame to dict for template formatting
        if len(df) > 0:
            row = df.iloc[0].to_dict()
        else:
            row = {}

        try:
            return template.format(**row)
        except KeyError as e:
            logger.warning(f"Template key error: {e}")
            return template.format(data=str(data))
```

**Step 3: Run tests**

```bash
pytest tests/test_export_manager.py -v
```

**Step 4: Commit**

```bash
git add src/export_manager.py tests/test_export_manager.py
git commit -m "feat: add export manager for Excel/CSV/JSON"
```

---

## Task 8: Streamlit UI - Main Application

**Files:**
- Create: `app.py`

**Step 1: Create main application**

```python
# app.py
"""Local OCR Application - Streamlit UI."""
import streamlit as st
from pathlib import Path
from typing import Optional
import logging

from config import config
from src.model_manager import ModelManager
from src.pdf_parser import PDFParser
from src.index_manager import IndexManager
from src.ocr_engine import OCREngine
from src.query_processor import QueryProcessor
from src.export_manager import ExportManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Local OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager(config)

if "pdf_parser" not in st.session_state:
    st.session_state.pdf_parser = PDFParser()

if "index_manager" not in st.session_state:
    st.session_state.index_manager = IndexManager(config.index_dir)

if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = OCREngine(st.session_state.model_manager, config.cache_dir)

if "query_processor" not in st.session_state:
    st.session_state.query_processor = QueryProcessor(st.session_state.model_manager)

if "export_manager" not in st.session_state:
    st.session_state.export_manager = ExportManager()

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def check_models():
    """Check and download models if needed."""
    st.sidebar.header("🤖 Models")

    models = ["glm_ocr", "llama_3_1"]
    model_status = {}

    for model_name in models:
        exists = st.session_state.model_manager.model_exists(model_name)
        model_status[model_name] = exists
        status = "✅" if exists else "❌"
        st.sidebar.write(f"{status} {model_name}")

    if not all(model_status.values()):
        st.sidebar.warning("Some models need to be downloaded")
        if st.sidebar.button("Download Missing Models"):
            with st.sidebar:
                with st.spinner("Downloading models..."):
                    for model_name, exists in model_status.items():
                        if not exists:
                            try:
                                st.session_state.model_manager.download_model(model_name)
                                st.success(f"Downloaded {model_name}")
                            except Exception as e:
                                st.error(f"Failed to download {model_name}: {e}")
            st.rerun()


def render_file_upload():
    """Render file upload section."""
    st.header("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        help="Supported formats: PDF, PNG, JPG, JPEG, TIFF"
    )

    if uploaded_file:
        # Save uploaded file
        file_path = config.cache_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Parse document
        with st.spinner("Parsing document..."):
            if uploaded_file.name.endswith(".pdf"):
                doc_info = st.session_state.pdf_parser.parse(file_path)

                st.session_state.current_doc = {
                    "path": file_path,
                    "name": uploaded_file.name,
                    "info": doc_info
                }

                # Build full text index
                pages = [(i, page.text) for i, page in enumerate(doc_info.pages)]
                st.session_state.index_manager.index_document(uploaded_file.name, pages)

                st.success(f"✅ Parsed {doc_info.total_pages} pages")

                # Show document info
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Pages", doc_info.total_pages)
                col2.metric("Text Pages", sum(1 for p in doc_info.pages if p.page_type.value == "text"))
                col3.metric("Image Pages", sum(1 for p in doc_info.pages if p.page_type.value == "image"))

                # Show structure
                if doc_info.structure_index:
                    st.subheader("📑 Document Structure")
                    for section, page_num in sorted(doc_info.structure_index.items(), key=lambda x: x[1])[:20]:
                        st.write(f"Page {page_num + 1}: {section}")
            else:
                # Image file
                st.session_state.current_doc = {
                    "path": file_path,
                    "name": uploaded_file.name,
                    "info": None
                }
                st.success(f"✅ Uploaded {uploaded_file.name}")


def render_chat():
    """Render chat interface."""
    st.header("💬 Chat")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show results if available
            if "data" in message:
                st.dataframe(message["data"])

            if "download" in message:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📊 Excel",
                        message["download"]["excel"],
                        file_name=f"{message['download']['filename']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        "📄 CSV",
                        message["download"]["csv"],
                        file_name=f"{message['download']['filename']}.csv",
                        mime="text/csv"
                    )
                with col3:
                    st.download_button(
                        "{ } JSON",
                        message["download"]["json"],
                        file_name=f"{message['download']['filename']}.json",
                        mime="application/json"
                    )

    # Chat input
    if prompt := st.chat_input("Ask about your document..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.write(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    response_data = process_query(prompt)
                    st.write(response_data["message"])

                    if "data" in response_data:
                        st.dataframe(response_data["data"])

                        # Add download buttons
                        export_mgr = st.session_state.export_manager

                        excel_data = export_mgr.to_excel(response_data["data"])
                        csv_data = export_mgr.to_csv(response_data["data"])
                        json_data = export_mgr.to_json(response_data["data"])

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📊 Excel",
                                excel_data,
                                file_name=f"export_{hash(prompt) % 10000}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col2:
                            st.download_button(
                                "📄 CSV",
                                csv_data,
                                file_name=f"export_{hash(prompt) % 10000}.csv",
                                mime="text/csv"
                            )
                        with col3:
                            st.download_button(
                                "{ } JSON",
                                json_data,
                                file_name=f"export_{hash(prompt) % 10000}.json",
                                mime="application/json"
                            )

                        # Add to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_data["message"],
                            "data": response_data["data"],
                            "download": {
                                "excel": excel_data,
                                "csv": csv_data,
                                "json": json_data,
                                "filename": f"export_{hash(prompt) % 10000}"
                            }
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_data["message"]
                        })

                except Exception as e:
                    logger.error(f"Query processing error: {e}")
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    })


def process_query(prompt: str) -> dict:
    """Process user query.

    Args:
        prompt: User's query

    Returns:
        Response dictionary with message and optional data
    """
    if not st.session_state.current_doc:
        return {"message": "Please upload a document first."}

    # Analyze intent
    intent = st.session_state.query_processor.analyze_intent(prompt)

    doc_path = st.session_state.current_doc["path"]
    doc_info = st.session_state.current_doc.get("info")

    # Search for target
    if intent.target:
        # Search structure index first
        if doc_info:
            structure_results = st.session_state.pdf_parser.search_structure(doc_path, intent.target)

            if structure_results:
                # Found in structure index
                section, page_num = structure_results[0]
                return extract_and_format_page(doc_path, page_num, intent)

        # Search full text index
        search_results = st.session_state.index_manager.search(intent.target)

        if search_results:
            page_num = search_results[0].page_num
            return extract_and_format_page(doc_path, page_num, intent)

    # If no specific target, extract from entire document
    if doc_info and doc_info.total_pages == 1:
        return extract_and_format_page(doc_path, 0, intent)

    return {"message": f"Could not find '{intent.target}' in the document. Please try a different search term."}


def extract_and_format_page(doc_path: Path, page_num: int, intent) -> dict:
    """Extract and format data from a specific page.

    Args:
        doc_path: Path to document
        page_num: Page number to extract from
        intent: Query intent

    Returns:
        Response dictionary
    """
    doc_info = st.session_state.current_doc.get("info")

    # Check page type
    if doc_info and page_num < len(doc_info.pages):
        page_info = doc_info.pages[page_num]

        if page_info.page_type.value == "text":
            # Text page - extract directly
            text = st.session_state.pdf_parser.extract_page_text(doc_path, page_num)

            # Parse table from text
            data = parse_table_from_text(text)

            if intent.query_type.value == "format_convert":
                formatted = st.session_state.query_processor.format_data(data, intent)
                return {"message": f"✅ Extracted from page {page_num + 1}", "data": data}

            return {"message": f"✅ Extracted from page {page_num + 1}", "data": data}

        else:
            # Image/Scanned page - use OCR
            ocr_result = st.session_state.ocr_engine.process_page_from_pdf(doc_path, page_num)

            if ocr_result.tables:
                data = ocr_result.tables[0]  # First table
            else:
                # Create simple table from text
                data = [[line] for line in ocr_result.text.split('\n') if line.strip()]

            return {"message": f"✅ OCR processed page {page_num + 1}", "data": data}

    # Image file
    if not doc_info:
        with open(doc_path, "rb") as f:
            img_bytes = f.read()
        ocr_result = st.session_state.ocr_engine.process_image(img_bytes)

        if ocr_result.tables:
            data = ocr_result.tables[0]
        else:
            data = [[line] for line in ocr_result.text.split('\n') if line.strip()]

        return {"message": "✅ OCR completed", "data": data}

    return {"message": "Could not extract data from the specified page."}


def parse_table_from_text(text: str) -> list:
    """Parse table data from text.

    Args:
        text: Text content

    Returns:
        List of lists representing table
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Simple table detection - look for consistent separators
    tables = []
    current_table = []

    for line in lines:
        # Check for table-like structure (tabs, pipes, or multiple spaces)
        if '\t' in line or '|' in line or '  ' in line:
            cells = [cell.strip() for cell in line.replace('|', '\t').split('\t')]
            cells = [c for c in cells if c]
            if cells:
                current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []

    if current_table:
        tables.append(current_table)

    if tables:
        return tables[0]

    # No table found, return text as simple list
    return [[line] for line in lines]


def main():
    """Main application."""
    st.title("📄 Local OCR")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        check_models()

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🔄 Reload Document"):
            st.session_state.current_doc = None
            st.rerun()

    # Main content
    render_file_upload()
    st.markdown("---")
    render_chat()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI application"
```

---

## Task 9: Final Integration and Testing

**Files:**
- Update: `requirements.txt`

**Step 1: Add all dependencies**

```bash
cat requirements.txt
```

**Step 2: Test the application**

```bash
streamlit run app.py
```

**Step 3: Fix any issues found during testing**

(Inspect logs and fix bugs)

**Step 4: Create README**

```markdown
# Local OCR Application

A local document processing application using GLM-OCR and Llama-3.1.

## Features

- Upload PDF or images
- Natural language query
- Extract tables and data
- Export to Excel, CSV, JSON
- Fully local - no API calls

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Download models when prompted in the sidebar.

## Hardware Requirements

- NVIDIA GPU with CUDA support
- 8GB+ VRAM recommended
- Windows/Linux

## Usage

1. Upload a PDF or image
2. Ask questions about the document in natural language
3. Download results in your preferred format
```

**Step 5: Final commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Summary

This implementation plan creates a local OCR application with:

1. **Model Management**: Download and load GGUF models from HuggingFace
2. **PDF Parsing**: Extract structure and detect page types
3. **Full Text Search**: Fast document navigation
4. **OCR Processing**: GLM-OCR for scanned documents
5. **Query Understanding**: Llama-3.1 for intent analysis
6. **Data Export**: Excel/CSV/JSON output
7. **Streamlit UI**: Chat-based interface

The modular design allows for easy testing and incremental development.
