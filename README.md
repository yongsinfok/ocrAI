# Local OCR Application

A local document processing application using Tesseract OCR and Llama-3.1 for intelligent query understanding.

## Features

- Upload PDF or images
- Natural language query (English and Chinese)
- Extract tables and data
- Export to Excel, CSV, JSON
- Fully local - no API calls required

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### System Requirements

**Tesseract OCR** (required for text extraction from images):
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

**Llama-3.1** (optional - for advanced query understanding):
- Requires 8GB+ VRAM
- NVIDIA GPU with CUDA support
- OR can run in CPU-only mode (slower)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (see above)

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF or image
2. Ask questions about the document in natural language:
   - "从 Table 1.2.1 提取数据" (Extract data from Table 1.2.1)
   - "把第一列数据转成 JSON"
   - "提取所有表格并导出为 Excel"
3. Download results in your preferred format

## Hardware Requirements

- **Minimum**: 4GB RAM, CPU-only mode
- **Recommended**: 8GB VRAM (RTX 3070Ti or better), CUDA-enabled GPU

## Architecture

```
Upload → PDF Parser → Index Manager
                      ↓
         User Query → Query Processor (Llama-3.1)
                      ↓
         OCR Engine (Tesseract) → Export Manager
                      ↓
                    Download (Excel/CSV/JSON)
```

## Project Structure

- `app.py` - Streamlit UI
- `src/pdf_parser.py` - PDF parsing and structure extraction
- `src/ocr_engine.py` - Tesseract OCR processing
- `src/query_processor.py` - Natural language query understanding
- `src/index_manager.py` - Full-text search with Whoosh
- `src/export_manager.py` - Data export
