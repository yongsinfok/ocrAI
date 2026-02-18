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
from src.query_processor import QueryProcessor, QueryIntent
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
    st.session_state.ocr_engine = OCREngine(config=config)

if "query_processor" not in st.session_state:
    st.session_state.query_processor = QueryProcessor(st.session_state.model_manager)

if "export_manager" not in st.session_state:
    st.session_state.export_manager = ExportManager()

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def check_ocr_engine():
    """Check GLM-OCR status and display in sidebar."""
    st.sidebar.header("🔍 OCR Engine")

    ocr_status = st.session_state.ocr_engine.glm_ocr_status

    if not ocr_status["enabled"]:
        st.sidebar.info("GLM-OCR is disabled. Using Tesseract fallback.")
        return

    mode = ocr_status.get("mode", "local")
    mode_label = "Local vLLM" if mode == "local" else "Cloud MaaS"

    if ocr_status["connected"]:
        st.sidebar.success(f"GLM-OCR Connected ({mode_label})")
        st.sidebar.caption(f"Server: {config.glm_ocr.api_host}:{config.glm_ocr.api_port}")
    elif ocr_status["available"]:
        st.sidebar.warning(f"GLM-OCR Available ({mode_label})")
        st.sidebar.caption("Not connected - will use Tesseract fallback")
        if ocr_status.get("error"):
            with st.sidebar.expander("Error Details"):
                st.text(ocr_status["error"])
    else:
        st.sidebar.error("GLM-OCR Not Available")
        if ocr_status.get("error"):
            with st.sidebar.expander("Setup Instructions"):
                st.text(ocr_status["error"])

        st.sidebar.info("Using Tesseract fallback")

    # Show Tesseract status
    try:
        import pytesseract
        tesseract_version = pytesseract.get_tesseract_version()
        st.sidebar.caption(f"Tesseract: {tesseract_version}")
    except Exception:
        st.sidebar.caption("Tesseract: Not installed")


def check_models():
    """Check and download models if needed."""
    st.sidebar.header("🤖 LLM Models")

    # Only check Llama-3.1 for query processing
    models = ["llama_3_1"]
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
                        "{} JSON",
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
                                "{} JSON",
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


def extract_and_format_page(doc_path: Path, page_num: int, intent: QueryIntent) -> dict:
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
        check_ocr_engine()
        st.markdown("---")
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
