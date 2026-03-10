# components/loader.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_file(uploaded_file):
    """
    Accept a Streamlit uploaded file (PDF or TXT)
    Returns list of chunked documents
    """

    # Save uploaded file temporarily to disk
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load based on file type
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        
        documents = loader.load()

    finally:
        os.unlink(tmp_path)  # clean up temp file

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Loaded {len(chunks)} chunks from {uploaded_file.name}")
    return chunks