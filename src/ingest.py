import os
import re
import fitz  # PyMuPDF
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
PDF_DIR = "./data/pdfs"
DB_DIR = "./db/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"  # Excellent for retrieval
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def clean_text(text):
    """Basic cleaning for Vision IAS PDFs."""
    # Remove header/footer patterns (Customize these based on actual PDF)
    text = re.sub(r'www\.visionias\.in', '', text)
    text = re.sub(r'© Vision IAS', '', text)
    
    # Fix hyphenated words at end of lines (e.g., "bureau- cracy")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_process_pdfs():
    documents = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDFs. Processing...")

    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        doc = fitz.open(path)
        
        for page_num, page in enumerate(doc):
            # Extract text
            raw_text = page.get_text()
            cleaned_text = clean_text(raw_text)
            
            # Skip empty pages
            if len(cleaned_text) < 50:
                continue

            # Create Document with Metadata
            # page_num + 1 because fitz is 0-indexed
            doc_obj = Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_file,
                    "page": page_num + 1,
                    "month": pdf_file.replace(".pdf", "") # useful for filtering later
                }
            )
            documents.append(doc_obj)
            
    return documents

def create_vector_db(documents):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")

    # Check for GPU, fallback to CPU if not found
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading embedding model on {device.upper()}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}  # <--- This forces GPU usage
    )

    # Create/Persist Vector Store
    print("Creating Vector Store (this may take time)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Vector DB saved successfully.")

if __name__ == "__main__":
    docs = load_and_process_pdfs()
    create_vector_db(docs)