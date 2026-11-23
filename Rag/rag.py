
from pathlib import Path
from typing import List

import os
import certifi
from dotenv import load_dotenv

from pypdf import PdfReader
import docx2txt

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================
# Project Paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# Environment Variables
# =====================================

load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =====================================
# Embeddings Model
# =====================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# =====================================
# Chroma Vector Database
# =====================================

vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


# =====================================
# Read File Text
# =====================================

def read_file_text(file_path: str) -> str:

    path = Path(file_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    suffix = path.suffix.lower()

    # PDF
    if suffix == ".pdf":

        reader = PdfReader(str(path))
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    # DOCX
    if suffix == ".docx":

        return docx2txt.process(str(path))

    # TXT, MD, PY, CSV
    if suffix in [".txt", ".md", ".py", ".csv"]:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    raise ValueError(
        "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
    )


# =====================================
# Add Document to RAG
# =====================================

def add_document_to_rag(
    file_path: str,
    thread_id: str
):

    # Read document
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError(
            "No text could be extracted from this file."
        )

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    path = Path(file_path)

    # Convert chunks into Documents
    docs: List[Document] = [

        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": path.name
            }
        )

        for chunk in chunks
    ]

    # Store documents in Chroma
    vectorstore.add_documents(docs)

    return {
        "filename": path.name,
        "chunks": len(docs)
    }


# =====================================
# Retrieve from RAG
# =====================================

def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4
) -> str:

    # Similarity Search
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={
            "thread_id": thread_id
        }
    )

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    # Extract Results
    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "uploaded document"
        )

        results.append(
            f"[Source {i}: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(results)
