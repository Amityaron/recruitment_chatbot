"""
Embedding Module
----------------
Reads the job description PDF, splits it into chunks,
converts each chunk to a vector using OpenAI Embeddings,
and stores everything in a local Chroma DB.

Run this file ONCE before starting the application.
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))                 # .../embedding/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))  # project root
PDF_PATH    = os.path.join(PROJECT_DIR, "data", "Python Developer Job Description.pdf")
CHROMA_DIR  = os.path.join(PROJECT_DIR, "chroma_db")


def load_pdf(pdf_path: str):
    """
    Load the PDF and return a list of Document objects.
    Each page becomes one Document.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages.")
    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    chunk_size=500   -> each chunk is ~500 characters
    chunk_overlap=50 -> 50 characters overlap between chunks
                        so context is not lost at chunk boundaries
    """
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks):
    """
    Convert each chunk to a vector using OpenAI Embeddings
    and save to a local Chroma DB.
    """
    print("Creating embeddings and saving to Chroma DB...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Saved {len(chunks)} vectors to: {CHROMA_DIR}")
    return vector_store


def load_vector_store():
    """
    Load an existing Chroma DB from disk.
    Used by the Info Advisor at runtime (not during embedding).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vector_store


def run_embedding():
    """
    Full embedding pipeline - run this once before starting the app.
    """
    print("=" * 50)
    print("Starting Embedding Pipeline")
    print("=" * 50)

    documents    = load_pdf(PDF_PATH)
    chunks       = split_documents(documents)
    vector_store = create_vector_store(chunks)

    print("=" * 50)
    print("Embedding complete!")
    print(f"Chroma DB saved at: {CHROMA_DIR}")
    print("=" * 50)
    return vector_store


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_embedding()
