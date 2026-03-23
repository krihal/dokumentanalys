"""Ingest PDF documents into ChromaDB for RAG-based decision support."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

load_dotenv()

console = Console()

CHUNK_SIZE = 800  # tokens (approx chars / 4)
CHUNK_OVERLAP = 200
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "decisions"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for last period, newline, or semicolon in the chunk
            for sep in ["\n\n", ".\n", ". ", "\n"]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    chunk = text[start:end]
                    break
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest(pdf_dir: str):
    """Ingest all PDFs from a directory into ChromaDB."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        console.print(f"[red]Directory not found: {pdf_dir}[/red]")
        sys.exit(1)

    pdfs = list(pdf_path.glob("**/*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDF files found in {pdf_dir}[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(pdfs)}[/bold] PDF files")

    # Load embedding model
    console.print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for pdf_file in track(pdfs, description="Processing PDFs"):
        # Check if already ingested
        existing = collection.get(where={"source": str(pdf_file)})
        if existing["ids"]:
            console.print(f"  [dim]Skipping (already ingested): {pdf_file.name}[/dim]")
            continue

        try:
            text = extract_text_from_pdf(pdf_file)
        except Exception as e:
            console.print(f"  [yellow]Skipping (cannot read): {pdf_file.name} — {e}[/yellow]")
            continue
        if not text.strip():
            console.print(f"  [yellow]Warning: no text extracted from {pdf_file.name}[/yellow]")
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        # Embed all chunks for this document
        embeddings = model.encode(chunks).tolist()

        # Store in ChromaDB
        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": str(pdf_file),
                "filename": pdf_file.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    console.print(f"\n[green]Done![/green] Ingested [bold]{total_chunks}[/bold] chunks from {len(pdfs)} PDFs")
    console.print(f"Database stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: uv run ingest.py <pdf_directory>")
        sys.exit(1)
    ingest(sys.argv[1])
