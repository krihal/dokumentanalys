"""Ingest PDF documents into ChromaDB for RAG-based decision support.

Improvements over basic chunking:
- Structural section detection (bakgrund, yrkande, bedömning, beslut, etc.)
- Metadata extraction (date, diarienummer, decision type)
- Full document text stored for expanded context retrieval
- BM25 index built alongside vector index
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
CHROMA_DIR = Path(__file__).parent / "chroma_db"
BM25_PATH = Path(__file__).parent / "bm25_index.pkl"
COLLECTION_NAME = "decisions"

# Swedish section headers commonly found in decision documents
SECTION_PATTERNS = [
    r"(?i)^#{0,3}\s*(bakgrund|ärendets?\s*bakgrund)",
    r"(?i)^#{0,3}\s*(yrkande[n]?|ansökan)",
    r"(?i)^#{0,3}\s*(bedömning|övervägande[n]?|skäl\s*för\s*beslutet?|motivering)",
    r"(?i)^#{0,3}\s*(beslut|avgörande|domslut|slutsats)",
    r"(?i)^#{0,3}\s*(sammanfattning|sammandrag)",
    r"(?i)^#{0,3}\s*(överklagande|besvär)",
    r"(?i)^#{0,3}\s*(bilaga|bilagor)",
    r"(?i)^#{0,3}\s*(parter|sökande|motpart|klagande)",
]

# Patterns for metadata extraction
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
DIARIENR_PATTERN = re.compile(r"(\d{4}-\d{3,6})")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_metadata(text: str, filename: str) -> dict:
    """Extract structured metadata from document text."""
    metadata = {}

    # Date
    dates = DATE_PATTERN.findall(text[:2000])
    if dates:
        metadata["date"] = dates[0]

    # Diarienummer — from filename or text
    dnr_match = DIARIENR_PATTERN.search(filename)
    if dnr_match:
        metadata["diarienummer"] = dnr_match.group(1)
    else:
        dnr_matches = DIARIENR_PATTERN.findall(text[:2000])
        if dnr_matches:
            metadata["diarienummer"] = dnr_matches[0]

    # Decision type heuristics
    text_lower = text[:5000].lower()
    if "avslår" in text_lower or "avslag" in text_lower:
        metadata["decision_type"] = "avslag"
    elif "bifaller" in text_lower or "bifall" in text_lower:
        metadata["decision_type"] = "bifall"
    elif "avvisar" in text_lower or "avvisning" in text_lower:
        metadata["decision_type"] = "avvisning"
    elif "återförvisar" in text_lower:
        metadata["decision_type"] = "återförvisning"

    return metadata


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Split text into named sections based on Swedish headings.

    Returns list of (section_name, section_text) tuples.
    """
    lines = text.split("\n")
    sections = []
    current_name = "inledning"
    current_lines = []

    for line in lines:
        matched = False
        for pattern in SECTION_PATTERNS:
            m = re.match(pattern, line.strip())
            if m:
                # Save previous section
                if current_lines:
                    sections.append((current_name, "\n".join(current_lines).strip()))
                current_name = m.group(1).lower().strip()
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
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


def structural_chunk(text: str) -> list[dict]:
    """Chunk document by structure, falling back to character-based chunking.

    Returns list of dicts with keys: text, section, chunk_index.
    """
    sections = detect_sections(text)
    result = []

    # If we found meaningful sections (more than just "inledning"), use them
    has_structure = len(sections) > 1 or (
        len(sections) == 1 and sections[0][0] != "inledning"
    )

    if has_structure:
        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            # Large sections still need to be chunked
            if len(section_text) > CHUNK_SIZE * 2:
                sub_chunks = chunk_text(section_text)
                for i, chunk in enumerate(sub_chunks):
                    result.append({
                        "text": chunk,
                        "section": section_name,
                        "chunk_index": len(result),
                    })
            else:
                result.append({
                    "text": section_text,
                    "section": section_name,
                    "chunk_index": len(result),
                })
    else:
        # Fallback to character-based chunking
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            result.append({
                "text": chunk,
                "section": "okänd",
                "chunk_index": i,
            })

    return result


def ingest(pdf_dir: str):
    """Ingest all PDFs from a directory into ChromaDB."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        console.print(f"[red]Mappen hittades inte: {pdf_dir}[/red]")
        sys.exit(1)

    pdfs = list(pdf_path.glob("**/*.pdf"))
    if not pdfs:
        console.print(f"[red]Inga PDF-filer hittades i {pdf_dir}[/red]")
        sys.exit(1)

    console.print(f"Hittade [bold]{len(pdfs)}[/bold] PDF-filer")

    # Load embedding model
    console.print("Laddar inbäddningsmodell...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # BM25 corpus — load existing or start fresh
    bm25_corpus: list[dict] = []  # list of {"tokens": [...], "doc_id": "...", "text": "..."}
    all_full_texts: dict[str, str] = {}  # filename -> full text

    total_chunks = 0
    skipped = 0

    for pdf_file in track(pdfs, description="Bearbetar PDF:er"):
        # Check if already ingested
        existing = collection.get(where={"source": str(pdf_file)})
        if existing["ids"]:
            skipped += 1
            continue

        try:
            text = extract_text_from_pdf(pdf_file)
        except Exception as e:
            console.print(f"  [yellow]Hoppar över (kan ej läsa): {pdf_file.name} — {e}[/yellow]")
            continue

        if not text.strip():
            console.print(f"  [yellow]Varning: ingen text från {pdf_file.name}[/yellow]")
            continue

        # Store full document text
        all_full_texts[pdf_file.name] = text

        # Extract metadata
        metadata = extract_metadata(text, pdf_file.name)

        # Structural chunking
        chunks = structural_chunk(text)
        if not chunks:
            continue

        # Embed all chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = model.encode(chunk_texts).tolist()

        # Store in ChromaDB
        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        metadatas = []
        for c in chunks:
            meta = {
                "source": str(pdf_file),
                "filename": pdf_file.name,
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "total_chunks": len(chunks),
                "full_text_chars": len(text),
            }
            meta.update(metadata)
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )

        # Add to BM25 corpus
        for c in chunks:
            tokens = c["text"].lower().split()
            bm25_corpus.append({
                "tokens": tokens,
                "doc_id": pdf_file.name,
                "text": c["text"],
                "section": c["section"],
            })

        total_chunks += len(chunks)

    # Save full texts for expanded context retrieval
    full_texts_path = Path(__file__).parent / "full_texts.json"
    if full_texts_path.exists():
        with open(full_texts_path) as f:
            existing_texts = json.load(f)
        existing_texts.update(all_full_texts)
        all_full_texts = existing_texts
    with open(full_texts_path, "w") as f:
        json.dump(all_full_texts, f, ensure_ascii=False)

    # Build and save BM25 index
    if bm25_corpus:
        # Load existing corpus if available
        if BM25_PATH.exists():
            with open(BM25_PATH, "rb") as f:
                existing_data = pickle.load(f)
            bm25_corpus = existing_data["corpus"] + bm25_corpus

        tokenized = [item["tokens"] for item in bm25_corpus]
        bm25 = BM25Okapi(tokenized)

        with open(BM25_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": bm25_corpus}, f)

    console.print(
        f"\n[green]Klart![/green] Matade in [bold]{total_chunks}[/bold] stycken "
        f"från {len(pdfs) - skipped} nya PDF:er ({skipped} redan inmatade)"
    )
    console.print(f"Databas: {CHROMA_DIR}")
    console.print(f"BM25-index: {BM25_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Användning: uv run ingest.py <pdf-mapp>")
        sys.exit(1)
    ingest(sys.argv[1])
