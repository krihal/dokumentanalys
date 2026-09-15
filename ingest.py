"""Ingest PDF documents into ChromaDB + BM25 for RAG-based decision support.

Pipeline per document:
  1. Text extraction with PyMuPDF (optional OCR fallback for scanned pages)
  2. Cleaning: repeated headers/footers, page numbers, TOC dot leaders,
     line-break hyphenation
  3. Metadata: date, year, diarienummer, language, document type
  4. Structural section detection (Swedish + English headings)
  5. Sentence-aware chunking sized to the embedding model's token limit
  6. Embedding with a multilingual model, stored in ChromaDB
  7. Full text stored in full_texts.json; BM25 index rebuilt from ChromaDB

Usage:
    uv run ingest.py <pdf-mapp> [--reset] [--ocr] [--limit N]

Environment:
    EMBED_MODEL   sentence-transformers model name (default: multilingual-e5-small)
    RAG_DATA_DIR  where chroma_db/, bm25_index.pkl and full_texts.json live
                  (default: this directory)
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import (
    track,
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

# --- Embedding -------------------------------------------------------------

EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
# e5 models are trained with these prefixes; other models get none.
_IS_E5 = "e5" in EMBED_MODEL.lower()
QUERY_PREFIX = "query: " if _IS_E5 else ""
PASSAGE_PREFIX = "passage: " if _IS_E5 else ""
MAX_TOKENS = 512  # hard limit of the embedding model
EMBED_BATCH_SIZE = 64

# --- Chunking --------------------------------------------------------------

# Measured ~3 chars/token for Swedish with the e5 tokenizer, so 1200 chars
# is roughly 400 tokens — comfortably under MAX_TOKENS with the prefix.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 80

# --- Storage ---------------------------------------------------------------

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", Path(__file__).parent))
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
FULL_TEXTS_PATH = DATA_DIR / "full_texts.json"
COLLECTION_NAME = "decisions"
CHECKPOINT_EVERY = 20  # docs between full_texts.json saves

# --- Section detection -----------------------------------------------------

# A heading is a whole short line: optional numbering, a known heading word,
# optional trailing number ("Bilaga 2") and punctuation. Nothing else.
_HEADING_WORDS = (
    r"bakgrund|ärendets\s+bakgrund|background|"
    r"yrkanden?|ansökan|"
    r"bedömning|överväganden?|skäl\s+för\s+beslutet?|motivering|assessment|"
    r"beslut|avgörande|domslut|decision|"
    r"slutsatser?|conclusions?|"
    r"sammanfattning|sammandrag|summary|"
    r"rekommendationer|recommendations?|"
    r"överklagande|besvär|"
    r"bilag(?:a|or)|appendix|appendices|"
    r"parter|sökande|motpart|klagande|"
    r"inledning|introduction|förord|preface|"
    r"innehåll(?:sförteckning)?|contents|"
    r"referenser|references|litteratur"
)
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(" + _HEADING_WORDS + r")(?:\s+\d+)?\s*[:.]?$",
    re.IGNORECASE,
)
MAX_HEADING_LEN = 60

# Canonical section names for variants
_SECTION_ALIASES = {
    "ärendets bakgrund": "bakgrund", "background": "bakgrund",
    "yrkande": "yrkanden", "överväganden": "bedömning", "övervägande": "bedömning",
    "skäl för beslutet": "bedömning", "skäl för beslut": "bedömning",
    "motivering": "bedömning", "assessment": "bedömning",
    "avgörande": "beslut", "domslut": "beslut", "decision": "beslut",
    "slutsatser": "slutsats", "conclusion": "slutsats", "conclusions": "slutsats",
    "sammandrag": "sammanfattning", "summary": "sammanfattning",
    "recommendation": "rekommendationer", "recommendations": "rekommendationer",
    "besvär": "överklagande",
    "bilagor": "bilaga", "appendix": "bilaga", "appendices": "bilaga",
    "introduction": "inledning", "preface": "förord",
    "innehållsförteckning": "innehåll", "contents": "innehåll",
    "references": "referenser", "litteratur": "referenser",
}

# --- Metadata patterns -----------------------------------------------------

DATE_PATTERN = re.compile(
    r"\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"
)
# VR diarienummer: 2015-06869, optionally prefixed "1.1.3-2015-6869"
DIARIENR_PATTERN = re.compile(r"(?<![\d.-])((?:19|20)\d{2}-\d{5})(?!\d)")
DIARIENR_PREFIXED = re.compile(r"\d(?:\.\d){1,3}-((?:19|20)\d{2})-(\d{4,5})(?!\d)")
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

_SV_STOP = set("och att det som för med till av på är den ett inte om".split())
_EN_STOP = set("the and of to in is for that with as on are by this".split())
_WORD_RE = re.compile(r"[a-zåäö]+")

_DOC_TYPES = [
    ("yttrande", r"yttrande|remissvar|remissyttrande"),
    ("beslut", r"\bbeslut"),
    ("föreskrift", r"föreskrift"),
    ("utvärdering", r"utvärdering|evaluation|granskning|review"),
    ("årsredovisning", r"årsredovisning|annual\s+report|årsbok|aarsbok"),
    ("budgetunderlag", r"budgetunderlag"),
    ("utlysning", r"utlysning|call\s+for|bidrag|grant"),
    ("rapport", r"rapport|report|översikt|kartläggning|barometer"),
    ("presentation", r"slide|presentation|webinar|workshop"),
]

# --- Text cleaning patterns ------------------------------------------------

PAGE_NUM_RE = re.compile(r"^(?:sida\s+|page\s+|s\.\s*)?\d{1,4}(?:\s*[(/]\s*\d{1,4}\s*\)?)?$", re.I)
DOT_LEADER_RE = re.compile(r"(?:\.\s?){5,}|_{5,}")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_HYPHEN_LOWER = re.compile(r"([a-zåäöéü])-\n([a-zåäöéü])")
_HYPHEN_OTHER = re.compile(r"([A-ZÅÄÖ0-9])-\n([a-zåäöéü])")


# ===========================================================================
# Shared helpers (used by worker.py / query.py too)
# ===========================================================================


def load_embed_model() -> SentenceTransformer:
    """Load the embedding model with the sequence length capped at MAX_TOKENS."""
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


def embed_passages(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()


def embed_query(model: SentenceTransformer, text: str) -> list[list[float]]:
    return model.encode(
        [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


def bm25_tokenize(text: str) -> list[str]:
    """Tokenizer shared by index build and query time — must be identical."""
    return TOKEN_RE.findall(text.lower())


# ===========================================================================
# Extraction and cleaning
# ===========================================================================


def _ocr_languages() -> str:
    try:
        tessdata = Path(fitz.get_tessdata())
    except Exception:
        return "eng"
    langs = [l for l in ("swe", "eng") if (tessdata / f"{l}.traineddata").exists()]
    return "+".join(langs) or "eng"


def extract_pages(pdf_path: Path, ocr: bool = False) -> list[str]:
    """Return raw text per page. With ocr=True, pages without a text layer are OCR'd."""
    pages = []
    ocr_langs = _ocr_languages() if ocr else None
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if ocr and len(text.strip()) < 20:
                try:
                    tp = page.get_textpage_ocr(language=ocr_langs, dpi=200, full=True)
                    text = page.get_text(textpage=tp)
                except Exception as e:  # tesseract missing, bad image, ...
                    console.print(f"  [yellow]OCR misslyckades ({pdf_path.name} s.{page.number + 1}): {e}[/yellow]")
            pages.append(text)
    return pages


def clean_pages(pages: list[str]) -> str:
    """Remove page furniture and join pages into one cleaned document text."""
    per_page = [[ln.strip() for ln in p.split("\n")] for p in pages]
    n = len(pages)

    # Lines that repeat on many pages are headers/footers.
    line_pages: Counter = Counter()
    for lines in per_page:
        for ln in set(lines):
            if ln:
                line_pages[ln] += 1
    threshold = max(3, int(0.2 * n))
    furniture = {
        ln for ln, c in line_pages.items() if n >= 4 and c >= threshold and len(ln) < 80
    }

    cleaned_pages = []
    for lines in per_page:
        kept = []
        edge = {i for i in range(len(lines)) if i < 2 or i >= len(lines) - 2}
        for i, ln in enumerate(lines):
            if not ln:
                kept.append("")
                continue
            if ln in furniture:
                continue
            if i in edge and PAGE_NUM_RE.match(ln):
                continue
            if DOT_LEADER_RE.search(ln):  # table of contents
                continue
            kept.append(ln)
        cleaned_pages.append("\n".join(kept))

    text = "\n\n".join(cleaned_pages)
    text = text.replace("\x00", "").replace("\f", "\n")
    # Soft hyphens: "närings\xadliv" -> "näringsliv", "närings\xad\nliv" -> "näringsliv"
    text = re.sub(r"\xad\s*\n?", "", text)
    text = _HYPHEN_LOWER.sub(r"\1\2", text)
    text = _HYPHEN_OTHER.sub(r"\1-\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path, ocr: bool = False) -> str:
    """Extract and clean all text from a PDF file."""
    return clean_pages(extract_pages(pdf_path, ocr=ocr))


# ===========================================================================
# Metadata
# ===========================================================================


def detect_language(text: str) -> str:
    words = _WORD_RE.findall(text[:20000].lower())
    sv = sum(w in _SV_STOP for w in words)
    en = sum(w in _EN_STOP for w in words)
    return "sv" if sv >= en else "en"


def extract_metadata(text: str, filename: str, pages: int = 0) -> dict:
    """Extract structured metadata from document text and filename."""
    head = text[:3000]
    metadata: dict = {}

    m = DATE_PATTERN.search(head)
    if m:
        metadata["date"] = m.group(0)

    # Diarienummer — prefer filename, then prefixed form in text, then plain
    m = DIARIENR_PATTERN.search(filename)
    if m:
        metadata["diarienummer"] = m.group(1)
    else:
        m = DIARIENR_PREFIXED.search(head)
        if m:
            metadata["diarienummer"] = f"{m.group(1)}-{int(m.group(2)):05d}"
        else:
            m = DIARIENR_PATTERN.search(head)
            if m:
                metadata["diarienummer"] = m.group(1)

    # Year: the document date if found, else the last plausible year in the
    # filename (e.g. "_VR_2014", "2019 Forskningsbarometer"; plan periods like
    # "2026-2037" must not win), else the most common plausible year in the head.
    max_year = time.localtime().tm_year + 1
    plausible = lambda ys: [y for y in ys if 1980 <= y <= max_year]
    year = None
    if "date" in metadata:
        year = int(metadata["date"][:4])
    if not year:
        years = plausible(int(y) for y in YEAR_PATTERN.findall(filename))
        year = years[-1] if years else None
    if not year:
        years = plausible(int(y) for y in YEAR_PATTERN.findall(head))
        if years:
            year = Counter(years).most_common(1)[0][0]
    if year:
        metadata["year"] = year

    metadata["language"] = detect_language(text)

    probe = (filename + " " + text[:600]).lower()
    for doc_type, pattern in _DOC_TYPES:
        if re.search(pattern, probe):
            metadata["doc_type"] = doc_type
            break
    else:
        metadata["doc_type"] = "okänd"

    if metadata["doc_type"] in ("beslut", "yttrande"):
        low = text[:5000].lower()
        for dtype, pattern in (
            ("avslag", r"\bavslå|\bavslag"),
            ("bifall", r"\bbifall"),
            ("avvisning", r"\bavvisa|\bavvisning"),
            ("återförvisning", r"\båterförvisa"),
        ):
            if re.search(pattern, low):
                metadata["decision_type"] = dtype
                break

    for line in text.split("\n"):
        line = line.strip()
        if 10 <= len(line) <= 120 and sum(c.isalpha() for c in line) > len(line) * 0.6:
            metadata["title"] = line
            break

    if pages:
        metadata["pages"] = pages
    return metadata


# ===========================================================================
# Sections and chunking
# ===========================================================================


def _heading_name(line: str) -> str | None:
    if not line or len(line) > MAX_HEADING_LEN or not (line[0].isupper() or line[0].isdigit()):
        return None
    m = HEADING_RE.match(line)
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group(1).lower().strip())
    return _SECTION_ALIASES.get(name, name)


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Split text into named sections based on Swedish/English headings.

    Returns list of (section_name, section_text) tuples.
    """
    sections: list[tuple[str, str]] = []
    current_name = "inledning"
    current_lines: list[str] = []

    for line in text.split("\n"):
        name = _heading_name(line.strip())
        if name:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = name
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))
    return [(n, t) for n, t in sections if t]


# Sentence boundary: terminal punctuation followed by whitespace and an
# uppercase/digit/quote start — or a paragraph break.
_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+(?=[A-ZÅÄÖ0-9\"“(\[])|\n{2,}")


def _hard_split(unit: str, chunk_size: int) -> list[str]:
    """Split an oversized unit (table, list without punctuation) on lines/spaces."""
    pieces, cur = [], ""
    for part in re.split(r"(?<=\n)|(?<= )", unit):
        if len(cur) + len(part) > chunk_size and cur:
            pieces.append(cur.strip())
            cur = ""
        cur += part
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks that start and end on sentence boundaries.

    Chunks are packed greedily from sentence/paragraph units up to chunk_size
    characters; the overlap is made of whole trailing units.
    """
    units: list[str] = []
    for u in _SENT_SPLIT.split(text):
        u = u.strip()
        if not u:
            continue
        if len(u) > chunk_size:
            units.extend(_hard_split(u, chunk_size))
        else:
            units.append(u)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > chunk_size:
            chunks.append(" ".join(cur))
            keep, keep_len = [], 0
            for x in reversed(cur):
                if keep_len + len(x) + 1 > overlap:
                    break
                keep.insert(0, x)
                keep_len += len(x) + 1
            cur, cur_len = keep, keep_len
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))

    # Collapse the PDF's hard line breaks inside a chunk.
    return [re.sub(r"[ \t]*\n[ \t]*", " ", c).strip() for c in chunks if c.strip()]


def structural_chunk(text: str) -> list[dict]:
    """Chunk document by detected sections, then by sentences within each.

    Returns list of dicts with keys: text, section, chunk_index.
    """
    sections = detect_sections(text)
    if not sections:
        return []

    result: list[dict] = []
    pending = ""  # crumbs with no previous chunk to merge into
    for section_name, section_text in sections:
        for chunk in chunk_text(section_text):
            if pending:
                chunk, pending = pending + " " + chunk, ""
            if len(chunk) < MIN_CHUNK_CHARS:
                if result:
                    result[-1]["text"] += " " + chunk  # merge crumbs into previous chunk
                else:
                    pending = chunk
                continue
            result.append({"text": chunk, "section": section_name, "chunk_index": len(result)})
    if pending and not result:
        result.append({"text": pending, "section": sections[0][0], "chunk_index": 0})
    return result


def fit_to_token_limit(chunks: list[dict], tokenizer, limit: int) -> list[dict]:
    """Guarantee no chunk exceeds the model's token limit; split the rare offenders."""
    budget = limit - len(tokenizer(PASSAGE_PREFIX)["input_ids"]) - 2
    out: list[dict] = []
    for c in chunks:
        queue = [c["text"]]
        while queue:
            t = queue.pop(0)
            n = len(tokenizer(t, add_special_tokens=False)["input_ids"])
            if n <= budget or len(t) < 200:
                out.append({"text": t, "section": c["section"], "chunk_index": len(out)})
            else:
                halves = chunk_text(t, chunk_size=max(200, len(t) // 2), overlap=0)
                queue = halves + queue if len(halves) > 1 else queue
                if len(halves) <= 1:  # cannot split further; keep (will be truncated)
                    out.append({"text": t, "section": c["section"], "chunk_index": len(out)})
    return out


# ===========================================================================
# Storage helpers
# ===========================================================================


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_id(source: str, index: int) -> str:
    return f"{hashlib.sha1(source.encode()).hexdigest()[:16]}_{index}"


def _iter_collection(collection, batch: int = 5000):
    offset = 0
    while True:
        res = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        if not res["ids"]:
            break
        yield from zip(res["documents"], res["metadatas"])
        offset += len(res["ids"])
        if len(res["ids"]) < batch:
            break


def build_bm25(collection) -> int:
    """Rebuild the BM25 index from everything in ChromaDB. Returns corpus size."""
    corpus = [
        {
            "doc_id": meta["filename"],
            "text": doc,
            "section": meta.get("section", ""),
            "source": meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", 0),
        }
        for doc, meta in _iter_collection(collection)
    ]
    if not corpus:
        return 0
    bm25 = BM25Okapi([bm25_tokenize(item["text"]) for item in corpus])
    tmp = BM25_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "tokenizer": "ingest.bm25_tokenize"}, f)
    tmp.replace(BM25_PATH)
    return len(corpus)


def _save_full_texts(full_texts: dict[str, str]) -> None:
    tmp = FULL_TEXTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(full_texts, f, ensure_ascii=False)
    tmp.replace(FULL_TEXTS_PATH)


def _load_full_texts() -> dict[str, str]:
    if FULL_TEXTS_PATH.exists():
        with open(FULL_TEXTS_PATH) as f:
            return json.load(f)
    return {}


# ===========================================================================
# Main ingest
# ===========================================================================


def ingest(pdf_dir: str, reset: bool = False, ocr: bool = False, limit: int | None = None):
    """Ingest all PDFs from a directory into ChromaDB, full_texts.json and BM25."""
    t_start = time.time()
    pdf_path = Path(pdf_dir).expanduser()
    if not pdf_path.exists():
        console.print(f"[red]Mappen hittades inte: {pdf_dir}[/red]")
        sys.exit(1)

    pdfs = sorted(p for p in pdf_path.glob("**/*") if p.suffix.lower() == ".pdf")
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        console.print(f"[red]Inga PDF-filer hittades i {pdf_dir}[/red]")
        sys.exit(1)
    console.print(f"Hittade [bold]{len(pdfs)}[/bold] PDF-filer i {pdf_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        console.print("[yellow]--reset: tar bort befintligt index[/yellow]")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        BM25_PATH.unlink(missing_ok=True)
        FULL_TEXTS_PATH.unlink(missing_ok=True)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )
    existing_model = (collection.metadata or {}).get("embed_model")
    if collection.count() and existing_model != EMBED_MODEL:
        console.print(
            f"[red]Befintligt index är byggt med '{existing_model or 'okänd modell'}', "
            f"men EMBED_MODEL är '{EMBED_MODEL}'. Kör med --reset för att bygga om.[/red]"
        )
        sys.exit(1)

    console.print(f"Laddar inbäddningsmodell [bold]{EMBED_MODEL}[/bold]...")
    model = load_embed_model()
    tokenizer = model.tokenizer
    max_batch = client.get_max_batch_size()

    # What is already in the index?
    existing_sources: dict[str, str] = {}  # source path -> filename
    existing_hashes: dict[str, str] = {}  # sha256 -> filename
    for _, meta in _iter_collection(collection):
        existing_sources[meta["source"]] = meta["filename"]
        if "sha256" in meta:
            existing_hashes[meta["sha256"]] = meta["filename"]

    full_texts = _load_full_texts()

    # Repair: docs in ChromaDB whose full text was lost (e.g. interrupted run)
    missing = [Path(s) for s, name in existing_sources.items() if name not in full_texts and Path(s).exists()]
    if missing:
        console.print(f"Återskapar fulltext för {len(missing)} dokument...")
        for p in missing:
            full_texts[p.name] = extract_text_from_pdf(p, ocr=ocr)
        _save_full_texts(full_texts)

    total_chunks = skipped = duplicates = 0
    no_text: list[str] = []
    failed: list[str] = []
    since_checkpoint = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Bearbetar", total=len(pdfs))
        for pdf_file in pdfs:
            progress.update(task, advance=1, description=pdf_file.name[:50])
            source = str(pdf_file)
            if source in existing_sources:
                skipped += 1
                continue

            sha = _file_sha256(pdf_file)
            if sha in existing_hashes:
                console.print(f"  [dim]Dubblett av {existing_hashes[sha]}: {pdf_file.name}[/dim]")
                duplicates += 1
                continue

            try:
                pages = extract_pages(pdf_file, ocr=ocr)
                text = clean_pages(pages)
            except Exception as e:
                console.print(f"  [yellow]Hoppar över (kan ej läsa): {pdf_file.name} — {e}[/yellow]")
                failed.append(pdf_file.name)
                continue

            if len(text) < 100:
                no_text.append(pdf_file.name)
                continue

            metadata = extract_metadata(text, pdf_file.name, pages=len(pages))
            chunks = fit_to_token_limit(structural_chunk(text), tokenizer, MAX_TOKENS)
            if not chunks:
                no_text.append(pdf_file.name)
                continue

            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_passages(model, chunk_texts)

            ids = [chunk_id(source, i) for i in range(len(chunks))]
            metadatas = []
            for c in chunks:
                meta = {
                    "source": source,
                    "filename": pdf_file.name,
                    "sha256": sha,
                    "section": c["section"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": len(chunks),
                    "full_text_chars": len(text),
                }
                meta.update(metadata)
                metadatas.append(meta)

            for i in range(0, len(ids), max_batch):
                collection.add(
                    ids=ids[i : i + max_batch],
                    embeddings=embeddings[i : i + max_batch],
                    documents=chunk_texts[i : i + max_batch],
                    metadatas=metadatas[i : i + max_batch],
                )
            existing_sources[source] = pdf_file.name
            existing_hashes[sha] = pdf_file.name
            full_texts[pdf_file.name] = text
            total_chunks += len(chunks)

            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                _save_full_texts(full_texts)
                since_checkpoint = 0

    _save_full_texts(full_texts)

    console.print("Bygger BM25-index från databasen...")
    bm25_size = build_bm25(collection)

    elapsed = time.time() - t_start
    new_docs = len(pdfs) - skipped - duplicates - len(no_text) - len(failed)
    console.print(
        f"\n[green]Klart på {elapsed / 60:.1f} min.[/green] Matade in [bold]{total_chunks}[/bold] stycken "
        f"från {new_docs} nya PDF:er ({skipped} redan inmatade, {duplicates} dubbletter)."
    )
    if no_text:
        console.print(
            f"[yellow]{len(no_text)} PDF:er utan textlager (skannade?) hoppades över"
            f"{'' if ocr else ' — prova --ocr'}:[/yellow]"
        )
        for name in no_text:
            console.print(f"  - {name}")
    if failed:
        console.print(f"[yellow]{len(failed)} PDF:er kunde inte läsas:[/yellow]")
        for name in failed:
            console.print(f"  - {name}")
    console.print(f"Databas: {CHROMA_DIR} ({collection.count()} stycken, {len(existing_sources)} dokument)")
    console.print(f"BM25-index: {BM25_PATH} ({bm25_size} stycken)")
    console.print(f"Fulltexter: {FULL_TEXTS_PATH} ({len(full_texts)} dokument)")


def refresh_metadata() -> int:
    """Recompute document metadata from stored full texts and update ChromaDB.

    Lets metadata heuristics improve without re-extracting or re-embedding.
    Returns the number of documents updated.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    full_texts = _load_full_texts()
    docs = {}
    for _, meta in _iter_collection(collection):
        docs.setdefault(meta["source"], meta)
    updated = 0
    for source, meta in track(list(docs.items()), description="Uppdaterar metadata"):
        text = full_texts.get(meta["filename"])
        if text is None:
            continue
        new = extract_metadata(text, meta["filename"], pages=meta.get("pages", 0))
        keys = ("date", "diarienummer", "year", "language", "doc_type", "decision_type", "title", "pages")
        if all(meta.get(k) == new.get(k) for k in keys):
            continue
        total = int(meta["total_chunks"])
        ids = [chunk_id(source, i) for i in range(total)]
        res = collection.get(ids=ids, include=["metadatas"])
        metadatas = []
        for m in res["metadatas"]:
            m = {k: v for k, v in m.items() if k not in keys}
            m.update(new)
            metadatas.append(m)
        collection.update(ids=res["ids"], metadatas=metadatas)
        updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="Mata in PDF:er i sökindexet.")
    parser.add_argument("pdf_dir", nargs="?", help="Mapp med PDF-filer (söks rekursivt)")
    parser.add_argument("--reset", action="store_true", help="Ta bort befintligt index först")
    parser.add_argument("--ocr", action="store_true", help="OCR:a sidor som saknar textlager (kräver tesseract)")
    parser.add_argument("--limit", type=int, default=None, help="Bearbeta bara de N första filerna (test)")
    parser.add_argument("--rebuild-bm25", action="store_true", help="Bygg bara om BM25-indexet från databasen")
    parser.add_argument("--refresh-metadata", action="store_true", help="Räkna om metadata från lagrade fulltexter utan ny inbäddning")
    args = parser.parse_args()
    if not args.pdf_dir and not (args.rebuild_bm25 or args.refresh_metadata):
        parser.error("ange en PDF-mapp, --rebuild-bm25 eller --refresh-metadata")
    if args.refresh_metadata:
        n = refresh_metadata()
        console.print(f"[green]Klart.[/green] Metadata uppdaterad för {n} dokument")
        return
    if args.rebuild_bm25:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        console.print("Bygger BM25-index från databasen...")
        n = build_bm25(collection)
        console.print(f"[green]Klart.[/green] BM25-index: {BM25_PATH} ({n} stycken)")
        return
    ingest(args.pdf_dir, reset=args.reset, ocr=args.ocr, limit=args.limit)


if __name__ == "__main__":
    main()
