"""Turn one PDF, DOCX or HTML file into searchable chunks and embeddings.

Runs on the worker. Everything happens in memory: the file arrives as bytes,
nothing is written to disk and nothing about the content is logged. The
result goes back to the app, which encrypts it for the user (see vault.py).

Pipeline per document:
  1. Text extraction: PyMuPDF for PDF (optional OCR for scanned pages),
     word/document.xml for DOCX, visible text for HTML (e.g. exported e-mail)
  2. Cleaning: repeated headers/footers, page numbers, TOC dot leaders,
     line-break hyphenation
  3. Metadata: date, year, diarienummer, language, document type
  4. Structural section detection (Swedish + English headings)
  5. Sentence-aware chunking sized to the embedding model's token limit
  6. Embedding with a multilingual model

Usage (local check, prints counts only):
    uv run ingest.py <fil.pdf|fil.docx> [--ocr]

Environment:
    EMBED_MODEL   sentence-transformers model name (default: multilingual-e5-small)
"""

import argparse
import base64
import codecs
import hashlib
import io
import os
import re
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import fitz  # pymupdf
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

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

# --- Input ------------------------------------------------------------------

MAX_FILE_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
MIN_TEXT_CHARS = 100

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

_HYPHEN_LOWER = re.compile(r"([a-zåäöéü])-\n([a-zåäöéü])")
_HYPHEN_OTHER = re.compile(r"([A-ZÅÄÖ0-9])-\n([a-zåäöéü])")


# ===========================================================================
# Embedding (the worker also embeds queries with these)
# ===========================================================================


def load_embed_model() -> SentenceTransformer:
    """Load the embedding model with the sequence length capped at MAX_TOKENS."""
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


def embed_passages(model: SentenceTransformer, texts: list[str], progress: Progress | None = None) -> np.ndarray:
    """Embed in batches so progress can be reported (and cancellation checked) between them."""
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [PASSAGE_PREFIX + t for t in texts[i : i + EMBED_BATCH_SIZE]]
        out.append(model.encode(batch, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False))
        if progress:
            progress("embed", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))
    return np.vstack(out).astype(np.float32)


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode(
        [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0].astype(np.float32)


class DocumentError(Exception):
    """The file cannot be ingested. The message is safe to show and never quotes content."""


class Cancelled(Exception):
    """Raised by a progress callback to stop processing."""


# progress(stage, done, total): stage is "extract" (pages) or "embed" (chunks).
# It may raise Cancelled to abort.
Progress = Callable[[str, int, int], None]


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


def extract_pdf_pages(data: bytes, ocr: bool = False, progress: Progress | None = None) -> list[str]:
    """Raw text per page of an in-memory PDF. With ocr=True, pages without a text layer are OCR'd."""
    pages = []
    ocr_langs = _ocr_languages() if ocr else None
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise DocumentError("PDF:en är lösenordsskyddad.")
        for page in doc:
            if progress:
                progress("extract", page.number, doc.page_count)
            text = page.get_text()
            if ocr and len(text.strip()) < 20:
                try:
                    tp = page.get_textpage_ocr(language=ocr_langs, dpi=200, full=True)
                    text = page.get_text(textpage=tp)
                except Exception:  # tesseract missing, bad image, ... keep the empty page
                    pass
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


class _HTMLText(HTMLParser):
    """Visible text of an HTML document. Nothing is fetched or executed: the
    parser only tokenizes, and script/style/head content is dropped."""

    SKIP = {"script", "style", "head", "title", "noscript", "template", "svg", "object", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article",
             "blockquote", "pre", "hr", "ul", "ol", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_charset(data: bytes) -> str:
    """Declared charset (meta tag), else UTF-8 if it decodes, else Windows-1252."""
    m = re.search(rb"""charset\s*=\s*["']?([A-Za-z0-9_.:-]+)""", data[:4096], re.I)
    if m:
        name = m.group(1).decode("ascii")
        try:
            codecs.lookup(name)
            return name
        except LookupError:
            pass
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def extract_html_pages(data: bytes) -> list[str]:
    """Text of an in-memory HTML document (e.g. an exported e-mail) as one "page"."""
    parser = _HTMLText()
    parser.feed(data.decode(html_charset(data), errors="replace"))
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return [re.sub(r"\n\s*\n\s*", "\n\n", text).strip()]


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX_MAX_XML_BYTES = 200_000_000  # decompressed word/document.xml; guards against zip bombs


def extract_docx_pages(data: bytes) -> list[str]:
    """Text of an in-memory DOCX as a single "page", one line per paragraph.

    Reads word/document.xml directly. Documents with a DTD are refused, so
    entity expansion and external entities are never processed.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > DOCX_MAX_XML_BYTES:
                raise DocumentError("DOCX-filen är för stor.")
            xml = z.read(info)
    except (zipfile.BadZipFile, KeyError):
        raise DocumentError("Filen är inte en giltig DOCX.") from None
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise DocumentError("DOCX-filen innehåller otillåten XML.")

    paragraphs, parts = [], []
    for event, el in ElementTree.iterparse(io.BytesIO(xml), events=("end",)):
        tag = el.tag
        if tag == _W + "t":
            parts.append(el.text or "")
        elif tag == _W + "tab":
            parts.append("\t")
        elif tag in (_W + "br", _W + "cr"):
            parts.append("\n")
        elif tag == _W + "p":
            paragraphs.append("".join(parts))
            parts = []
            el.clear()
    return ["\n".join(paragraphs)]


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

    subject = re.search(r"^(?:Ämne|Subject):[ \t]*(\S.{3,150})$", text[:3000], re.M)  # exported e-mail
    if subject:
        metadata["title"] = subject.group(1).strip()
    for line in text.split("\n") if not subject else ():
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
# Whole document
# ===========================================================================


_HTML_START = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:<!--.*?-->\s*)*<(?:!doctype\s+html|html|head|body|meta)\b", re.I | re.S)


def detect_kind(data: bytes) -> str:
    """"pdf", "docx" or "html" from the file's content; the file name is not trusted.
    (Mail and web pages exported from other systems are often saved as .pdf.)"""
    if data[:5] == b"%PDF-":
        return "pdf"
    if _HTML_START.match(data[:4096]):
        return "html"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    raise DocumentError("Filen är varken PDF, DOCX eller HTML, oavsett vad den heter.")


def process_document(
    data: bytes, filename: str, model: SentenceTransformer, ocr: bool = False, progress: Progress | None = None
) -> dict:
    """Extract, clean, chunk and embed one file. Returns the document record
    the app encrypts: text, chunks, metadata and base64 float32 embeddings."""
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("Filen är för stor.")
    kind = detect_kind(data)
    if progress:
        progress("extract", 0, 0)
    try:
        if kind == "pdf":
            pages = extract_pdf_pages(data, ocr=ocr, progress=progress)
        elif kind == "docx":
            pages = extract_docx_pages(data)
        else:
            pages = extract_html_pages(data)
    except (DocumentError, Cancelled):
        raise
    except Exception:
        raise DocumentError("Filen kunde inte läsas.") from None
    text = clean_pages(pages)
    if len(text) < MIN_TEXT_CHARS:
        raise DocumentError("Ingen text kunde extraheras (skannad fil utan OCR?).")

    metadata = extract_metadata(text, filename, pages=len(pages) if kind == "pdf" else 0)
    chunks = fit_to_token_limit(structural_chunk(text), model.tokenizer, MAX_TOKENS)
    if not chunks:
        raise DocumentError("Ingen text kunde extraheras.")
    embeddings = embed_passages(model, [c["text"] for c in chunks], progress)
    return {
        "v": 1,
        "filename": filename,
        "kind": kind,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "metadata": metadata,
        "full_text": text,
        "chunks": [{"text": c["text"], "section": c["section"]} for c in chunks],
        "embed_model": EMBED_MODEL,
        "dim": int(embeddings.shape[1]),
        "embeddings": base64.b64encode(embeddings.tobytes()).decode(),
    }


def main():
    parser = argparse.ArgumentParser(description="Provkör inläsning av en fil (skriver bara ut antal).")
    parser.add_argument("file", help="PDF- eller DOCX-fil")
    parser.add_argument("--ocr", action="store_true", help="OCR:a sidor som saknar textlager (kräver tesseract)")
    args = parser.parse_args()
    path = Path(args.file).expanduser()
    t = time.time()
    doc = process_document(path.read_bytes(), path.name, load_embed_model(), ocr=args.ocr)
    print(f"{doc['kind']}: {len(doc['full_text'])} tecken, {len(doc['chunks'])} stycken, "
          f"{doc['dim']} dim, {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
