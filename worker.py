"""Worker — connects TO the UI server via WebSocket and processes jobs.

Run on the machine with GPU/resources:
    uv run python worker.py
    uv run python worker.py wss://ui-server.example.com/ws/worker

Retrieval pipeline:
    1. Hybrid search: vector (ChromaDB) + keyword (BM25)
    2. Re-ranking: cross-encoder scores merged results
    3. Expanded context: retrieve full document text for top hits

Protocol (JSON over WebSocket):
    UI -> Worker:  {"type": "ask",     "id": "...", "question": "..."}
    UI -> Worker:  {"type": "ask-pdf", "id": "...", "pdf_base64": "..."}
    Worker -> UI:  {"type": "status",  "chunks": N, "model": "..."}
    Worker -> UI:  {"type": "chunk",   "id": "...", "text": "...", "num_sources": N}
    Worker -> UI:  {"type": "done",    "id": "..."}
    Worker -> UI:  {"type": "error",   "id": "...", "message": "..."}
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import pickle
import sys
import tempfile

# Disable telemetry and analytics before importing any third-party libraries
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # ChromaDB
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

import chromadb
import fitz
import ollama
import websockets

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import BM25_PATH, CHROMA_DIR, COLLECTION_NAME, chunk_text

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger(__name__)

DEFAULT_URL = os.getenv("UI_URL", "ws://localhost:7777/ws/worker")
DEFAULT_MODEL = os.getenv("MODEL", "gemma3")
MODEL = DEFAULT_MODEL  # overridden by --model argument

# Retrieval parameters
VECTOR_TOP_K = 30  # candidates from vector search
BM25_TOP_K = 30  # candidates from BM25 search
RERANK_TOP_K = 10  # final results after re-ranking
MAX_CONTEXT_CHARS = 30000  # max chars sent to LLM (fits ~8k tokens)

SYSTEM_PROMPT = """Du är en beslutstödsassistent. Du hjälper till att fatta nya beslut baserat på tidigare beslut som tillhandahålls som kontext.

Du kan hantera olika typer av frågor:

**Beslutsstöd** (t.ex. "Hur ska vi hantera detta ärende?"):
1. Analysera de relevanta tidigare besluten som tillhandahålls som kontext
2. Identifiera mönster, prejudikat och principer från dessa beslut
3. Tillämpa dem på den nya situationen
4. Ge en tydlig rekommendation med motivering
5. Hänvisa till vilka tidigare beslut som stödjer din rekommendation (referera med filnamn)

**Analytiska frågor** (t.ex. "Hur många dokument handlar om X?", "Vilka dokument nämner Y?"):
1. Granska den tillhandahållna kontexten noggrant
2. Räkna eller lista dokument som matchar frågan
3. Var exakt — ange filnamn för varje träff
4. Om du fått en textsökning, rapportera exakt antal träffar och vilka dokument som matchade

**Visualiseringar:**
När det är relevant att visa data grafiskt (t.ex. fördelningar, jämförelser, antal per kategori), inkludera ett diagram med detta exakta format:

```chart
{
  "title": "Diagramtitel",
  "type": "bar",
  "data": [
    {"label": "Kategori A", "value": 10},
    {"label": "Kategori B", "value": 20}
  ],
  "x_label": "X-axel",
  "y_label": "Y-axel"
}
```

Tillgängliga diagramtyper: bar, pie, line, scatter. Använd diagram framför allt vid analytiska frågor om antal, fördelningar eller jämförelser. Inkludera alltid en textuell sammanfattning utöver diagrammet.

Om kontexten inte innehåller relevanta resultat, säg det tydligt istället för att gissa. Svara alltid på svenska."""

PDF_SYSTEM_PROMPT = """Du är en beslutstödsassistent. Du analyserar ett nytt ärende (en uppladdad PDF) och jämför det med tidigare beslut som tillhandahålls som kontext.

När du svarar:
1. Sammanfatta det uppladdade ärendet kort
2. Identifiera relevanta tidigare beslut från kontexten
3. Jämför det nya ärendet med tidigare prejudikat
4. Ge en tydlig rekommendation med motivering
5. Hänvisa till vilka tidigare beslut som stödjer din rekommendation (referera med filnamn)
6. Om det uppladdade ärendet inte liknar något tidigare beslut i kontexten, säg det tydligt istället för att gissa.

Om kontexten inte innehåller relevanta prejudikat, säg det tydligt istället för att gissa. Svara alltid på svenska."""

# Load models at startup
log.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

log.info("Loading re-ranking model...")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

log.info("Loading BM25 index...")
bm25_data = None
if BM25_PATH.exists():
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    log.info("  BM25: %d documents", len(bm25_data['corpus']))
else:
    log.warning("  BM25 index missing — run ingest.py first for hybrid search.")

# Load full texts for expanded context
FULL_TEXTS_PATH = Path(__file__).parent / "full_texts.json"
full_texts: dict[str, str] = {}
if FULL_TEXTS_PATH.exists():
    with open(FULL_TEXTS_PATH) as f:
        full_texts = json.load(f)
    log.info("  Full texts: %d documents", len(full_texts))

log.info("All models loaded.")


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def vector_search(query: str, top_k: int = VECTOR_TOP_K) -> list[dict]:
    """Search using vector similarity."""
    collection = get_collection()
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine similarity
                "source": "vector",
            }
        )
    return hits


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Search using BM25 keyword matching."""
    if not bm25_data:
        return []

    bm25 = bm25_data["bm25"]
    corpus = bm25_data["corpus"]

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    hits = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        item = corpus[idx]
        hits.append(
            {
                "text": item["text"],
                "metadata": {
                    "filename": item["doc_id"],
                    "section": item.get("section", ""),
                },
                "score": float(scores[idx]),
                "source": "bm25",
            }
        )
    return hits


def merge_and_deduplicate(vector_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
    """Merge results from both sources, deduplicate by text content."""
    seen_texts = set()
    merged = []

    for hit in vector_hits + bm25_hits:
        # Use first 200 chars as dedup key
        key = hit["text"][:200]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(hit)

    return merged


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """Re-rank candidates using cross-encoder."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = rerank_model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]


def expand_context(hits: list[dict]) -> str:
    """Build context with expanded full documents for top hits.

    Strategy: for the top 3 source documents, include as much of the full text
    as possible. For the rest, include only the matched chunks.
    """
    # Group by source filename
    by_source: dict[str, list[dict]] = {}
    source_order = []
    for hit in hits:
        filename = hit["metadata"].get("filename", "okänd")
        if filename not in by_source:
            by_source[filename] = []
            source_order.append(filename)
        by_source[filename].append(hit)

    context_parts = []
    total_chars = 0

    for i, filename in enumerate(source_order):
        source_hits = by_source[filename]
        best_score = max(h.get("rerank_score", h.get("score", 0)) for h in source_hits)

        # For top 3 sources: try to include full document
        if i < 3 and filename in full_texts:
            doc_text = full_texts[filename]
            # Budget: leave room for other sources
            budget = min(
                len(doc_text),
                (MAX_CONTEXT_CHARS - total_chars) // max(1, len(source_order) - i),
            )
            if budget > 500:
                excerpt = doc_text[:budget]
                if budget < len(doc_text):
                    excerpt += "\n\n[...dokumentet fortsätter...]"
                context_parts.append(
                    f"### Källa: {filename} (relevans: {best_score:.2f})\n\n{excerpt}"
                )
                total_chars += len(excerpt)
                continue

        # Fallback: include matched chunks
        for hit in source_hits:
            section = hit["metadata"].get("section", "")
            section_label = f" [{section}]" if section and section != "okänd" else ""
            score = hit.get("rerank_score", hit.get("score", 0))

            entry = f"### Källa: {filename}{section_label} (relevans: {score:.2f})\n\n{hit['text']}"
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        if total_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n\n---\n\n".join(context_parts)


def text_search(terms: list[str], case_sensitive: bool = False) -> list[dict]:
    """Search for exact substrings across all full document texts."""
    results = []
    for filename, text in full_texts.items():
        search_text = text if case_sensitive else text.lower()
        matches = {}
        for term in terms:
            search_term = term if case_sensitive else term.lower()
            count = search_text.count(search_term)
            if count > 0:
                matches[term] = count
        if matches:
            # Extract a short snippet around the first match
            first_term = list(matches.keys())[0]
            st = first_term if case_sensitive else first_term.lower()
            idx = search_text.find(st)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(first_term) + 200)
            snippet = text[start:end].replace("\n", " ").strip()
            results.append({
                "filename": filename,
                "matches": matches,
                "total_matches": sum(matches.values()),
                "snippet": f"...{snippet}...",
            })
    results.sort(key=lambda x: x["total_matches"], reverse=True)
    return results


def retrieve_broad(query: str, top_k: int = 50) -> tuple[str, int]:
    """Broader retrieval for aggregate/analytical questions."""
    vector_hits = vector_search(query, top_k=min(top_k, 100))
    bm25_hits = bm25_search(query, top_k=min(top_k, 100))
    candidates = merge_and_deduplicate(vector_hits, bm25_hits)

    if not candidates:
        return "", 0

    ranked = rerank(query, candidates, top_k=top_k)
    if not ranked:
        return "", 0

    # For analytical questions, list all matching documents with brief excerpts
    seen_sources = set()
    context_parts = []
    total_chars = 0

    for hit in ranked:
        filename = hit["metadata"].get("filename", "okänd")
        if filename in seen_sources:
            continue
        seen_sources.add(filename)
        score = hit.get("rerank_score", hit.get("score", 0))
        excerpt = hit["text"][:300].replace("\n", " ")
        entry = f"### {filename} (relevans: {score:.2f})\n{excerpt}..."
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context = "\n\n".join(context_parts)
    return context, len(seen_sources)


import re as _re

# Patterns that indicate analytical/aggregate questions
_ANALYTICAL_PATTERNS = [
    r"hur många",
    r"hur\s+stor\s+andel",
    r"vilka dokument",
    r"vilka beslut",
    r"lista alla",
    r"räkna",
    r"antal",
    r"finns det.*som handlar om",
    r"finns det.*som nämner",
    r"finns det.*som innehåller",
    r"som handlar om",
    r"som nämner",
    r"som innehåller",
    r"sök efter",
    r"hitta alla",
    r"innehåller.*strängen",
    r"innehåller.*ordet",
    r"innehåller.*texten",
]


def _is_analytical(question: str) -> bool:
    q = question.lower()
    return any(_re.search(p, q) for p in _ANALYTICAL_PATTERNS)


def _extract_search_terms(question: str) -> list[str]:
    """Extract quoted strings or key nouns as text search terms."""
    # First try quoted strings: "katter" or 'katter'
    quoted = _re.findall(r'["\u201c\u201d\'](.*?)["\u201c\u201d\']', question)
    if quoted:
        return quoted
    return []


def retrieve(query: str) -> tuple[str, int]:
    """Full retrieval pipeline: hybrid search -> rerank -> expand context."""
    # Step 1: Hybrid search
    vector_hits = vector_search(query)
    bm25_hits = bm25_search(query)
    candidates = merge_and_deduplicate(vector_hits, bm25_hits)

    if not candidates:
        return "", 0

    # Step 2: Re-rank
    ranked = rerank(query, candidates)

    if not ranked:
        return "", 0

    # Count unique sources
    seen_sources = {h["metadata"].get("filename", "") for h in ranked}

    # Step 3: Expand context
    context = expand_context(ranked)

    return context, len(seen_sources)


def process_question(question: str):
    """Generator yielding (num_sources, token) tuples."""
    analytical = _is_analytical(question)
    search_terms = _extract_search_terms(question)

    # Build text search results if we have explicit search terms
    text_search_context = ""
    if search_terms:
        ts_results = text_search(search_terms)
        if ts_results:
            parts = [f"**Textsökning för {search_terms}:** {len(ts_results)} dokument matchade.\n"]
            for r in ts_results[:50]:
                match_info = ", ".join(f'"{t}": {c} träffar' for t, c in r["matches"].items())
                parts.append(f"- **{r['filename']}** ({match_info})\n  Utdrag: {r['snippet']}")
            text_search_context = "\n".join(parts)
        else:
            text_search_context = f"**Textsökning för {search_terms}:** Inga dokument matchade."

    # Retrieve with broader scope for analytical questions
    if analytical:
        context, num_sources = retrieve_broad(question)
    else:
        context, num_sources = retrieve(question)

    if not context and not text_search_context:
        yield num_sources, "Inga relevanta beslut hittades i databasen."
        return

    # Build prompt
    context_sections = []
    if text_search_context:
        context_sections.append(f"## Textsökningsresultat\n\n{text_search_context}")
    if context:
        context_sections.append(f"## Tidigare beslut (Kontext)\n\n{context}")

    combined_context = "\n\n---\n\n".join(context_sections)

    if analytical:
        instruction = "Baserat på informationen ovan, svara på frågan. Var exakt med antal och filnamn."
    else:
        instruction = "Baserat på tidigare beslut ovan, ge din analys och rekommendation."

    user_prompt = f"""{combined_context}

## Fråga

{question}

## Ditt svar

{instruction}"""

    log.info("[%s] %s", "analytical" if analytical else "standard", question)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in response:
        yield num_sources, chunk["message"]["content"]


def process_pdf(pdf_base64: str):
    """Generator yielding (num_sources, token) tuples."""
    pdf_bytes = base64.b64decode(pdf_base64)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        doc.close()
    except Exception as ex:
        yield 0, f"Kunde inte läsa PDF:en: {ex}"
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pdf_text.strip():
        yield 0, "Ingen text kunde extraheras från PDF:en. Filen kan vara skannad utan OCR."
        return

    # Use first chunks as search query
    chunks = chunk_text(pdf_text)
    search_text = " ".join(chunks[:3])[:2000]
    context, num_sources = retrieve(search_text)

    if not context:
        yield 0, "Inga relevanta tidigare beslut hittades i databasen."
        return

    # Include as much of the uploaded PDF as possible
    pdf_excerpt = pdf_text[:8000]
    if len(pdf_text) > 8000:
        pdf_excerpt += "\n\n[...dokumentet fortsätter...]"

    user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Nytt ärende (uppladdad PDF)

{pdf_excerpt}

## Din rekommendation

Analysera det nya ärendet ovan och ge ett utlåtande baserat på tidigare beslut."""

    log.debug("PDF prompt: %s", user_prompt[:200])

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": PDF_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in response:
        yield num_sources, chunk["message"]["content"]


async def handle_job(ws, msg: dict):
    """Process a single job and stream results back."""
    job_id = msg.get("id", "unknown")
    job_type = msg.get("type")

    try:
        if job_type == "ask":
            gen = process_question(msg["question"])
        elif job_type == "ask-pdf":
            gen = process_pdf(msg["pdf_base64"])
        else:
            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "id": job_id,
                        "message": f"Unknown job type: {job_type}",
                    }
                )
            )
            return

        for num_sources, token in gen:
            await ws.send(
                json.dumps(
                    {
                        "type": "chunk",
                        "id": job_id,
                        "text": token,
                        "num_sources": num_sources,
                    }
                )
            )

        await ws.send(json.dumps({"type": "done", "id": job_id}))

    except Exception as ex:
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "id": job_id,
                    "message": str(ex),
                }
            )
        )


async def send_status(ws):
    """Send current status to UI."""
    collection = get_collection()
    await ws.send(
        json.dumps(
            {
                "type": "status",
                "chunks": collection.count(),
                "model": MODEL,
            }
        )
    )


async def connect(url: str):
    """Connect to UI and process jobs forever."""
    while True:
        try:
            log.info("Connecting to %s...", url)
            async with websockets.connect(url, max_size=100_000_000) as ws:
                log.info("Connected to %s", url)
                await send_status(ws)

                async for raw in ws:
                    msg = json.loads(raw)
                    await handle_job(ws, msg)

        except (ConnectionRefusedError, OSError) as ex:
            log.warning("Connection failed: %s. Retrying in 5s...", ex)
        except websockets.ConnectionClosed:
            log.warning("Connection closed. Reconnecting in 2s...")
            await asyncio.sleep(2)
            continue

        await asyncio.sleep(5)


def download_models():
    """Download embedding and re-ranking models for offline use."""
    # Temporarily allow network access
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    log.info("Downloading embedding model: all-MiniLM-L6-v2...")
    SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Downloading re-ranking model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    log.info("All models downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for decision support RAG system")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--download", action="store_true", help="Download models for offline use and exit")
    parser.add_argument("--log-file", metavar="PATH", help="Log to file in addition to stderr")
    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(file_handler)

    if args.download:
        download_models()
        sys.exit(0)

    MODEL = args.model
    log.info("Using model: %s", MODEL)
    asyncio.run(connect(args.url))
