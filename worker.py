"""Worker — connects TO the UI server via WebSocket and processes jobs.

Run on the machine with GPU/resources:
    uv run python worker.py
    uv run python worker.py wss://ui-server.example.com/ws/worker

LLM backend: LLM_URL (Ollama native, or any OpenAI-compatible /v1 endpoint:
vLLM, mlx-lm, llama-server). See .env.example.

Retrieval pipeline:
    1. Hybrid search: vector (ChromaDB) + keyword (BM25)
    2. Re-ranking: cross-encoder scores merged results
    3. Expanded context: retrieve full document text for top hits

Protocol (JSON over WebSocket):
    UI -> Worker:  {"type": "ask",     "id": "...", "question": "..."}
    UI -> Worker:  {"type": "ask-pdf", "id": "...", "pdf_base64": "..."}
    Worker -> UI:  {"type": "status",  "chunks": N, "model": "..."}
    Worker -> UI:  {"type": "sources", "id": "...", "sources": [{"filename": ..., "year": ...}, ...]}
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
import threading

# Disable telemetry and analytics before importing any third-party libraries
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# huggingface_hub and transformers freeze these into constants at import time,
# so they must be decided here, not inside download_models().
if "--download" not in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # ChromaDB
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

import chromadb
import fitz
import httpx
import websockets

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import (
    BM25_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    FULL_TEXTS_PATH,
    bm25_tokenize,
    chunk_id,
    chunk_text,
    clean_pages,
    embed_query,
    load_embed_model,
)

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger(__name__)

DEFAULT_URL = os.getenv("UI_URL", "ws://localhost:7777/ws/worker")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")  # shared secret checked by the UI server
DEFAULT_MODEL = os.getenv("MODEL", "gemma4:26b")
MODEL = DEFAULT_MODEL  # overridden by --model argument

# LLM backend. Two dialects:
#   ollama  — Ollama's native API (URL without /v1). Context window is set per
#             request with num_ctx, thinking can be switched off.
#   openai  — any OpenAI-compatible server: vLLM, mlx-lm, llama-server, or
#             Ollama's /v1. Context window is a server-side setting there.
# LLM_API=auto picks "openai" when the URL path ends in /v1, else "ollama".
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434").rstrip("/")
LLM_API = os.getenv("LLM_API", "auto")
if LLM_API == "auto":
    LLM_API = "openai" if LLM_URL.endswith("/v1") else "ollama"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_THINK = os.getenv("LLM_THINK", "0") == "1"  # reasoning models: keep thinking on?
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "1800"))  # seconds, long prompts on slow hardware

# Multilingual cross-encoder for re-ranking. On a CUDA box with headroom,
# RERANK_MODEL=BAAI/bge-reranker-v2-m3 scores better but is ~5x slower.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Retrieval parameters (tuned for 96GB Mac Studio Ultra M3 / RTX 6000 Pro)
VECTOR_TOP_K = 80  # candidates from vector search
BM25_TOP_K = 60  # candidates from BM25 search
MAX_CHUNKS_PER_DOC = 6  # cap per document before re-ranking, so one report can't hog the list
RERANK_TOP_K = 20  # final results after re-ranking
NEIGHBOUR_RADIUS = 2  # chunks before/after each hit included in the context

# Context budget. Ollama silently truncates prompts longer than num_ctx, so
# the char cap is derived from the token window. Measured on this corpus:
# Swedish text tokenizes to ~3.1 chars/token with Gemma's tokenizer.
NUM_CTX = int(os.getenv("NUM_CTX", "65536"))  # Ollama context window (tokens)
CHARS_PER_TOKEN = 3.0
RESERVE_TOKENS = 4096  # system prompt + question + generated answer
MAX_CONTEXT_CHARS = int((NUM_CTX - RESERVE_TOKENS) * CHARS_PER_TOKEN)
PDF_MAX_CHARS = 60000  # uploaded PDF excerpt; shares the budget with context

SYSTEM_PROMPT = """Du är en analysassistent för Vetenskapsrådets dokument: rapporter, utvärderingar, utlysningar, forskningsöversikter, yttranden, beslut och föreskrifter. Du svarar utifrån de dokumentutdrag som tillhandahålls som kontext.

Du kan hantera olika typer av frågor:

**Sakfrågor och beslutsstöd** (t.ex. "Vad säger Vetenskapsrådet om X?", "Hur ska vi hantera detta ärende?"):
1. Analysera de relevanta dokumenten som tillhandahålls som kontext
2. Identifiera ställningstaganden, principer, krav och mönster i dem
3. Tillämpa dem på frågan eller den nya situationen
4. Ge ett tydligt svar eller en rekommendation med motivering
5. Hänvisa till vilka dokument som stödjer svaret (referera med filnamn och år)

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

PDF_SYSTEM_PROMPT = """Du är en analysassistent för Vetenskapsrådets dokument. Du analyserar ett uppladdat dokument (PDF) och jämför det med Vetenskapsrådets tidigare dokument som tillhandahålls som kontext.

När du svarar:
1. Sammanfatta det uppladdade dokumentet kort
2. Identifiera relevanta dokument från kontexten
3. Jämför det uppladdade dokumentet med tidigare ställningstaganden, krav och beslut
4. Ge en tydlig bedömning eller rekommendation med motivering
5. Hänvisa till vilka dokument som stödjer din bedömning (referera med filnamn och år)
6. Om det uppladdade dokumentet inte liknar något i kontexten, säg det tydligt istället för att gissa.

Om kontexten inte innehåller relevant material, säg det tydligt istället för att gissa. Svara alltid på svenska."""

# Models and indexes, populated by load_models()
embed_model: SentenceTransformer | None = None
rerank_model: CrossEncoder | None = None
bm25_data: dict | None = None
full_texts: dict[str, str] = {}
doc_table: list[dict] = []  # one metadata row per document (chunk_index == 0)


def load_models():
    """Load embedding model, re-ranker, BM25 index and full texts."""
    global embed_model, rerank_model, bm25_data, full_texts, doc_table

    log.info("Loading embedding model %s...", EMBED_MODEL)
    embed_model = load_embed_model()

    log.info("Loading re-ranking model %s...", RERANK_MODEL)
    rerank_model = CrossEncoder(RERANK_MODEL)

    log.info("Loading BM25 index...")
    if BM25_PATH.exists():
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        log.info("  BM25: %d documents", len(bm25_data["corpus"]))
    else:
        log.warning("  BM25 index missing — run ingest.py first for hybrid search.")

    if FULL_TEXTS_PATH.exists():
        with open(FULL_TEXTS_PATH) as f:
            full_texts = json.load(f)
        log.info("  Full texts: %d documents", len(full_texts))

    doc_table = load_doc_table()
    log.info("  Document table: %d documents", len(doc_table))

    log.info("All models loaded. Context window: %d tokens (~%d chars).", NUM_CTX, MAX_CONTEXT_CHARS)
    log.info("LLM backend: %s at %s (%d models listed)", LLM_API, LLM_URL, len(llm_list_models()))


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_doc_table() -> list[dict]:
    """One metadata row per document, read in pages (ChromaDB caps unpaginated gets)."""
    collection = get_collection()
    rows, offset, page = [], 0, 1000
    while True:
        res = collection.get(where={"chunk_index": 0}, limit=page, offset=offset, include=["metadatas"])
        rows.extend(res["metadatas"])
        offset += len(res["ids"])
        if len(res["ids"]) < page:
            break
    rows.sort(key=lambda m: m.get("filename", ""))
    return rows


def vector_search(query: str, top_k: int = VECTOR_TOP_K) -> list[dict]:
    """Search using vector similarity."""
    collection = get_collection()
    query_embedding = embed_query(embed_model, query)

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

    tokens = bm25_tokenize(query)
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
                    "source": item.get("source", ""),
                    "chunk_index": item.get("chunk_index"),
                },
                "score": float(scores[idx]),
                "source": "bm25",
            }
        )
    return hits


def merge_and_deduplicate(
    vector_hits: list[dict], bm25_hits: list[dict], max_per_doc: int = MAX_CHUNKS_PER_DOC
) -> list[dict]:
    """Merge results from both sources, deduplicate, and cap chunks per document.

    Vector and BM25 lists are interleaved so neither source dominates the
    candidate set that goes to the re-ranker.
    """
    seen_texts: set[str] = set()
    per_doc: dict[str, int] = {}
    merged = []

    interleaved = []
    for pair in zip(vector_hits, bm25_hits):
        interleaved.extend(pair)
    longer = vector_hits if len(vector_hits) > len(bm25_hits) else bm25_hits
    interleaved.extend(longer[min(len(vector_hits), len(bm25_hits)) :])

    for hit in interleaved:
        key = hit["text"][:200]
        if key in seen_texts:
            continue
        filename = hit["metadata"].get("filename", "")
        if per_doc.get(filename, 0) >= max_per_doc:
            continue
        seen_texts.add(key)
        per_doc[filename] = per_doc.get(filename, 0) + 1
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


def _doc_header(meta: dict) -> str:
    """Short citation header: filename plus the metadata worth showing the LLM."""
    bits = []
    if meta.get("title"):
        bits.append(str(meta["title"]))
    if meta.get("doc_type") and meta["doc_type"] != "okänd":
        bits.append(str(meta["doc_type"]))
    if meta.get("year"):
        bits.append(str(meta["year"]))
    if meta.get("diarienummer"):
        bits.append(f"dnr {meta['diarienummer']}")
    return " | ".join(bits)


def _source_entry(meta: dict) -> dict:
    """Metadata the UI needs to list and link a source document."""
    return {
        "filename": meta.get("filename", "okänd"),
        "title": meta.get("title") or "",
        "doc_type": meta.get("doc_type") or "",
        "year": meta.get("year"),
        "diarienummer": meta.get("diarienummer") or "",
        "url": meta.get("url") or "",
        "page_url": meta.get("page_url") or "",
    }


def _unique_sources(hits: list[dict]) -> list[dict]:
    """One entry per document, in hit order."""
    seen, out = set(), []
    for hit in hits:
        meta = hit["metadata"]
        filename = meta.get("filename", "okänd")
        if filename in seen:
            continue
        seen.add(filename)
        out.append(_source_entry(meta))
    return out


def _fetch_windows(source: str, centers: list[int], total: int, radius: int) -> list[tuple[int, str]]:
    """Fetch chunks around each hit by id; merged, ordered by chunk index."""
    wanted = set()
    for c in centers:
        for j in range(max(0, c - radius), min(total, c + radius + 1)):
            wanted.add(j)
    if not wanted:
        return []
    ids = [chunk_id(source, j) for j in sorted(wanted)]
    res = get_collection().get(ids=ids, include=["documents", "metadatas"])
    rows = [(m["chunk_index"], d) for d, m in zip(res["documents"], res["metadatas"])]
    rows.sort()
    return rows


def expand_context(hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context from the matched chunks plus their neighbours.

    For every hit we include the chunk itself and NEIGHBOUR_RADIUS chunks on
    each side, fetched from ChromaDB by id. Windows in the same document are
    merged and ordered; gaps are marked. If the budget is exceeded the radius
    shrinks, and as a last resort documents are dropped from the end.
    """
    by_source: dict[str, dict] = {}
    order: list[str] = []
    for hit in hits:
        meta = hit["metadata"]
        filename = meta.get("filename", "okänd")
        if filename not in by_source:
            by_source[filename] = {"meta": meta, "hits": [], "score": hit.get("rerank_score", hit.get("score", 0))}
            order.append(filename)
        by_source[filename]["hits"].append(hit)

    def build(radius: int) -> list[str]:
        parts = []
        for filename in order:
            entry = by_source[filename]
            meta = entry["meta"]
            source = meta.get("source", "")
            centers = [h["metadata"]["chunk_index"] for h in entry["hits"] if h["metadata"].get("chunk_index") is not None]
            total = int(meta.get("total_chunks") or (max(centers) + 1 if centers else 0))
            rows = _fetch_windows(source, centers, total, radius) if source and centers else []
            if not rows:  # fallback: just the matched chunks
                rows = [(i, h["text"]) for i, h in enumerate(entry["hits"])]
            body, prev = [], None
            for idx, text in rows:
                if prev is not None and idx != prev + 1:
                    body.append("[...]")
                body.append(text)
                prev = idx
            header = _doc_header(meta)
            section = ", ".join(sorted({h["metadata"].get("section", "") for h in entry["hits"]} - {"", "okänd"}))
            title = f"### Källa: {filename}"
            if header:
                title += f"\n{header}"
            if section:
                title += f" [{section}]"
            title += f" (relevans: {entry['score']:.2f})"
            parts.append(f"{title}\n\n" + "\n\n".join(body))
        return parts

    for radius in range(NEIGHBOUR_RADIUS, -1, -1):
        parts = build(radius)
        if sum(len(p) for p in parts) <= max_chars:
            break

    out, total = [], 0
    for p in parts:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p)
    return "\n\n---\n\n".join(out)


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


def retrieve_broad(query: str, top_k: int = 50) -> tuple[str, list[dict]]:
    """Broader retrieval for aggregate/analytical questions. Returns (context, sources)."""
    vector_hits = vector_search(query, top_k=min(top_k, 100))
    bm25_hits = bm25_search(query, top_k=min(top_k, 100))
    candidates = merge_and_deduplicate(vector_hits, bm25_hits)

    if not candidates:
        return "", []

    ranked = rerank(query, candidates, top_k=top_k)
    if not ranked:
        return "", []

    # For analytical questions, list all matching documents with brief excerpts
    sources: list[dict] = []
    seen_sources = set()
    context_parts = []
    total_chars = 0

    for hit in ranked:
        filename = hit["metadata"].get("filename", "okänd")
        if filename in seen_sources:
            continue
        score = hit.get("rerank_score", hit.get("score", 0))
        excerpt = hit["text"][:300].replace("\n", " ")
        entry = f"### {filename} (relevans: {score:.2f})\n{excerpt}..."
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        seen_sources.add(filename)
        sources.append(_source_entry(hit["metadata"]))
        context_parts.append(entry)
        total_chars += len(entry)

    context = "\n\n".join(context_parts)
    return context, sources


import re as _re

# Patterns that indicate analytical/aggregate questions
_ANALYTICAL_PATTERNS = [
    r"hur många",
    r"hur\s+stor\s+andel",
    r"vilka dokument",
    r"vilka beslut",
    r"lista alla",
    r"\bräkna\b",
    r"\bantal",
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


# Questions about counts/distributions over the whole corpus are answered from
# metadata, not from retrieved excerpts.
_AGGREGATE_PATTERNS = [
    r"\bper\s+år\b",
    r"\bper\s+(dokument)?typ\b",
    r"\bper\s+språk\b",
    r"\bfördelning",
    r"\bhur\s+många\s+(dokument|beslut|yttranden|rapporter|utlysningar|utvärderingar|pdf)",
    r"\bantal(et)?\s+(dokument|beslut|yttranden|rapporter|utlysningar|utvärderingar)",
    r"\böver\s+(åren|tid)\b",
    r"\bvilka\s+år\b",
    r"\bäldsta\b|\bnyaste\b|\bsenaste\s+(dokument|beslut|yttrande)",
]

_DOC_TYPE_WORDS = {
    "beslut": "beslut",
    "yttrande": "yttrande",
    "yttranden": "yttrande",
    "rapport": "rapport",
    "rapporter": "rapport",
    "utlysning": "utlysning",
    "utlysningar": "utlysning",
    "utvärdering": "utvärdering",
    "utvärderingar": "utvärdering",
    "årsredovisning": "årsredovisning",
    "årsredovisningar": "årsredovisning",
    "föreskrift": "föreskrift",
    "föreskrifter": "föreskrift",
    "presentation": "presentation",
    "presentationer": "presentation",
}


def _is_aggregate(question: str) -> bool:
    q = question.lower()
    return any(_re.search(p, q) for p in _AGGREGATE_PATTERNS)


def stats_context(question: str) -> tuple[str, int]:
    """Exact counts from document metadata for aggregate questions.

    Returns (markdown, number of documents in the selection).
    """
    if not doc_table:
        return "", 0
    q = question.lower()
    wanted = {t for word, t in _DOC_TYPE_WORDS.items() if _re.search(rf"\b{word}\b", q)}
    rows = [m for m in doc_table if not wanted or m.get("doc_type") in wanted]
    label = ", ".join(sorted(wanted)) if wanted else "alla dokument"

    def table(title: str, key: str, rows: list[dict]) -> str:
        counts: dict = {}
        for m in rows:
            counts[m.get(key, "okänd") or "okänd"] = counts.get(m.get(key, "okänd") or "okänd", 0) + 1
        lines = [f"**{title}**", "", "| " + key + " | antal |", "|---|---|"]
        for k in sorted(counts, key=lambda x: (isinstance(x, str), x)):
            lines.append(f"| {k} | {counts[k]} |")
        return "\n".join(lines)

    parts = [
        f"Databasen innehåller {len(doc_table)} dokument totalt; {len(rows)} matchar urvalet ({label}). "
        "Siffrorna nedan är exakta och kommer från dokumentens metadata, inte från sökträffar.",
        table(f"Antal per år ({label})", "year", rows),
        table(f"Antal per dokumenttyp ({label})", "doc_type", rows),
        table(f"Antal per språk ({label})", "language", rows),
    ]
    if wanted and len(rows) <= 60:
        listing = [f"- {m['filename']} ({m.get('year', '?')})" for m in sorted(rows, key=lambda m: (m.get('year', 0) or 0, m['filename']))]
        parts.append(f"**Dokument i urvalet ({label})**\n" + "\n".join(listing))
    return "\n\n".join(parts), len(rows)


def _extract_search_terms(question: str) -> list[str]:
    """Extract quoted strings or key nouns as text search terms."""
    # First try quoted strings: "katter" or 'katter'
    quoted = _re.findall(r'["\u201c\u201d\'](.*?)["\u201c\u201d\']', question)
    if quoted:
        return quoted
    return []


def retrieve(query: str, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
    """Full retrieval pipeline: hybrid search -> rerank -> expand context. Returns (context, sources)."""
    # Step 1: Hybrid search
    vector_hits = vector_search(query)
    bm25_hits = bm25_search(query)
    candidates = merge_and_deduplicate(vector_hits, bm25_hits)

    if not candidates:
        return "", []

    # Step 2: Re-rank
    ranked = rerank(query, candidates)

    if not ranked:
        return "", []

    # Step 3: Expand context
    context = expand_context(ranked, max_chars=max_chars)

    return context, _unique_sources(ranked)


def _llm_headers() -> dict:
    return {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}


def llm_chat_stream(model: str, messages: list[dict]):
    """Stream assistant text from the configured LLM backend, one piece at a time.

    Reasoning/thinking tokens are never yielded.
    """
    if LLM_API == "ollama":
        url = f"{LLM_URL}/api/chat"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": LLM_THINK,
            "options": {"num_ctx": NUM_CTX},
        }
    else:
        url = f"{LLM_URL}/chat/completions"
        body = {"model": model, "messages": messages, "stream": True}
        if not LLM_THINK:
            # Honoured by vLLM/mlx-lm for models with a thinking switch; ignored elsewhere.
            body["chat_template_kwargs"] = {"enable_thinking": False}

    with httpx.stream("POST", url, json=body, headers=_llm_headers(), timeout=LLM_TIMEOUT) as resp:
        if resp.status_code != 200:
            detail = resp.read().decode(errors="replace")[:500]
            raise RuntimeError(f"LLM-servern svarade {resp.status_code}: {detail}")
        for line in resp.iter_lines():
            if not line:
                continue
            if LLM_API == "ollama":
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"LLM-fel: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    return
            else:
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError(f"LLM-fel: {data['error']}")
                for choice in data.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece


def llm_list_models() -> list[str]:
    """Model ids the backend can serve; empty list if the backend is unreachable."""
    try:
        if LLM_API == "ollama":
            r = httpx.get(f"{LLM_URL}/api/tags", headers=_llm_headers(), timeout=10)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        r = httpx.get(f"{LLM_URL}/models", headers=_llm_headers(), timeout=10)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception as ex:
        log.warning("Could not list models from %s: %s", LLM_URL, ex)
        return []


def process_question(question: str, model: str = ""):
    """Generator yielding (sources, num_sources, token) tuples.

    sources lists the cited documents; num_sources may exceed len(sources)
    for aggregate answers, where the count comes from metadata.
    """
    model = model or MODEL
    analytical = _is_analytical(question)
    aggregate = _is_aggregate(question)
    search_terms = _extract_search_terms(question)

    stats, stats_docs = stats_context(question) if aggregate else ("", 0)

    # Build text search results if we have explicit search terms
    text_search_context = ""
    text_search_sources: list[dict] = []
    if search_terms:
        ts_results = text_search(search_terms)
        if ts_results:
            by_name = {m.get("filename"): m for m in doc_table}
            text_search_sources = [_source_entry(by_name.get(r["filename"], {"filename": r["filename"]})) for r in ts_results[:50]]
            parts = [f"**Textsökning för {search_terms}:** {len(ts_results)} dokument matchade.\n"]
            for r in ts_results[:50]:
                match_info = ", ".join(f'"{t}": {c} träffar' for t, c in r["matches"].items())
                parts.append(f"- **{r['filename']}** ({match_info})\n  Utdrag: {r['snippet']}")
            text_search_context = "\n".join(parts)
        else:
            text_search_context = f"**Textsökning för {search_terms}:** Inga dokument matchade."

    # Retrieve with broader scope for analytical questions
    if aggregate and not search_terms:
        context, sources = "", []  # metadata answers the question
    elif analytical:
        context, sources = retrieve_broad(question)
    else:
        context, sources = retrieve(question)

    seen = {s["filename"] for s in sources}
    sources += [s for s in text_search_sources if s["filename"] not in seen]
    num_sources = len(sources) or stats_docs

    if not context and not text_search_context and not stats:
        yield sources, num_sources, "Inga relevanta dokument hittades i databasen."
        return

    # Build prompt
    context_sections = []
    if stats:
        context_sections.append(f"## Databasstatistik (exakta siffror)\n\n{stats}")
    if text_search_context:
        context_sections.append(f"## Textsökningsresultat\n\n{text_search_context}")
    if context:
        context_sections.append(f"## Dokument från databasen (Kontext)\n\n{context}")

    combined_context = "\n\n---\n\n".join(context_sections)

    if aggregate:
        instruction = (
            "Svara på frågan med siffrorna i databasstatistiken ovan. Använd exakt de antal som anges, "
            "gissa aldrig och räkna inte själv från sökträffar. Visa gärna ett diagram."
        )
    elif analytical:
        instruction = "Baserat på informationen ovan, svara på frågan. Var exakt med antal och filnamn."
    else:
        instruction = "Baserat på dokumenten ovan, besvara frågan eller ge din analys och rekommendation."

    user_prompt = f"""{combined_context}

## Fråga

{question}

## Ditt svar

{instruction}"""

    log.info("[%s] %s", "aggregate" if aggregate else "analytical" if analytical else "standard", question)

    log.info("Using model: %s", model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for piece in llm_chat_stream(model, messages):
        yield sources, num_sources, piece


def process_pdf(pdf_base64: str, model: str = ""):
    """Generator yielding (sources, num_sources, token) tuples."""
    model = model or MODEL
    pdf_bytes = base64.b64decode(pdf_base64)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with fitz.open(tmp_path) as doc:
            pdf_text = clean_pages([page.get_text() for page in doc])
    except Exception as ex:
        yield [], 0, f"Kunde inte läsa PDF:en: {ex}"
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pdf_text.strip():
        yield [], 0, "Ingen text kunde extraheras från PDF:en. Filen kan vara skannad utan OCR."
        return

    # Use first chunks as search query
    chunks = chunk_text(pdf_text)
    search_text = " ".join(chunks[:3])[:2000]

    # Include as much of the uploaded PDF as possible; the rest of the
    # budget goes to retrieved context.
    pdf_excerpt = pdf_text[:PDF_MAX_CHARS]
    if len(pdf_text) > PDF_MAX_CHARS:
        pdf_excerpt += "\n\n[...dokumentet fortsätter...]"

    context, sources = retrieve(search_text, max_chars=MAX_CONTEXT_CHARS - len(pdf_excerpt))

    if not context:
        yield [], 0, "Inga relevanta dokument hittades i databasen."
        return

    user_prompt = f"""## Dokument från databasen (Kontext)

{context}

## Uppladdat dokument (PDF)

{pdf_excerpt}

## Din bedömning

Analysera det uppladdade dokumentet ovan och ge ett utlåtande baserat på Vetenskapsrådets dokument i kontexten."""

    log.debug("PDF prompt: %s", user_prompt[:200])

    log.info("Using model: %s", model)
    messages = [
        {"role": "system", "content": PDF_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for piece in llm_chat_stream(model, messages):
        yield sources, len(sources), piece


_STREAM_END = object()


async def iterate_in_thread(make_generator):
    """Run a blocking generator in a worker thread, yielding its items on the event loop.

    Retrieval, re-ranking and Ollama prompt evaluation block for tens of
    seconds. Running them on the loop starved websocket pings and the
    connection dropped mid-job; this keeps the loop responsive.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for item in make_generator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as ex:  # forwarded to the awaiting coroutine
            loop.call_soon_threadsafe(queue.put_nowait, ex)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = await queue.get()
        if item is _STREAM_END:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


async def handle_job(ws, msg: dict):
    """Process a single job and stream results back."""
    job_id = msg.get("id", "unknown")
    job_type = msg.get("type")

    # Allow per-job model override from UI
    job_model = msg.get("model") or MODEL

    try:
        if job_type == "ask":
            gen = iterate_in_thread(lambda: process_question(msg["question"], model=job_model))
        elif job_type == "ask-pdf":
            gen = iterate_in_thread(lambda: process_pdf(msg["pdf_base64"], model=job_model))
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

        sources_sent = False
        async for sources, num_sources, token in gen:
            if not sources_sent:
                sources_sent = True
                await ws.send(json.dumps({"type": "sources", "id": job_id, "sources": sources}))
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
                "documents": len(doc_table),
                "model": MODEL,
                "models": llm_list_models(),
                "backend": f"{LLM_API} @ {LLM_URL}",
            }
        )
    )


async def connect(url: str):
    """Connect to UI and process jobs forever."""
    while True:
        try:
            log.info("Connecting to %s...", url)
            headers = {"Authorization": f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}
            async with websockets.connect(
                url,
                max_size=100_000_000,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=60,
            ) as ws:
                log.info("Connected to %s", url)
                await send_status(ws)

                async for raw in ws:
                    msg = json.loads(raw)
                    await handle_job(ws, msg)

        except websockets.InvalidStatus as ex:
            log.error("UI server refused the connection (%s). Check WORKER_TOKEN.", ex)
        except (ConnectionRefusedError, OSError) as ex:
            log.warning("Connection failed: %s. Retrying in 5s...", ex)
        except websockets.ConnectionClosed:
            log.warning("Connection closed. Reconnecting in 2s...")
            await asyncio.sleep(2)
            continue

        await asyncio.sleep(5)


def download_models():
    """Download embedding and re-ranking models for offline use.

    Network access is enabled at the top of the module when --download is given.
    """
    log.info("Downloading embedding model: %s...", EMBED_MODEL)
    load_embed_model()
    log.info("Downloading re-ranking model: %s...", RERANK_MODEL)
    CrossEncoder(RERANK_MODEL)
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
    load_models()
    asyncio.run(connect(args.url))
