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

import asyncio
import base64
import json
import os
import pickle
import sys
import tempfile

from pathlib import Path

import chromadb
import fitz
import ollama
import websockets

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import BM25_PATH, CHROMA_DIR, COLLECTION_NAME, chunk_text

load_dotenv()

UI_URL = os.getenv("UI_URL", "ws://localhost:7777/ws/worker")
MODEL = "gemma3"

# Retrieval parameters
VECTOR_TOP_K = 30  # candidates from vector search
BM25_TOP_K = 30  # candidates from BM25 search
RERANK_TOP_K = 10  # final results after re-ranking
MAX_CONTEXT_CHARS = 30000  # max chars sent to LLM (fits ~8k tokens)

SYSTEM_PROMPT = """Du är en beslutstödsassistent. Du hjälper till att fatta nya beslut baserat på tidigare beslut som tillhandahålls som kontext.

När du svarar:
1. Analysera de relevanta tidigare besluten som tillhandahålls som kontext
2. Identifiera mönster, prejudikat och principer från dessa beslut
3. Tillämpa dem på den nya situationen
4. Ge en tydlig rekommendation med motivering
5. Hänvisa till vilka tidigare beslut som stödjer din rekommendation (referera med filnamn)

Om kontexten inte innehåller relevanta prejudikat, säg det tydligt istället för att gissa. Svara alltid på svenska."""

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
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading re-ranking model...")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading BM25 index...")
bm25_data = None
if BM25_PATH.exists():
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    print(f"  BM25: {len(bm25_data['corpus'])} documents")
else:
    print("  BM25 index missing — run ingest.py first for hybrid search.")

# Load full texts for expanded context
FULL_TEXTS_PATH = Path(__file__).parent / "full_texts.json"
full_texts: dict[str, str] = {}
if FULL_TEXTS_PATH.exists():
    with open(FULL_TEXTS_PATH) as f:
        full_texts = json.load(f)
    print(f"  Full texts: {len(full_texts)} documents")

print("All models loaded.")


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
    context, num_sources = retrieve(question)

    if not context:
        yield num_sources, "Inga relevanta beslut hittades i databasen."
        return

    user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Ny situation

{question}

## Din rekommendation

Baserat på tidigare beslut ovan, ge din analys och rekommendation."""

    print(question)

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

    print(user_prompt)

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
            print(f"Connecting to {url}...")
            async with websockets.connect(url, max_size=100_000_000) as ws:
                print(f"Connected to {url}")
                await send_status(ws)

                async for raw in ws:
                    msg = json.loads(raw)
                    await handle_job(ws, msg)

        except (ConnectionRefusedError, OSError) as ex:
            print(f"Connection failed: {ex}. Retrying in 5s...")
        except websockets.ConnectionClosed:
            print("Connection closed. Reconnecting in 2s...")
            await asyncio.sleep(2)
            continue

        await asyncio.sleep(5)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else UI_URL
    asyncio.run(connect(url))
