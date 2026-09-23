"""In-memory search over one user's decrypted documents. Runs in the app.

The app decrypts a user's documents only while the user's key is unlocked
and keeps this index in memory for the session; it is never written to disk.
Candidates found here go to the worker, which re-ranks them and asks the LLM.

Retrieval pipeline:
    1. Hybrid search: vector (dot product over normalized embeddings) + BM25
    2. Merge, deduplicate and cap chunks per document
    3. Each candidate carries its neighbouring chunks, so the worker can
       expand context without access to the index
"""

import base64
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

VECTOR_TOP_K = 80  # candidates from vector search
BM25_TOP_K = 60  # candidates from BM25 search
BROAD_TOP_K = 100  # both, for analytical questions
MAX_CHUNKS_PER_DOC = 6  # cap per document before re-ranking, so one report can't hog the list
NEIGHBOUR_RADIUS = 2  # chunks before/after each hit sent along for context expansion
TEXT_SEARCH_MAX_DOCS = 50

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

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

# Questions about counts/distributions over the whole library are answered
# from metadata, not from retrieved excerpts.
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


@dataclass
class Plan:
    analytical: bool
    aggregate: bool
    search_terms: list[str]

    @property
    def needs_retrieval(self) -> bool:
        """Aggregate questions without quoted terms are answered from metadata alone."""
        return not (self.aggregate and not self.search_terms)


def plan_question(question: str) -> Plan:
    q = question.lower()
    return Plan(
        analytical=any(re.search(p, q) for p in _ANALYTICAL_PATTERNS),
        aggregate=any(re.search(p, q) for p in _AGGREGATE_PATTERNS),
        # Quoted strings: "katter" or 'katter'
        search_terms=re.findall(r'["“”\'](.*?)["“”\']', question),
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass
class Doc:
    doc_id: str
    filename: str
    kind: str
    size: int
    created: float
    sha256: str
    metadata: dict
    full_text: str
    chunks: list[dict]
    embed_model: str

    def source_entry(self) -> dict:
        """What the UI needs to list and link a source document."""
        m = self.metadata
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": m.get("title") or "",
            "doc_type": m.get("doc_type") or "",
            "year": m.get("year"),
            "diarienummer": m.get("diarienummer") or "",
        }


@dataclass
class UserIndex:
    docs: dict[str, Doc] = field(default_factory=dict)
    refs: list[tuple[str, int]] = field(default_factory=list)  # row -> (doc_id, chunk_index)
    embed_model: str = ""
    skipped_models: set[str] = field(default_factory=set)  # docs embedded with another model
    version: int = 0  # bumped on every add, so views know to refresh
    # Search structures are rebuilt lazily, on the first search after a change:
    # adding documents one by one (a large upload) stays cheap.
    _vectors: list[np.ndarray] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list)
    _matrix: np.ndarray | None = None
    _bm25: BM25Okapi | None = None
    _dirty: bool = False

    @classmethod
    def build(cls, records: list[tuple[str, float, dict]], embed_model: str) -> "UserIndex":
        """records: (doc_id, created, decrypted document record from ingest.process_document)."""
        idx = cls(embed_model=embed_model)
        for doc_id, created, rec in records:
            idx.add(doc_id, created, rec)
        return idx

    def add(self, doc_id: str, created: float, rec: dict) -> None:
        doc = Doc(
            doc_id=doc_id,
            filename=rec["filename"],
            kind=rec.get("kind", ""),
            size=rec.get("size", 0),
            created=created,
            sha256=rec.get("sha256", ""),
            metadata=rec.get("metadata", {}),
            full_text=rec.get("full_text", ""),
            chunks=rec["chunks"],
            embed_model=rec.get("embed_model", ""),
        )
        self.docs[doc_id] = doc
        self.version += 1
        if self.embed_model and doc.embed_model != self.embed_model:
            self.skipped_models.add(doc.embed_model)
            return
        emb = np.frombuffer(base64.b64decode(rec["embeddings"]), dtype=np.float32).reshape(-1, rec["dim"])
        self._vectors.append(emb)
        for i, c in enumerate(doc.chunks):
            self.refs.append((doc_id, i))
            self._tokens.append(bm25_tokenize(c["text"]))
        self._dirty = True

    def _search_structures(self) -> tuple[np.ndarray | None, BM25Okapi | None]:
        if self._dirty:
            self._matrix = np.vstack(self._vectors) if self._vectors else None
            self._vectors = [self._matrix] if self._matrix is not None else []
            self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
            self._dirty = False
        return self._matrix, self._bm25

    # --- search -------------------------------------------------------------

    def _hit(self, row: int, score: float, source: str) -> dict:
        doc_id, ci = self.refs[row]
        doc = self.docs[doc_id]
        return {
            "doc_id": doc_id,
            "chunk_index": ci,
            "text": doc.chunks[ci]["text"],
            "section": doc.chunks[ci].get("section", ""),
            "score": score,
            "source": source,
        }

    def vector_search(self, qvec: np.ndarray, top_k: int) -> list[dict]:
        matrix, _ = self._search_structures()
        if matrix is None or qvec.shape[0] != matrix.shape[1]:
            return []
        scores = matrix @ qvec
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self._hit(int(r), float(scores[r]), "vector") for r in top]

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        _, bm25 = self._search_structures()
        if bm25 is None:
            return []
        scores = bm25.get_scores(bm25_tokenize(query))
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self._hit(int(r), float(scores[r]), "bm25") for r in top if scores[r] > 0]

    def candidates(self, query: str, qvec: np.ndarray, broad: bool = False) -> list[dict]:
        """Hybrid candidates with their neighbour windows, ready for the worker."""
        vhits = self.vector_search(qvec, BROAD_TOP_K if broad else VECTOR_TOP_K)
        bhits = self.bm25_search(query, BROAD_TOP_K if broad else BM25_TOP_K)
        merged = merge_and_deduplicate(vhits, bhits)
        for hit in merged:
            doc = self.docs[hit["doc_id"]]
            lo = max(0, hit["chunk_index"] - NEIGHBOUR_RADIUS)
            hi = min(len(doc.chunks), hit["chunk_index"] + NEIGHBOUR_RADIUS + 1)
            hit["window"] = [[j, doc.chunks[j]["text"]] for j in range(lo, hi)]
            hit["meta"] = {**doc.source_entry(), "total_chunks": len(doc.chunks)}
        return merged

    # --- whole-library answers ---------------------------------------------

    def text_search(self, terms: list[str]) -> tuple[str, list[dict]]:
        """Exact, case-insensitive substring counts across all full texts."""
        results = []
        for doc in self.docs.values():
            low = doc.full_text.lower()
            matches = {t: low.count(t.lower()) for t in terms if t and low.count(t.lower())}
            if not matches:
                continue
            first = next(iter(matches))
            i = low.find(first.lower())
            snippet = doc.full_text[max(0, i - 100) : i + len(first) + 200].replace("\n", " ").strip()
            results.append((sum(matches.values()), doc, matches, snippet))
        results.sort(key=lambda r: r[0], reverse=True)
        if not results:
            return f"**Textsökning för {terms}:** Inga dokument matchade.", []
        top = results[:TEXT_SEARCH_MAX_DOCS]
        parts = [f"**Textsökning för {terms}:** {len(results)} dokument matchade.\n"]
        for _, doc, matches, snippet in top:
            info = ", ".join(f'"{t}": {c} träffar' for t, c in matches.items())
            parts.append(f"- **{doc.filename}** ({info})\n  Utdrag: ...{snippet}...")
        return "\n".join(parts), [doc.source_entry() for _, doc, _, _ in top]

    def stats_context(self, question: str) -> tuple[str, int]:
        """Exact counts from document metadata. Returns (markdown, documents in the selection)."""
        if not self.docs:
            return "", 0
        q = question.lower()
        wanted = {t for word, t in _DOC_TYPE_WORDS.items() if re.search(rf"\b{word}\b", q)}
        rows = [d for d in self.docs.values() if not wanted or d.metadata.get("doc_type") in wanted]
        label = ", ".join(sorted(wanted)) if wanted else "alla dokument"

        def table(title: str, key: str) -> str:
            counts: dict = {}
            for d in rows:
                k = d.metadata.get(key) or "okänd"
                counts[k] = counts.get(k, 0) + 1
            lines = [f"**{title}**", "", f"| {key} | antal |", "|---|---|"]
            for k in sorted(counts, key=lambda x: (isinstance(x, str), x)):
                lines.append(f"| {k} | {counts[k]} |")
            return "\n".join(lines)

        parts = [
            f"Biblioteket innehåller {len(self.docs)} dokument totalt; {len(rows)} matchar urvalet ({label}). "
            "Siffrorna nedan är exakta och kommer från dokumentens metadata, inte från sökträffar.",
            table(f"Antal per år ({label})", "year"),
            table(f"Antal per dokumenttyp ({label})", "doc_type"),
            table(f"Antal per språk ({label})", "language"),
        ]
        if wanted and len(rows) <= 60:
            listing = [
                f"- {d.filename} ({d.metadata.get('year', '?')})"
                for d in sorted(rows, key=lambda d: (d.metadata.get("year") or 0, d.filename))
            ]
            parts.append(f"**Dokument i urvalet ({label})**\n" + "\n".join(listing))
        return "\n\n".join(parts), len(rows)


def merge_and_deduplicate(
    vector_hits: list[dict], bm25_hits: list[dict], max_per_doc: int = MAX_CHUNKS_PER_DOC
) -> list[dict]:
    """Merge results from both sources, deduplicate, and cap chunks per document.

    Vector and BM25 lists are interleaved so neither source dominates the
    candidate set that goes to the re-ranker.
    """
    interleaved = []
    for pair in zip(vector_hits, bm25_hits):
        interleaved.extend(pair)
    longer = vector_hits if len(vector_hits) > len(bm25_hits) else bm25_hits
    interleaved.extend(longer[min(len(vector_hits), len(bm25_hits)) :])

    seen: set[tuple[str, int]] = set()
    per_doc: dict[str, int] = {}
    merged = []
    for hit in interleaved:
        key = (hit["doc_id"], hit["chunk_index"])
        if key in seen or per_doc.get(hit["doc_id"], 0) >= max_per_doc:
            continue
        seen.add(key)
        per_doc[hit["doc_id"]] = per_doc.get(hit["doc_id"], 0) + 1
        merged.append(hit)
    return merged
