"""NiceGUI web app — accounts, encrypted document library and questions.

Security model (see README):
  * Accounts are created by an admin (users.py). First login forces a new
    password, which also protects the user's private key (vault.py).
  * The private key is unlocked with the password at every login and
    lives only in this process's memory, for this browser session. It is
    dropped on logout, after SESSION_IDLE_MINUTES of inactivity, after
    SESSION_MAX_HOURS, and on restart. Nothing decrypted is written to disk.
  * Uploaded files are read into memory (never spooled to temp files), sent
    to the worker for text extraction and embedding, and the result is
    encrypted with the user's public key before it is stored (vault.py).
  * Nothing here logs questions, answers, file names or document content.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import resource
import secrets
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from functools import partial
from urllib.parse import unquote, urlparse

import numpy as np
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from nacl.public import PrivateKey
from nicegui import app, run, ui
from starlette.requests import ClientDisconnect
from starlette.websockets import WebSocket, WebSocketDisconnect

import plotly.graph_objects as go

import vault
from retrieval import UserIndex, plan_question

load_dotenv()

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Secrets come from the environment (.env). Refuse to start without them.
STORAGE_SECRET = os.getenv("STORAGE_SECRET", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
_missing = [n for n, v in (("STORAGE_SECRET", STORAGE_SECRET), ("WORKER_TOKEN", WORKER_TOKEN)) if len(v) < 32]
if _missing:
    raise SystemExit(
        f"Saknade eller för korta miljövariabler (minst 32 tecken): {', '.join(_missing)}. "
        "Sätt dem i .env (se .env.example), t.ex. med: openssl rand -hex 32"
    )

COOKIE_SECURE = os.getenv("COOKIE_SECURE", "1") == "1"  # 0 only for plain-http development
SESSION_IDLE = int(os.getenv("SESSION_IDLE_MINUTES", "30")) * 60
SESSION_MAX = int(os.getenv("SESSION_MAX_HOURS", "12")) * 3600
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
INGEST_TIMEOUT = 900  # seconds; OCR of a large scanned PDF is slow
ANSWER_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "1800"))  # longest silence from the worker during an answer

# Fallback list, used only until a worker reports what its LLM backend
# actually serves. Labels here also decorate matching ids from the worker.
AVAILABLE_MODELS = [
    {
        "value": "gemma4:26b",
        "label": "Gemma 4 26B MoE — snabb vid långa kontexter, hög kvalitet",
    },
    {"value": "gemma4:31b", "label": "Gemma 4 31B — bäst kvalitet, långsammare"},
    {"value": "gemma4:12b", "label": "Gemma 4 12B — snabbast, kompakt"},
    {
        "value": "qwen2.5:72b",
        "label": "Qwen 2.5 72B — stark flerspråkig, bra på svenska",
    },
    {"value": "llama3.3:70b", "label": "Llama 3.3 70B — hög kvalitet, 128K kontext"},
    {"value": "mistral-small:24b", "label": "Mistral Small 24B — snabb, bra kvalitet"},
]
DEFAULT_MODEL = os.getenv("MODEL", "gemma4:26b")
_MODEL_LABELS = {m["value"]: m["label"] for m in AVAILABLE_MODELS}


def model_options() -> dict[str, str]:
    """Dropdown options: models the connected worker's backend lists, else the fallback."""
    reported = (worker_status_info or {}).get("models") or []
    if reported:
        return {m: _MODEL_LABELS.get(m, m) for m in reported}
    return dict(_MODEL_LABELS)

_CHART_RE = re.compile(r"```chart\s*\n(.*?)```", re.DOTALL)


def _doc_stem(filename: str) -> str:
    return re.sub(r"\.(pdf|docx|html?)$", "", filename, flags=re.IGNORECASE)


def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", text)


# Code blocks and inline code: left exactly as written.
_CODE_MD = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)


def protect_source_names(text: str, sources: list[dict]) -> str:
    """Escape Markdown characters in cited file names, so "dok_0413.pdf, dok_0658.pdf"
    is not read as italics. A code span that is exactly a file name becomes plain text."""
    names = {s.get("filename", "") for s in sources} | {_doc_stem(s.get("filename", "")) for s in sources}
    names = sorted((n for n in names if n and re.search(r"[\\`*_\[\]<>]", n)), key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile("|".join(re.escape(n) for n in names))
    out = []
    for i, piece in enumerate(_CODE_MD.split(text)):
        if i % 2 == 0:
            out.append(pattern.sub(lambda m: _md_escape(m.group(0)), piece))
        elif piece.startswith("`") and not piece.startswith("```") and pattern.fullmatch(piece[1:-1].strip()):
            out.append(_md_escape(piece[1:-1].strip()))
        else:
            out.append(piece)
    return "".join(out)


def normalize_bullets(text: str) -> str:
    """Put inline ' * item' bullets on their own lines so markdown renders a list.

    Only a '*' that follows sentence punctuation counts, so "2 * 3" is left
    alone. A blank line is inserted before a list that follows prose.
    """
    text = re.sub(r"(?<=[:.!?;)\"\u201d]) \* (?=\S)", "\n* ", text)
    lines, out = text.split("\n"), []
    for line in lines:
        if line.startswith("* ") and out and out[-1].strip() and not re.match(r"\s*[*\-] ", out[-1]):
            out.append("")
        out.append(line)
    return "\n".join(out)


def parse_chart_segments(text: str) -> list[tuple[str, str]]:
    """Split text into ('md', markdown) and ('chart', json_str) segments."""
    segments = []
    last_end = 0
    for m in _CHART_RE.finditer(text):
        if m.start() > last_end:
            segments.append(("md", text[last_end : m.start()]))
        segments.append(("chart", m.group(1).strip()))
        last_end = m.end()
    if last_end < len(text):
        segments.append(("md", text[last_end:]))
    return segments


def build_plotly_figure(spec: dict, dark: bool = False) -> go.Figure:
    """Convert a simplified chart spec to a Plotly figure."""
    chart_type = spec.get("type", "bar")
    title = spec.get("title", "")
    data = spec.get("data", [])
    x_label = spec.get("x_label", "")
    y_label = spec.get("y_label", "")

    labels = [d.get("label", "") for d in data]
    values = [d.get("value", 0) for d in data]

    # Theme colors
    fg = "#ffffff" if dark else "#000000"
    bg = "#1a1a1a" if dark else "#ffffff"
    grid = "#444444" if dark else "#e0e0e0"
    # Palette that works on both light and dark backgrounds
    palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    if chart_type == "pie":
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=palette[: len(values)]),
                    textfont=dict(color=fg),
                    outsidetextfont=dict(color=fg),
                )
            ]
        )
    elif chart_type == "line":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=labels,
                    y=values,
                    mode="lines+markers",
                    line=dict(color=palette[0]),
                    marker=dict(color=palette[0]),
                )
            ]
        )
    elif chart_type == "scatter":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=labels,
                    y=values,
                    mode="markers",
                    marker=dict(color=palette[0]),
                )
            ]
        )
    else:  # bar
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker=dict(color=palette[: len(values)]),
                )
            ]
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=fg)),
        xaxis_title=dict(text=x_label, font=dict(color=fg)),
        yaxis_title=dict(text=y_label, font=dict(color=fg)),
        xaxis=dict(
            color=fg,
            gridcolor=grid,
            linecolor=grid,
            zerolinecolor=grid,
        ),
        yaxis=dict(
            color=fg,
            gridcolor=grid,
            linecolor=grid,
            zerolinecolor=grid,
        ),
        legend=dict(font=dict(color=fg)),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# JS snippet injected once per page to watch for dark-mode changes and
# re-theme all Plotly charts automatically.
_PLOTLY_THEME_JS = """
<script>
(function() {
    function isDark() {
        return document.body.classList.contains('body--dark');
    }

    function themeCharts() {
        var dark = isDark();
        var fg = dark ? '#ffffff' : '#000000';
        var bg = dark ? '#1a1a1a' : '#ffffff';
        var grid = dark ? '#444444' : '#e0e0e0';
        document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
            Plotly.relayout(el, {
                'paper_bgcolor': bg,
                'plot_bgcolor': bg,
                'title.font.color': fg,
                'xaxis.color': fg,
                'xaxis.gridcolor': grid,
                'xaxis.linecolor': grid,
                'xaxis.zerolinecolor': grid,
                'yaxis.color': fg,
                'yaxis.gridcolor': grid,
                'yaxis.linecolor': grid,
                'yaxis.zerolinecolor': grid,
                'legend.font.color': fg,
                'xaxis.title.font.color': fg,
                'yaxis.title.font.color': fg,
            });
            // Re-style trace text colors per chart type
            var data = el.data;
            if (data && data[0]) {
                if (data[0].type === 'pie') {
                    Plotly.restyle(el, {
                        'textfont.color': fg,
                        'outsidetextfont.color': fg
                    });
                } else if (data[0].type === 'bar') {
                    Plotly.restyle(el, {
                        'textfont.color': fg,
                        'marker.line.color': bg,
                        'marker.line.width': 1
                    });
                }
            }
        });
    }

    // Watch for Quasar dark-mode class changes on <body>
    var obs = new MutationObserver(function(mutations) {
        for (var m of mutations) {
            if (m.attributeName === 'class') { themeCharts(); break; }
        }
    });
    obs.observe(document.body, {attributes: true});

    // Also run on load after a short delay (charts may render after DOM ready)
    setTimeout(themeCharts, 500);
})();
</script>
"""

# --- Sessions -----------------------------------------------------------------
#
# Keyed by NiceGUI's browser id, which lives only in the signed session cookie.
# Nothing session-related is written to app.storage.user (that is a file).

# "password": must replace a temporary password; "unlock": legacy account, one
# last passphrase entry to move its key to the password; "ready": key unlocked.
STAGE_PAGES = {"password": "/password", "unlock": "/unlock", "ready": "/"}


@dataclass
class Session:
    user_id: int
    username: str
    stage: str  # see STAGE_PAGES
    created: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    key: PrivateKey | None = None
    index: UserIndex | None = None
    index_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turns: list[dict] = field(default_factory=list)  # {"question", "text", "sources"}
    active_job: str | None = None
    uploads: dict[str, dict] = field(default_factory=dict)  # upload id -> state, see _new_upload
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    saving: int = 0  # vault writes in progress; they are never interrupted
    receiving: bool = False  # an upload body is being read (one at a time)
    auth_fp: bytes = b""  # fingerprint of password hash + wrapped key when the session was granted
    migration: tuple[bytes, bytes] | None = None  # legacy account: (kek, salt) from the password, until converted
    changing_auth: bool = False  # this session is changing its own password/key; skip the stale-credentials check

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def expired(self) -> bool:
        now = time.monotonic()
        return now - self.last_seen > SESSION_IDLE or now - self.created > SESSION_MAX

    def wipe(self) -> None:
        """Drop the key and everything decrypted. Python cannot zero memory,
        but without references the garbage collector frees it."""
        self.key = None
        self.migration = None
        self.index = None
        self.turns = []
        for entry in self.uploads.values():
            if entry.get("task"):
                entry["task"].cancel()
        self.uploads = {}
        self.active_job = None


SESSIONS: dict[str, Session] = {}


def _browser_id() -> str:
    return app.storage.browser["id"]


def auth_fingerprint(user) -> bytes:
    """Changes whenever the password or the key pair changes, or the account goes."""
    if user is None:
        return b""
    return hashlib.sha256(bytes(user["pw_hash"]) + b"|" + bytes(user["wrapped_private_key"] or b"")).digest()


async def own_auth_change(s: Session, fn, *args):
    """Run a change of this session's own password or key (in a thread) and
    adopt the new credentials, without the session invalidating itself midway."""
    s.changing_auth = True
    try:
        result = await run.io_bound(fn, *args)
        refresh_auth(s)
        return result
    finally:
        s.changing_auth = False


def refresh_auth(s: Session) -> None:
    """After this session itself changed password or keys: keep it, end the others."""
    s.auth_fp = auth_fingerprint(vault.get_user(s.user_id))


def _still_valid(s: Session) -> bool:
    """False once the password was changed or reset elsewhere, the key pair
    re-wrapped, or the account deleted (also from users.py)."""
    if s.changing_auth:
        return True
    fp = auth_fingerprint(vault.get_user(s.user_id))
    return bool(fp) and secrets.compare_digest(fp, s.auth_fp)


def get_session(browser_id: str | None = None) -> Session | None:
    bid = browser_id or _browser_id()
    s = SESSIONS.get(bid)
    if s and (s.expired() or not _still_valid(s)):
        end_session(bid)
        return None
    return s


def end_session(browser_id: str | None = None) -> None:
    s = SESSIONS.pop(browser_id or _browser_id(), None)
    if s:
        s.wipe()
        # Answers kept for page reloads go too, unless the user is still logged in elsewhere.
        if not any(o.user_id == s.user_id for o in SESSIONS.values()):
            for job_id, buf in list(active_jobs.items()):
                if buf["owner"] == s.user_id:
                    active_jobs.pop(job_id, None)


def invalidate_indexes(user_id: int) -> None:
    for s in SESSIONS.values():
        if s.user_id == user_id:
            s.index = None


async def stop_user_activity(user_id: int) -> None:
    """Before deleting data: cancel the user's uploads and answers in every
    session, forget conversations, and wait for vault writes in progress."""
    sessions = [s for s in SESSIONS.values() if s.user_id == user_id]
    for s in sessions:
        for entry in s.uploads.values():
            if entry["state"] not in FINAL_STATES:
                entry["state"] = "cancelled"  # the upload endpoint refuses cancelled ids
            if entry.get("task"):
                entry["task"].cancel()
        s.turns = []
        s.active_job = None
        s.index = None
    for job_id, buf in list(active_jobs.items()):
        if buf["owner"] == user_id:
            _finish(buf, error="Avbrutet: data raderades.")
            active_jobs.pop(job_id, None)
    for _ in range(600):  # a save takes well under a second; never delete mid-write
        if not any(s.saving for s in sessions):
            break
        await asyncio.sleep(0.1)


async def _save_document(s: Session, record: dict) -> None:
    """Encrypt and store, then add to the user's loaded indexes (no re-decryption).
    Shielded by callers: once started it always completes, and
    stop_user_activity waits for it."""
    s.saving += 1
    try:
        doc_id = await run.io_bound(vault.add_document, s.user_id, json.dumps(record).encode())
        created = time.time()
        for other in SESSIONS.values():
            if other.user_id == s.user_id and other.index is not None:
                other.index.add(doc_id, created, record)
    finally:
        s.saving -= 1


async def _sweep_sessions() -> None:
    while True:
        await asyncio.sleep(30)
        for bid, s in list(SESSIONS.items()):
            if s.expired() or not _still_valid(s):
                end_session(bid)
        now = time.monotonic()
        for token, (_, expires, _) in list(PENDING_LOGINS.items()):
            if expires < now:
                PENDING_LOGINS.pop(token, None)
        throttle.prune()


app.on_startup(lambda: asyncio.create_task(_sweep_sessions()))


# One-time login hand-over: token -> (session, expiry, browser id that logged in)
PENDING_LOGINS: dict[str, tuple[Session, float, str]] = {}


@app.get("/login/complete")
async def login_complete(request: Request):
    """Start the logged-in session under a new session id (new signed cookie)."""
    entry = PENDING_LOGINS.pop(request.query_params.get("t", ""), None)
    old_id = request.session.get("id")
    if not entry or entry[1] < time.monotonic() or entry[2] != old_id:
        return RedirectResponse("/login")
    session = entry[0]
    new_id = str(uuid.uuid4())
    request.session["id"] = new_id  # SessionMiddleware sends the new cookie with this response
    SESSIONS[new_id] = session
    return RedirectResponse(STAGE_PAGES[session.stage])


def guard(*stages: str) -> tuple[Session | None, Response | None]:
    """The session if it is in one of `stages`, else a redirect to where it belongs."""
    s = get_session()
    if s is None:
        return None, RedirectResponse("/login")
    if s.stage not in stages:
        return None, RedirectResponse(STAGE_PAGES[s.stage])
    s.touch()
    return s, None


class Throttle:
    """Failed-attempt counter: after `limit` failures within WINDOW seconds, locked for LOCK seconds.

    Keys: "u:<username>" (login, 5 tries), "k:<user id>" (legacy passphrase, 5 tries) and
    "ip:<address>" (login, IP_LIMIT tries: a loose limit, since many users can
    share an address)."""

    MAX_FAILS = 5
    IP_LIMIT = 30
    WINDOW = 15 * 60
    LOCK = 15 * 60
    MAX_KEYS = 50_000  # bound memory when someone sprays usernames

    def __init__(self):
        self.fails: dict[str, list[float]] = {}
        self.locked_until: dict[str, float] = {}

    def locked(self, key: str) -> bool:
        return self.locked_until.get(key, 0) > time.monotonic()

    def fail(self, key: str, limit: int = MAX_FAILS) -> None:
        now = time.monotonic()
        if len(self.fails) >= self.MAX_KEYS:
            self.prune()
        recent = [t for t in self.fails.get(key, []) if now - t < self.WINDOW] + [now]
        self.fails[key] = recent[-limit:]
        if len(recent) >= limit:
            self.locked_until[key] = now + self.LOCK
            self.fails.pop(key, None)

    def reset(self, key: str) -> None:
        self.fails.pop(key, None)
        self.locked_until.pop(key, None)

    def prune(self) -> None:
        now = time.monotonic()
        for key in [k for k, ts in self.fails.items() if not ts or now - ts[-1] >= self.WINDOW]:
            self.fails.pop(key, None)
        for key in [k for k, until in self.locked_until.items() if until <= now]:
            self.locked_until.pop(key, None)
        if len(self.fails) >= self.MAX_KEYS:  # still full: all recent, drop the oldest half
            for key in sorted(self.fails, key=lambda k: self.fails[k][-1])[: self.MAX_KEYS // 2]:
                self.fails.pop(key, None)


throttle = Throttle()

# Reverse proxies whose X-Forwarded-For is trusted (comma-separated addresses).
# Without this, every user behind a proxy has the proxy's address.
TRUSTED_PROXIES = {a.strip() for a in os.getenv("TRUSTED_PROXIES", "").split(",") if a.strip()}


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "?"
    forwarded = request.headers.get("x-forwarded-for", "")
    if peer in TRUSTED_PROXIES and forwarded:
        return forwarded.split(",")[-1].strip()  # the address our proxy saw
    return peer


# --- User index -------------------------------------------------------------


def _worker_embed_model() -> str:
    return (worker_status_info or {}).get("embed_model") or ""


def _load_index(user_id: int, key: PrivateKey, embed_model: str) -> UserIndex:
    records = []
    for d in vault.list_documents(user_id):
        try:
            rec = json.loads(vault.read_document(user_id, d["id"], key))
        except vault.VaultError:
            log.warning("A document could not be decrypted and was skipped.")
            continue
        records.append((d["id"], d["created"], rec))
    return UserIndex.build(records, embed_model)


async def ensure_index(s: Session) -> UserIndex:
    """The user's decrypted index, built on first use and cached for the session."""
    async with s.index_lock:
        model = _worker_embed_model()
        if s.index is None or (model and s.index.embed_model != model):
            if s.key is None:
                raise vault.VaultError("Nyckeln är låst.")
            s.index = await run.io_bound(_load_index, s.user_id, s.key, model)
        return s.index


# --- Worker connection state ---

worker_ws = None  # Active worker WebSocket connection
worker_status_info: dict | None = None  # Latest status from worker
pending_jobs: dict[str, asyncio.Queue] = {}  # job_id -> queue of messages

# Server-side buffer for in-progress answers so page reloads can resume.
# job_id -> {"owner": user_id, "question", "text", "done", "error", "num_sources", "sources", "events"}
active_jobs: dict[str, dict] = {}
JOB_BUFFER_TTL = 600  # seconds a finished job stays resumable after the last update

_TERMINAL = ("done", "error", "ingested", "embedding")


@app.websocket("/ws/worker")
async def worker_endpoint(ws: WebSocket):
    """WebSocket endpoint that the worker connects to. Requires the shared WORKER_TOKEN."""
    global worker_ws, worker_status_info

    auth = ws.headers.get("authorization", "")
    if not secrets.compare_digest(auth.encode(), f"Bearer {WORKER_TOKEN}".encode()):
        log.warning("Worker connection from %s rejected: bad token.", ws.client.host if ws.client else "?")
        await ws.close(code=1008)
        return

    await ws.accept()
    if worker_ws is not None:
        log.info("Replacing previously connected worker.")
    worker_ws = ws
    log.info("Worker connected.")

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            msg_type = msg.get("type")
            if msg_type == "status":
                worker_status_info = msg
            elif msg_type in ("sources", "chunk", "progress") + _TERMINAL:
                job_id = msg.get("id")
                if job_id and job_id in pending_jobs:
                    await pending_jobs[job_id].put(msg)
    except WebSocketDisconnect:
        log.info("Worker disconnected.")
    finally:
        if worker_ws is ws:
            worker_ws = None
            worker_status_info = None
            # Fail every job that was waiting on this worker instead of timing out
            for job_id, queue in list(pending_jobs.items()):
                await queue.put({"type": "error", "id": job_id, "message": "Workern kopplade ner mitt i jobbet."})


async def send_job(job: dict, timeout: float = 300):
    """Send a job to the worker and yield response messages until a terminal one."""
    if not worker_ws:
        yield {"type": "error", "id": job.get("id", ""), "message": "Ingen worker ansluten."}
        return

    job_id = job["id"]
    queue: asyncio.Queue = asyncio.Queue()
    pending_jobs[job_id] = queue
    try:
        await worker_ws.send_text(json.dumps(job))
        while True:
            msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            yield msg
            if msg["type"] in _TERMINAL:
                break
    except asyncio.TimeoutError:
        yield {"type": "error", "id": job_id, "message": "Timeout — inget svar från workern."}
    finally:
        pending_jobs.pop(job_id, None)


async def worker_call(job: dict, timeout: float) -> dict:
    """One request, one terminal reply (ingest, embed)."""
    msg = {"type": "error", "message": "Inget svar från workern."}
    async for msg in send_job({**job, "id": str(uuid.uuid4())}, timeout=timeout):
        pass
    return msg


def _new_job_buffer(job_id: str, owner: int, question: str) -> dict:
    active_jobs[job_id] = {
        "owner": owner,
        "question": question,
        "text": "",
        "done": False,
        "error": None,
        "num_sources": 0,
        "sources": [],
        "note": "",  # shown instead of the source list when no document is cited
        "events": set(),
    }
    return active_jobs[job_id]


def _notify(buf: dict) -> None:
    for event in buf["events"]:
        event.set()


def _finish(buf: dict, text: str = "", error: str | None = None) -> None:
    buf["text"] += text
    buf["error"] = error
    buf["done"] = True
    _notify(buf)


async def run_question(s: Session, job_id: str, question: str, model: str) -> None:
    """Search the user's index, then let the worker re-rank and answer. Buffered
    server-side so a page reload can resume following it."""
    buf = active_jobs[job_id]
    try:
        idx = await ensure_index(s)
        if not idx.docs:
            buf["note"] = "Biblioteket är tomt."
            _finish(buf, "Du har inga dokument ännu. Ladda upp PDF- eller DOCX-filer under **Dokument**.")
            return
        plan = plan_question(question)
        candidates: list[dict] = []
        if plan.needs_retrieval:
            msg = await worker_call({"type": "embed", "text": question}, timeout=120)
            if msg["type"] != "embedding":
                _finish(buf, error=msg.get("message", "Okänt fel"))
                return
            if msg.get("embed_model") != idx.embed_model:
                idx = await ensure_index(s)
            qvec = np.asarray(msg["vector"], dtype=np.float32)
            candidates = await run.io_bound(idx.candidates, question, qvec, plan.analytical)
        stats, stats_docs = idx.stats_context(question) if plan.aggregate else ("", 0)
        buf["note"] = (
            f"Svaret bygger på uppgifter om alla {len(idx.docs)} dokument i biblioteket (år, typ, språk), inte på enskilda dokument."
            if stats else "Inga dokument i biblioteket användes för svaret."
        )
        text_search, ts_sources = idx.text_search(plan.search_terms) if plan.search_terms else ("", [])

        job = {
            "type": "answer",
            "id": job_id,
            "model": model,
            "question": question,
            "analytical": plan.analytical,
            "aggregate": plan.aggregate,
            "candidates": candidates,
            "stats": stats,
            "text_search": text_search,
            "extra_sources": ts_sources,
            "num_docs": stats_docs,
        }
        # Reading a full context window can take minutes before the first token.
        async for msg in send_job(job, timeout=ANSWER_TIMEOUT):
            if msg["type"] == "sources":
                buf["sources"] = msg.get("sources", [])
                buf["num_sources"] = msg.get("num_sources", len(buf["sources"]))
            elif msg["type"] == "chunk":
                buf["text"] += msg.get("text", "")
                buf["num_sources"] = msg.get("num_sources", 0)
            elif msg["type"] == "error":
                buf["error"] = msg.get("message", "Okänt fel")
                buf["done"] = True
            elif msg["type"] == "done":
                buf["done"] = True
            _notify(buf)
    except vault.VaultError as ex:
        _finish(buf, error=str(ex))
    except Exception as ex:
        log.warning("Question failed: %s", type(ex).__name__)
        _finish(buf, error="Internt fel.")
    finally:
        if not buf["done"]:
            _finish(buf, error="Svaret avbröts.")
        # Drop the buffer after a grace period if nobody is following it.
        asyncio.get_running_loop().call_later(JOB_BUFFER_TTL, active_jobs.pop, job_id, None)


async def follow_job(job_id: str):
    """Yield updates from a buffered job (works for both new and resumed jobs)."""
    buf = active_jobs.get(job_id)
    if not buf:
        return

    last_len = 0
    sources_sent = False
    event = asyncio.Event()
    buf["events"].add(event)

    try:
        while True:
            if buf["sources"] and not sources_sent:
                sources_sent = True
                yield {"type": "sources", "sources": buf["sources"]}
            current_text = buf["text"]
            if len(current_text) > last_len:
                yield {"type": "chunk", "text": current_text, "num_sources": buf["num_sources"]}
                last_len = len(current_text)

            if buf["done"]:
                if buf["error"]:
                    yield {"type": "error", "message": buf["error"]}
                else:
                    yield {"type": "done", "note": buf.get("note", "")}
                break

            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=ANSWER_TIMEOUT + 30)
            except asyncio.TimeoutError:
                yield {"type": "error", "message": "Timeout — inget svar från workern."}
                break
    finally:
        buf["events"].discard(event)
        if buf["done"]:
            active_jobs.pop(job_id, None)


# --- Uploads ------------------------------------------------------------------


class UploadError(Exception):
    pass


_HTML_START = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:<!--.*?-->\s*)*<(?:!doctype\s+html|html|head|body|meta)\b", re.I | re.S)


def _sniff_kind(data: bytes) -> str | None:
    """Cheap content check (the name is not trusted); the worker validates the file properly."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if _HTML_START.match(data[:4096]):
        return "html"
    if data[:4] == b"PK\x03\x04":
        entries = _zip_entry_count(data)
        if entries is None:
            return None
        if entries > MAX_ZIP_ENTRIES:
            return "zip"  # refused by _zip_plan without parsing the directory
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return "docx" if "word/document.xml" in z.namelist() else "zip"
        except zipfile.BadZipFile:
            return None
    return None


def _zip_entry_count(data: bytes) -> int | None:
    """Entry count from the end-of-central-directory record, read before
    zipfile parses the directory (millions of entries would cost gigabytes).
    None if there is no such record. ZIP64 archives report 0xFFFF and are
    treated as too many, which only matters above 65534 entries. A directory
    larger than the limit allows also counts as too many."""
    tail_start = max(0, len(data) - 65_557)  # 22-byte record + comment of at most 65535 bytes
    pos = bytes(data[tail_start:]).rfind(b"PK\x05\x06")
    if pos < 0:
        return None
    pos += tail_start
    if pos + 22 > len(data):
        return None
    entries = int.from_bytes(data[pos + 10 : pos + 12], "little")
    # zipfile parses the directory by its size, not by the count, so a forged
    # small count with a huge directory must be caught too.
    directory_size = int.from_bytes(data[pos + 12 : pos + 16], "little")
    if directory_size > MAX_ZIP_ENTRIES * 800:
        return MAX_ZIP_ENTRIES + 1
    return entries


def _clean_filename(raw: str) -> str:
    name = unquote(raw).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()[:200]
    return name or "dokument"


# Upload states, in order. "queued" and "uploading" happen in the browser (which
# reports them with emitEvent); the rest on the server.
ACTIVE_STATES = ("queued", "uploading", "waiting", "processing", "saving")
FINAL_STATES = ("done", "error", "cancelled")
MAX_ACTIVE_UPLOADS = 5000  # per session, files queued or in progress
MAX_BUFFERED_BYTES = 4 * MAX_UPLOAD_BYTES + int(os.getenv("MAX_ZIP_MB", "500")) * 1024 * 1024  # per session, received, not yet processed


def _new_upload(s: Session, upload_id: str, name: str, size: int) -> dict | None:
    """Register an upload announced by the browser. None if the id is bad or the queue is full."""
    try:
        uuid.UUID(upload_id)
    except (ValueError, TypeError):
        return None
    if upload_id in s.uploads:
        return s.uploads[upload_id]
    if sum(u["state"] in ACTIVE_STATES for u in s.uploads.values()) >= MAX_ACTIVE_UPLOADS:
        return None
    entry = {
        "name": _clean_filename(name),
        "size": int(size or 0),
        "state": "queued",
        "loaded": 0,  # bytes sent by the browser
        "stage": "",  # worker stage: "extract" or "embed"
        "done": 0,
        "total": 0,
        "message": "",
        "job_id": None,
        "task": None,
        "updated": time.monotonic(),
        "kind": "file",  # or "zip", decided from the content once received
        # ZIP only: members to process, processed so far, outcomes, member in progress
        "files": 0,
        "processed": 0,
        "ok": 0,
        "dupes": 0,
        "skipped": 0,
        "failures": [],
        "current": "",
    }
    s.uploads[upload_id] = entry
    return entry


def _set_upload(entry: dict, **fields) -> None:
    entry.update(fields, updated=time.monotonic())


class Duplicate(UploadError):
    pass


async def _ingest_file(s: Session, entry: dict, name: str, data: bytes, known: set[str]) -> None:
    """One document: worker extraction and embedding, then encrypt and store.
    `known` holds the sha256 of every document already in the library."""
    sha = hashlib.sha256(data).hexdigest()
    if sha in known:
        raise Duplicate("Dokumentet finns redan i biblioteket.")
    entry["job_id"] = job_id = str(uuid.uuid4())
    job = {"type": "ingest", "id": job_id, "filename": name, "data_base64": base64.b64encode(data).decode()}
    msg: dict = {"type": "error", "message": "Inget svar från workern."}
    async for msg in send_job(job, timeout=INGEST_TIMEOUT):
        if msg["type"] == "progress":
            _set_upload(entry, stage=msg.get("stage", ""), done=msg.get("done", 0), total=msg.get("total", 0))
    entry["job_id"] = None
    if msg["type"] != "ingested":
        raise UploadError(msg.get("message", "Bearbetningen misslyckades."))
    record = msg["doc"]
    record["filename"] = name
    if record.get("sha256") != sha:
        raise UploadError("Bearbetningen misslyckades.")
    known.add(sha)
    await asyncio.shield(_save_document(s, record))


# ZIP archives: unpacked in memory, one member at a time, only when it is its turn.
MAX_ZIP_BYTES = int(os.getenv("MAX_ZIP_MB", "500")) * 1024 * 1024
MAX_ZIP_FILES = 5000
MAX_ZIP_ENTRIES = 10000  # files plus directories and junk, checked before parsing
_ZIP_JUNK = re.compile(r"(^|/)(__MACOSX/|\.)|(^|/)(Thumbs\.db|desktop\.ini)$", re.I)


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    """Display name: base name, with Swedish characters repaired for archives
    made on Windows (names not flagged UTF-8 are stored in an OEM code page)."""
    name = info.filename
    if not info.flag_bits & 0x800:
        raw = name.encode("cp437", errors="replace")
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError:
            name = raw.decode("cp850", errors="replace")
    return _clean_filename(name.rsplit("/", 1)[-1].replace("%", "%25"))


def _zip_plan(data: bytes) -> tuple[list[zipfile.ZipInfo], int]:
    """Members worth reading (files, not junk, not encrypted), and how many were skipped."""
    entries = _zip_entry_count(data)
    if entries is None or entries > MAX_ZIP_ENTRIES:
        raise UploadError(f"Arkivet innehåller för många filer (högst {MAX_ZIP_FILES}).")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
    members, skipped = [], 0
    for info in infos:
        if info.is_dir():
            continue
        if _ZIP_JUNK.search(info.filename):
            skipped += 1
            continue
        members.append(info)
    if len(members) > MAX_ZIP_FILES:
        raise UploadError(f"Arkivet innehåller fler än {MAX_ZIP_FILES} filer.")
    return members, skipped


def _zip_read(data: bytes, info: zipfile.ZipInfo) -> bytes:
    """One member, never more than MAX_UPLOAD_BYTES however the archive is built."""
    if info.flag_bits & 0x1:
        raise UploadError("Filen är lösenordsskyddad i arkivet.")
    if info.file_size > MAX_UPLOAD_BYTES:
        raise UploadError("Filen är för stor.")
    with zipfile.ZipFile(io.BytesIO(data)) as z, z.open(info) as f:
        out = f.read(MAX_UPLOAD_BYTES + 1)
    if len(out) > MAX_UPLOAD_BYTES:
        raise UploadError("Filen är för stor.")
    return out


async def _ingest_zip(s: Session, entry: dict, data: bytes, known: set[str]) -> None:
    try:
        members, skipped = await run.io_bound(_zip_plan, data)
    except zipfile.BadZipFile:
        raise UploadError("Arkivet är trasigt eller inte en ZIP-fil.") from None
    _set_upload(entry, files=len(members), skipped=skipped)
    for info in members:
        name = _zip_member_name(info)
        _set_upload(entry, current=name, stage="", done=0, total=0)
        try:
            member = await run.io_bound(_zip_read, data, info)
            if _sniff_kind(member) not in ("pdf", "docx", "html"):
                entry["skipped"] += 1  # images, spreadsheets, nested archives, ...
            else:
                await _ingest_file(s, entry, name, member, known)
                entry["ok"] += 1
        except Duplicate:
            entry["dupes"] += 1
        except (UploadError, vault.VaultError) as ex:
            entry["failures"].append((name, str(ex)))
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, ValueError, OSError, EOFError):  # corrupt or unsupported member
            entry["failures"].append((name, "Filen kunde inte packas upp."))
        _set_upload(entry, processed=entry["processed"] + 1, current="")


async def ingest_upload(s: Session, entry: dict, data: bytes) -> None:
    """Ingest one uploaded file or ZIP archive. One upload at a time per session, in upload order."""
    try:
        async with s.ingest_lock:
            if entry["state"] == "cancelled":
                return
            _set_upload(entry, state="processing", stage="", done=0, total=0)
            known = {d.sha256 for d in (await ensure_index(s)).docs.values()}
            if entry["kind"] == "zip":
                await _ingest_zip(s, entry, data, known)
                if entry["ok"] == 0 and entry["failures"] and not entry["dupes"]:
                    _set_upload(entry, state="error", message="Inga dokument kunde läsas in.")
                else:
                    _set_upload(entry, state="done", message="")
                return
            await _ingest_file(s, entry, entry["name"], data, known)
            _set_upload(entry, state="done", message="")
    except asyncio.CancelledError:
        if entry.get("job_id"):
            asyncio.get_running_loop().create_task(_cancel_worker_job(entry["job_id"]))
        _set_upload(entry, state="cancelled")
    except (UploadError, vault.VaultError) as ex:
        _set_upload(entry, state="error", message=str(ex))
    except Exception as ex:
        log.warning("Upload failed: %s", type(ex).__name__)
        _set_upload(entry, state="error", message="Uppladdningen misslyckades.")
    finally:
        entry["task"] = None


async def _cancel_worker_job(job_id: str) -> None:
    if worker_ws:
        try:
            await worker_ws.send_text(json.dumps({"type": "cancel", "id": job_id}))
        except Exception:
            pass


async def cancel_upload(s: Session, upload_id: str) -> None:
    """Stop an upload wherever it is: in the browser, waiting, or on the worker."""
    entry = s.uploads.get(upload_id)
    if not entry or entry["state"] not in ACTIVE_STATES or entry["state"] == "saving":
        return
    if entry["state"] in ("queued", "uploading"):
        ui.run_javascript(f"window.vrCancelUpload && window.vrCancelUpload({json.dumps(upload_id)})")
    if entry.get("task"):
        entry["task"].cancel()  # ingest_upload marks it cancelled and stops the worker job
    else:
        _set_upload(entry, state="cancelled")


def _reject(entry: dict, status: int, message: str) -> JSONResponse:
    """Refuse an upload body and record why, so its row does not wait forever."""
    if entry["state"] not in FINAL_STATES:
        _set_upload(entry, state="error", message=message)
    return JSONResponse({"error": message}, status_code=status)


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin", "")
    return bool(origin) and urlparse(origin).netloc == request.headers.get("host", "")


@app.post("/api/upload")
async def upload_endpoint(request: Request):
    """Raw-body upload, read into memory only. (ui.upload and multipart parsing
    spool large files to plaintext temp files, so they are not used.)"""
    s = get_session(request.session.get("id"))
    if not s or s.stage != "ready" or s.key is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    # CSRF: the session cookie is SameSite=Strict; also require same origin and a custom header.
    if request.headers.get("x-vr-upload") != "1" or not _same_origin(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    entry = _new_upload(s, request.headers.get("x-upload-id", ""), request.headers.get("x-filename", ""), 0)
    if entry is None:
        return JSONResponse({"error": "busy"}, status_code=429)
    if entry["state"] == "cancelled":
        return JSONResponse({"error": "cancelled"}, status_code=409)
    try:
        size = int(request.headers.get("content-length") or 0)
    except ValueError:
        return JSONResponse({"error": "bad request"}, status_code=400)
    limit = MAX_ZIP_BYTES if entry["name"].lower().endswith(".zip") else MAX_UPLOAD_BYTES
    if size > limit:
        return _reject(entry, 413, _UPLOAD_ERRORS[413])
    # Memory: one body read at a time per session, and received files waiting
    # for processing are capped. Counted in real bytes: the header may be absent.
    # Busy: the browser waits and sends the same file again (backpressure while
    # the worker catches up), so a large selection is never rejected for speed.
    buffered = sum(u["size"] for u in s.uploads.values() if u["state"] in ("waiting", "processing"))
    if s.receiving or buffered + size > MAX_BUFFERED_BYTES:
        if entry["state"] == "uploading":
            _set_upload(entry, state="queued", loaded=0)
        return JSONResponse({"error": "busy", "retry": True}, status_code=429)

    s.receiving = True
    buf = bytearray()
    try:
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > limit:
                return _reject(entry, 413, _UPLOAD_ERRORS[413])
            if buffered + len(buf) > MAX_BUFFERED_BYTES:  # the declared size was absent or wrong
                return _reject(entry, 413, _UPLOAD_ERRORS[413])
    except ClientDisconnect:  # the browser aborted (cancel, navigation, network)
        if entry["state"] != "cancelled":
            _set_upload(entry, state="error", message="Överföringen avbröts.")
        return Response(status_code=499)
    finally:
        s.receiving = False
    if entry["state"] == "cancelled":
        return JSONResponse({"error": "cancelled"}, status_code=409)
    kind = _sniff_kind(buf)
    if not kind:
        return _reject(entry, 415, _UPLOAD_ERRORS[415])
    if kind != "zip" and len(buf) > MAX_UPLOAD_BYTES:
        return _reject(entry, 413, _UPLOAD_ERRORS[413])
    # A ZIP stays a bytearray (no second copy of up to MAX_ZIP_MB); its members are read out as bytes.
    data = buf if kind == "zip" else bytes(buf)
    del buf

    s.touch()
    _set_upload(entry, state="waiting", size=len(data), loaded=len(data), kind="zip" if kind == "zip" else "file")
    entry["task"] = asyncio.create_task(ingest_upload(s, entry, data))
    return JSONResponse({"ok": True})


# --- HTTP hardening -----------------------------------------------------------

_CSP = "; ".join(
    [
        "default-src 'self'",
        # NiceGUI/Vue compile templates at runtime and inject inline scripts.
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob: https://www.vr.se",
        "font-src 'self' data:",
        "connect-src 'self'",
        "frame-src 'self'",
        "object-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    h = response.headers
    h.setdefault("Content-Security-Policy", _CSP)
    h["X-Frame-Options"] = "DENY"
    h["X-Content-Type-Options"] = "nosniff"
    h["Referrer-Policy"] = "no-referrer"
    h["Cross-Origin-Opener-Policy"] = "same-origin"
    h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    if COOKIE_SECURE:
        h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path in STAGE_PAGES.values() or request.url.path in ("/login", "/login/complete", "/documents"):
        h["Cache-Control"] = "no-store"
    return response


# --- UI ---

VR_LOGO_DARK_URL = "https://www.vr.se/images/18.4671cb4d18c80cb5f4324cb/1703058189428/logotyp_vetenskapsr%C3%A5det_liggande_sv.svg"
VR_LOGO_LIGHT_URL = "https://www.vr.se/images/18.4671cb4d18c80cb5f4324c9/1703058125493/logotyp_vetenskapsr%C3%A5det_liggande_sv_vit.svg"

VR_STYLE = """
:root {
    --vr-fg: #000000;
    --vr-bg: #ffffff;
    --vr-bg-subtle: #f5f5f5;
    --vr-border: #e0e0e0;
    --vr-text: #333333;
    --vr-text-muted: rgba(0,0,0,0.65);
    --vr-border-strong: #9e9e9e;
    --vr-quote: #555555;
}

@media (prefers-color-scheme: dark) {
    :root {
        --vr-fg: #ffffff;
        --vr-bg: #1a1a1a;
        --vr-bg-subtle: #2a2a2a;
        --vr-border: #444444;
        --vr-text: #e0e0e0;
        --vr-text-muted: rgba(255,255,255,0.7);
        --vr-border-strong: #8a8a8a;
        --vr-quote: #aaaaaa;
    }
}

body.body--dark {
    --vr-fg: #ffffff;
    --vr-bg: #1a1a1a;
    --vr-bg-subtle: #2a2a2a;
    --vr-border: #444444;
    --vr-text: #e0e0e0;
    --vr-text-muted: rgba(255,255,255,0.7);
    --vr-border-strong: #8a8a8a;
    --vr-quote: #aaaaaa;
}

body.body--dark .q-btn--outline,
body.body--dark .q-btn--outline .q-btn__content,
body.body--dark .q-btn--outline .q-btn__content *,
body.body--dark .q-btn--outline .q-icon {
    color: #ffffff !important;
}

body.body--dark .q-btn--outline.q-btn--outline {
    border-color: #ffffff !important;
    border: 1px solid #ffffff !important;
}

body.body--dark .q-spinner,
body.body--dark .q-btn--flat .q-icon {
    color: #ffffff !important;
}

body.body--dark .q-card {
    background-color: var(--vr-bg) !important;
}

body.body--dark .q-page {
    background-color: var(--vr-bg) !important;
}


body, .nicegui-content {
    font-family: 'Open Sans', Arial, sans-serif !important;
    background-color: var(--vr-bg) !important;
    color: var(--vr-text) !important;
}

.q-header {
    background-color: var(--vr-bg) !important;
}

.q-page {
    background-color: var(--vr-bg) !important;
}

.vr-btn {
    font-family: 'Open Sans', Arial, sans-serif !important;
    text-transform: none !important;
    font-weight: 600 !important;
    border-radius: 2px !important;
}


.vr-card {
    border: 1px solid var(--vr-border);
    border-radius: 2px;
    box-shadow: none !important;
    background-color: var(--vr-bg) !important;
    overflow: visible !important;
}

.vr-login-card {
    border-top: 4px solid var(--vr-fg);
}

.q-field--outlined .q-field__control:before {
    border-color: var(--vr-border) !important;
}

.q-field--focused .q-field__control:before {
    border-color: var(--vr-fg) !important;
}

.q-field--focused .q-field__control:after {
    border-bottom-color: var(--vr-fg) !important;
    background-color: var(--vr-fg) !important;
}

.q-field--focused .q-field__label {
    color: var(--vr-fg) !important;
}

.q-field__label {
    color: var(--vr-text-muted) !important;
}

.q-field__native, .q-field__input {
    color: var(--vr-text) !important;
}

.q-spinner {
    color: var(--vr-fg) !important;
}


/* Always dark: Quasar's layered text-white on the message can't be overridden. */
.q-notification {
    background-color: #000000 !important;
    border: 1px solid var(--vr-border-strong);
}

.vr-icon-btn {
    color: var(--vr-fg) !important;
}

.vr-model-select .q-field__native,
.vr-model-select .q-field__input,
.vr-model-select .q-field__append {
    color: black !important;
}

body.body--dark .vr-model-select .q-field__native,
body.body--dark .vr-model-select .q-field__input,
body.body--dark .vr-model-select .q-field__append {
    color: white !important;
}

.vr-model-select .q-field__control {
    background: transparent !important;
}

.vr-model-select .q-field__control:before {
    border-color: black !important;
}

.vr-status {
    color: var(--vr-text-muted) !important;
}

.vr-result {
    font-family: 'Open Sans', Arial, sans-serif;
    line-height: 1.7;
    color: var(--vr-text);
    max-width: 72ch;
}
.vr-result a { color: var(--vr-fg); text-decoration-thickness: 1px; text-underline-offset: 2px; }
.vr-link { color: var(--vr-fg) !important; }

/* Conversation */
.vr-conversation {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.vr-empty {
    padding-top: 18vh;
    text-align: center;
}
.vr-greeting {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--vr-fg);
    line-height: 1.3;
}
.vr-greeting-sub {
    color: var(--vr-text-muted);
    max-width: 48ch;
    line-height: 1.5;
}
.vr-example {
    border-radius: 999px !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    padding: 0.35rem 1.1rem !important;
}
.vr-user-msg {
    background-color: var(--vr-bg-subtle);
    border-radius: 14px 14px 2px 14px;
    padding: 0.7rem 1rem;
    max-width: 85%;
    white-space: pre-wrap;
    line-height: 1.5;
    color: var(--vr-fg);
}

/* Composer */
.q-footer {
    background-color: var(--vr-bg) !important;
    padding: 0.5rem 0 0.75rem;
}
.vr-composer {
    border: 1px solid var(--vr-border-strong);
    border-radius: 14px;
    padding: 0.5rem 0.75rem 0.4rem;
    background-color: var(--vr-bg);
}
.vr-composer:focus-within {
    border-color: var(--vr-fg);
}
.vr-composer-input textarea::placeholder {
    color: var(--vr-text-muted);
    opacity: 1;
}
.vr-composer-input textarea {
    max-height: 40vh;
    font-size: 1rem;
    line-height: 1.5;
}
.vr-composer .q-field--dense .q-field__control { min-height: 32px; }
/* Send button colours are set inline; Quasar's bg-primary wins over class rules */
/* The global focus rule fills :after with the foreground colour, which covers
   a whole borderless control (textarea, model select) in black */
.vr-composer .q-field__control:before,
.vr-composer .q-field__control:after {
    display: none !important;
}
.vr-send.disabled, .vr-send[disabled] {
    opacity: 0.3 !important;
}
/* Selected model in the dropdown: brand colour instead of Quasar blue */
.q-menu .q-item.q-item--active {
    color: var(--vr-fg) !important;
    font-weight: 600;
}
body.body--dark .q-menu .q-item.q-item--active {
    color: #ffffff !important;
}

.vr-disclaimer {
    font-size: 0.78rem;
    color: var(--vr-text-muted);
    text-align: center;
    width: 100%;
    margin-top: 0.4rem;
}

.vr-result h1, .vr-result h2, .vr-result h3 {
    color: var(--vr-fg);
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}

.vr-result p {
    margin-bottom: 0.8rem;
}

.vr-result ul, .vr-result ol {
    padding-left: 1.5rem;
    margin-bottom: 0.8rem;
}

.vr-result code {
    background-color: var(--vr-border);
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    font-size: 0.9em;
}

.vr-result blockquote {
    border-left: 3px solid var(--vr-border);
    padding-left: 1rem;
    color: var(--vr-quote);
    margin: 0.8rem 0;
}

.vr-progress {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.25rem 0;
}

.vr-progress-text {
    font-family: 'Open Sans', Arial, sans-serif;
    font-size: 0.85rem;
    color: var(--vr-text);
}

/* Dark mode logo switching */
.vr-logo-dark { display: block; }
.vr-logo-light { display: none; }
@media (prefers-color-scheme: dark) {
    .vr-logo-dark { display: none; }
    .vr-logo-light { display: block; }
}
body.body--dark .vr-logo-dark { display: none; }
body.body--dark .vr-logo-light { display: block; }

/* Plain buttons get color=None: Quasar's .text-* colour classes (including
   the default "primary") are !important inside a cascade layer, which beats
   any unlayered !important rule, so they can't be themed from here. */
.vr-plain, .vr-plain .q-icon { color: var(--vr-fg) !important; }
.vr-nav { opacity: 0.7; }
.vr-nav-active { opacity: 1; text-decoration: underline; text-underline-offset: 6px; }
.vr-bar { color: var(--vr-fg); }
.vr-bar .q-linear-progress__track { opacity: 0.15; }
.vr-upload-name { color: var(--vr-fg); font-size: 0.9rem; }
.vr-upload-detail { color: var(--vr-text-muted); white-space: normal; word-break: break-word; }
.vr-upload-status { color: var(--vr-text-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.vr-danger { border: 1px solid #c10015; border-radius: 2px; padding: 1rem 1.25rem; }
.vr-source-name { color: var(--vr-fg); word-break: break-word; }
.vr-doc-row { padding: 0.6rem 0; border-bottom: 1px solid var(--vr-border); }
"""


def _page_setup() -> ui.dark_mode:
    dark = ui.dark_mode()
    dark.auto()
    ui.add_head_html(f"<style>{VR_STYLE}</style>")
    return dark


def _logout() -> None:
    end_session()
    ui.navigate.to("/login")


def _logo() -> None:
    ui.image(VR_LOGO_DARK_URL).classes("w-48 vr-logo-dark").on("click", lambda: ui.navigate.to("/")).style("cursor: pointer;")
    ui.image(VR_LOGO_LIGHT_URL).classes("w-48 vr-logo-light").on("click", lambda: ui.navigate.to("/")).style("cursor: pointer;")


def _session_watch() -> None:
    """Send the browser to the login page once the session has expired."""
    def check():
        if get_session() is None:
            ui.navigate.to("/login")
    ui.timer(30.0, check)


def _auth_card(title: str, subtitle: str = ""):
    """Centered card used by the login, password and unlock pages."""
    col = ui.column().classes("absolute-center items-center")
    with col:
        card = ui.card().classes("w-96 vr-card vr-login-card")
        with card:
            ui.label(title).classes("text-h5 font-bold q-mb-sm")
            if subtitle:
                ui.label(subtitle).classes("vr-greeting-sub q-mb-sm").style("text-align: left;")
    return card


def _secret_input(label: str, autocomplete: str) -> ui.input:
    field_ = ui.input(label=label, password=True, password_toggle_button=True).classes("w-full")
    field_.props(f'autocomplete="{autocomplete}"')
    return field_


def _submit_button(text: str, handler) -> ui.button:
    return ui.button(text, on_click=handler).classes("w-full q-mt-md vr-btn").props('outline no-caps color="black"')


async def _busy(button: ui.button, coro):
    """Run a slow step (Argon2id) with the button disabled."""
    button.disable()
    try:
        return await coro
    finally:
        button.enable()


def _open_account(s: Session, user, password: str) -> None:
    """After a correct password: unlock the key with it (creating the key pair
    on first use), or prepare the one-time move of a legacy passphrase account.
    Sets the session's stage. Runs Argon2id: call from a thread."""
    if user["must_change_pw"]:
        s.stage = "password"
    elif not vault.has_keys(user):
        s.key = vault.create_keys(user["id"], password)
        s.stage = "ready"
    elif vault.password_is_key(user):
        s.key = vault.unlock(user["id"], password)
        if s.key is None:
            raise vault.VaultError("Kontots nyckel kunde inte låsas upp.")
        s.stage = "ready"
    else:
        s.migration = vault.password_kek(password)
        s.stage = "unlock"
    s.auth_fp = auth_fingerprint(vault.get_user(user["id"]))


@ui.page("/login")
def login_page(request: Request):
    _page_setup()
    s = get_session()
    if s is not None:
        return RedirectResponse(STAGE_PAGES[s.stage])

    with _auth_card("Logga in"):
        if request.query_params.get("raderad"):
            ui.label("Kontot och all data är raderade.").classes("q-mb-sm").style("font-weight: 600;")
        # Browsers drop Secure cookies on plain http (Safari even on localhost), so
        # login would silently bounce back here. Say so instead.
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        if COOKIE_SECURE and not https:
            ui.label(
                "Anslutningen är inte https, så webbläsaren sparar inte inloggningen. Använd https, "
                "eller sätt COOKIE_SECURE=0 i .env vid lokal utveckling utan TLS."
            ).classes("text-red q-mb-sm")
        username = ui.input(label="Användarnamn").classes("w-full").props('autocomplete="username" autofocus')
        password = _secret_input("Lösenord", "current-password")
        error = ui.label("").classes("text-red q-mt-sm")

        async def do_login():
            name = (username.value or "").strip().lower()
            ip_key, user_key = f"ip:{_client_ip(request)}", f"u:{name}"
            if throttle.locked(ip_key) or throttle.locked(user_key):
                error.set_text("För många misslyckade försök. Försök igen om en stund.")
                return
            secret = password.value or ""
            password.set_value("")
            user = await _busy(button, run.io_bound(vault.verify_login, name, secret))
            if not user:
                throttle.fail(ip_key, Throttle.IP_LIMIT)
                throttle.fail(user_key)
                error.set_text("Fel användarnamn eller lösenord.")
                return
            throttle.reset(user_key)
            if throttle.locked(f"k:{user['id']}"):
                error.set_text("Kontot är tillfälligt spärrat efter för många felaktiga lösenfraser.")
                return
            session = Session(user_id=user["id"], username=user["username"], stage="password")
            try:
                await _busy(button, run.io_bound(_open_account, session, user, secret))
            except vault.VaultError as ex:
                error.set_text(str(ex))
                return
            finally:
                secret = ""
            end_session()
            # Hand over through /login/complete, which issues a fresh session id
            # (a session id planted before login must not become a logged-in one).
            token = secrets.token_urlsafe(32)
            PENDING_LOGINS[token] = (session, time.monotonic() + 60, _browser_id())
            ui.navigate.to(f"/login/complete?t={token}")

        username.on("keydown.enter", lambda: password.run_method("focus"))
        password.on("keydown.enter", do_login)
        button = _submit_button("Logga in", do_login)


@ui.page("/password")
def password_page():
    s, redirect = guard("password")
    if redirect:
        return redirect
    _page_setup()
    _session_watch()
    with _auth_card(
        "Välj lösenord",
        f"Kontot {s.username} har ett tillfälligt lösenord. Välj ett eget på minst {vault.MIN_PASSWORD_LEN} tecken. "
        "Det skyddar också nyckeln som krypterar dina dokument: glömmer du det kan dokumenten inte räddas.",
    ):
        new = _secret_input("Nytt lösenord", "new-password")
        again = _secret_input("Upprepa lösenordet", "new-password")
        error = ui.label("").classes("text-red q-mt-sm")

        async def do_change():
            secret = new.value or ""
            try:
                if secret != (again.value or ""):
                    raise vault.VaultError("Lösenorden är inte lika.")
                vault.check_password_policy(secret)

                def apply():
                    vault.set_password(s.user_id, secret)
                    _open_account(s, vault.get_user(s.user_id), secret)

                await _busy(button, own_auth_change(s, apply))
            except vault.VaultError as ex:
                error.set_text(str(ex))
                return
            finally:
                new.set_value("")
                again.set_value("")
                secret = ""
            ui.navigate.to(STAGE_PAGES[s.stage])

        again.on("keydown.enter", do_change)
        button = _submit_button("Spara lösenord", do_change)
        ui.button("Logga ut", on_click=_logout, color=None).props("flat no-caps").classes("w-full vr-btn vr-plain")


@ui.page("/unlock")
def unlock_page():
    """Accounts from before the password became the key: the old passphrase once more."""
    s, redirect = guard("unlock")
    if redirect:
        return redirect
    if s.migration is None:
        end_session()
        return RedirectResponse("/login")
    _page_setup()
    _session_watch()
    with _auth_card(
        "En sista gång",
        f"Kontot {s.username} skapades med en separat lösenfras. Ange den en sista gång; "
        "därefter låser ditt lösenord upp dokumenten och lösenfrasen behövs inte mer. "
        "Observera: lösenfrasen, inte lösenordet. Kontrollera att webbläsaren inte har fyllt i lösenordet.",
    ):
        # Not "current-password": browsers would autofill the login password here.
        phrase = _secret_input("Lösenfras (inte lösenordet)", "off").props('autofocus name="legacy-passphrase"')
        error = ui.label("").classes("text-red q-mt-sm")

        async def do_unlock():
            key_throttle = f"k:{s.user_id}"
            if throttle.locked(key_throttle):
                end_session()
                ui.navigate.to("/login")
                return
            key = await _busy(button, run.io_bound(vault.unlock, s.user_id, phrase.value or ""))
            phrase.set_value("")
            if key is None:
                throttle.fail(key_throttle)
                if throttle.locked(key_throttle):
                    end_session()
                    ui.navigate.to("/login")
                    return
                left = Throttle.MAX_FAILS - len(throttle.fails.get(key_throttle, []))
                error.set_text(f"Fel lösenfras. {left} försök kvar innan kontot spärras i 15 minuter.")
                return
            throttle.reset(key_throttle)
            kek, salt = s.migration
            await own_auth_change(s, vault.protect_key_with_kek, s.user_id, key, kek, salt)
            s.migration = None
            s.key = key
            s.stage = "ready"
            ui.navigate.to("/")

        phrase.on("keydown.enter", do_unlock)
        button = _submit_button("Lås upp", do_unlock)
        ui.button("Logga ut", on_click=_logout, color=None).props("flat no-caps").classes("w-full vr-btn vr-plain")


# Upload, browser side: a hidden file input queues the chosen files and sends
# them one at a time as raw bodies with XMLHttpRequest (fetch cannot report
# upload progress). Every step is reported to the server with emitEvent; the
# server keeps the state and renders it (UploadPanel).
_UPLOAD_JS = """
<input type="file" id="vr-file" multiple style="display:none"
       accept=".pdf,.docx,.htm,.html,.zip,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/html,application/zip">
<script>
(function () {
    const MAX = %(max)d, MAX_ZIP = %(max_zip)d;
    const queue = [];
    let active = null;

    function pump() {
        if (active || !queue.length) return;
        const item = queue.shift();
        const xhr = new XMLHttpRequest();
        active = item;
        item.xhr = xhr;
        let last = 0;
        xhr.open('POST', '/api/upload');
        xhr.setRequestHeader('X-VR-Upload', '1');
        xhr.setRequestHeader('X-Upload-Id', item.id);
        xhr.setRequestHeader('X-Filename', encodeURIComponent(item.file.name));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');
        xhr.upload.onprogress = function (e) {
            const now = Date.now();
            if (now - last > 200 || e.loaded === e.total) {
                last = now;
                emitEvent('vr_upload_progress', {id: item.id, loaded: e.loaded});
            }
        };
        xhr.onloadend = function () {
            active = null;
            let retry = false;
            if (xhr.status === 429) { try { retry = JSON.parse(xhr.responseText).retry === true; } catch (err) {} }
            if (retry && !item.cancelled) {  // server busy: same file again shortly, order kept
                queue.unshift(item);
                setTimeout(pump, 2000);
                return;
            }
            emitEvent('vr_upload_done', {id: item.id, status: xhr.status});
            pump();
        };
        xhr.send(item.file);
    }

    document.getElementById('vr-file').addEventListener('change', function (e) {
        for (const f of Array.from(e.target.files)) {
            const id = crypto.randomUUID();
            emitEvent('vr_upload_queued', {id: id, name: f.name, size: f.size});
            const zip = /\\.zip$/i.test(f.name);
            if (f.size <= (zip ? MAX_ZIP : MAX) && /\\.(pdf|docx|html?|zip)$/i.test(f.name)) queue.push({id: id, file: f});
        }
        e.target.value = '';
        pump();
    });

    window.vrCancelUpload = function (id) {
        const i = queue.findIndex(function (q) { return q.id === id; });
        if (i >= 0) queue.splice(i, 1);
        if (active && active.id === id) { active.cancelled = true; active.xhr.abort(); }
    };
    window.vrCancelAllUploads = function () {
        queue.length = 0;
        if (active) { active.cancelled = true; active.xhr.abort(); }
    };

    // Leaving the page aborts transfers still in the browser (processing on
    // the server continues), so ask first.
    window.addEventListener('beforeunload', function (e) {
        if (active || queue.length) { e.preventDefault(); e.returnValue = ''; }
    });
})();
</script>
""" % {"max": MAX_UPLOAD_BYTES, "max_zip": MAX_ZIP_BYTES}

_UPLOAD_ERRORS = {
    401: "Sessionen har gått ut. Logga in igen.",
    403: "Uppladdningen nekades.",
    413: f"Filen är för stor (högst {MAX_UPLOAD_BYTES // 2**20} MB, ZIP-arkiv {MAX_ZIP_BYTES // 2**20} MB).",
    415: "Filen är varken PDF, DOCX, HTML eller ZIP, oavsett vad den heter.",
    429: "För många filer i kö. Vänta tills några är klara.",
}


def _upload_support(s: Session) -> None:
    """Hidden file input plus the handlers for its events."""
    ui.add_body_html(_UPLOAD_JS)

    def queued(e):
        a = e.args
        entry = _new_upload(s, a.get("id", ""), a.get("name", ""), a.get("size", 0))
        if entry is None:
            ui.run_javascript(f"window.vrCancelUpload({json.dumps(a.get('id', ''))})")
            ui.notify(_UPLOAD_ERRORS[429], type="negative")
        elif entry["size"] > (MAX_ZIP_BYTES if entry["name"].lower().endswith(".zip") else MAX_UPLOAD_BYTES):
            _set_upload(entry, state="error", message=_UPLOAD_ERRORS[413])
        elif not re.search(r"\.(pdf|docx|html?|zip)$", entry["name"], re.IGNORECASE):
            _set_upload(entry, state="error", message=_UPLOAD_ERRORS[415])
        s.touch()

    def progress(e):
        entry = s.uploads.get(e.args.get("id", ""))
        if entry and entry["state"] in ("queued", "uploading"):
            _set_upload(entry, state="uploading", loaded=int(e.args.get("loaded") or 0))

    def done(e):
        entry = s.uploads.get(e.args.get("id", ""))
        status = e.args.get("status")
        if status == 401:
            ui.navigate.to("/login")
        if not entry or status == 200 or entry["state"] in FINAL_STATES:
            return
        message = _UPLOAD_ERRORS.get(status, "Överföringen avbröts." if not status else "Uppladdningen misslyckades.")
        _set_upload(entry, state="error", message=message)

    ui.on("vr_upload_queued", queued)
    ui.on("vr_upload_progress", progress)
    ui.on("vr_upload_done", done)


def _stage_status(u: dict) -> tuple[str, float | None]:
    """Worker stage of the file in progress; None means indeterminate."""
    done, total = u["done"], u["total"]
    if u["stage"] == "extract" and total:
        return f"Läser text · sida {done} av {total}", done / total
    if u["stage"] == "extract":
        return "Läser text", None
    if u["stage"] == "embed" and total:
        return f"Analyserar · {done} av {total} avsnitt", done / total
    return "Väntar på workern", None


def _zip_summary(u: dict) -> str:
    parts = [
        f"{u['ok']} inlästa",
        f"{u['dupes']} fanns redan" if u["dupes"] else "",
        f"{len(u['failures'])} misslyckades" if u["failures"] else "",
        f"{u['skipped']} hoppades över (inte PDF/DOCX/HTML)" if u["skipped"] else "",
    ]
    return " · ".join(p for p in parts if p)


def _upload_status(u: dict) -> tuple[str, str, float | None]:
    """Status text, detail line and bar value for an upload; None means indeterminate."""
    state = u["state"]
    if state == "queued":
        return "I kö", "", 0.0
    if state == "uploading":
        frac = u["loaded"] / u["size"] if u["size"] else 0.0
        return f"Laddar upp · {frac:.0%}", "", frac
    if state == "waiting":
        return "Väntar på tur", "", 0.0
    zip_ = u["kind"] == "zip"
    if state == "processing" and zip_:
        if not u["files"]:
            return "Packar upp", "", None
        stage, frac = _stage_status(u) if u["current"] else ("", 0.0)
        overall = (u["processed"] + (frac or 0.0)) / u["files"]
        detail = f"{u['current']} · {stage}" if u["current"] else ""
        return f"Fil {min(u['processed'] + 1, u['files'])} av {u['files']}", detail, overall
    if state == "processing":
        text, frac = _stage_status(u)
        return text, "", frac
    if state == "saving":
        return "Krypterar och sparar", "", None
    failures = ""
    if zip_ and u["failures"]:
        shown = "; ".join(f"{n} ({m})" for n, m in u["failures"][:5])
        more = f" och {len(u['failures']) - 5} till" if len(u["failures"]) > 5 else ""
        failures = f"Misslyckades: {shown}{more}"
    if state == "done":
        return "Klar", " — ".join(x for x in (_zip_summary(u) if zip_ else "", failures) if x), 1.0
    if state == "cancelled":
        detail = f"Avbrutet efter {u['processed']} av {u['files']} filer · {_zip_summary(u)}" if zip_ and u["files"] else ""
        return "Avbruten", detail, 0.0
    return u["message"] or "Misslyckades", failures, 0.0


async def cancel_all_uploads(s: Session) -> None:
    ui.run_javascript("window.vrCancelAllUploads && window.vrCancelAllUploads()")
    for uid in [k for k, u in s.uploads.items() if u["state"] in ACTIVE_STATES]:
        await cancel_upload(s, uid)


class UploadPanel:
    """Upload progress. Up to BATCH_ROWS uploads: one row each. More: a summary
    row for the whole batch, plus rows only for files in progress, ZIP archives
    and errors, so a selection of thousands of files stays readable and cheap."""

    DONE_LINGER = 5  # seconds a finished row stays visible (small batches)
    CANCELLED_LINGER = 3
    BATCH_ROWS = 8
    MAX_ERROR_ROWS = 20
    BUSY_STATES = ("uploading", "processing", "saving")

    def __init__(self, s: Session):
        self.s = s
        self.rows: dict[str, dict] = {}
        self.box = ui.column().classes("w-full gap-3 vr-uploads")
        with self.box:
            with ui.column().classes("w-full gap-1 vr-upload-row vr-upload-summary") as self.summary:
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    ui.icon("upload_file").classes("opacity-60")
                    with ui.column().classes("col gap-0").style("min-width: 0;"):
                        self.sum_title = ui.label("").classes("vr-upload-name")
                        self.sum_detail = ui.label("").classes("text-caption vr-upload-detail")
                    self.sum_button = ui.button("Avbryt alla", on_click=self._summary_action, color=None).props(
                        "flat no-caps size=sm"
                    ).classes("vr-btn vr-plain")
                self.sum_bar = ui.linear_progress(value=0, show_value=False, size="4px", color=None).classes("vr-bar")
            self.summary.set_visibility(False)
        self.sum_key = None
        self.refresh()
        ui.timer(0.25, self.refresh)

    def _add_row(self, uid: str, u: dict) -> dict:
        with self.box:
            with ui.column().classes("w-full gap-1 vr-upload-row") as root:
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    icon = ui.icon("description").classes("opacity-60")
                    with ui.column().classes("col gap-0").style("min-width: 0;"):
                        ui.label(u["name"]).classes("ellipsis vr-upload-name")
                        detail = ui.label("").classes("text-caption vr-upload-detail")
                    status = ui.label("").classes("text-caption vr-upload-status")
                    button = ui.button(icon="close", on_click=partial(self._close, uid), color=None).props(
                        "flat round size=sm"
                    ).classes("vr-plain")
                    with button:
                        tooltip = ui.tooltip("Avbryt")
                bar = ui.linear_progress(value=0, show_value=False, size="4px", color=None).classes("vr-bar")
        return {"root": root, "icon": icon, "status": status, "detail": detail, "button": button, "tooltip": tooltip,
                "bar": bar, "key": None}

    async def _close(self, uid: str):
        u = self.s.uploads.get(uid)
        if u and u["state"] in FINAL_STATES:
            self.s.uploads.pop(uid, None)
            self.refresh()
        else:
            await cancel_upload(self.s, uid)

    async def _summary_action(self):
        if any(u["state"] in ACTIVE_STATES for u in self.s.uploads.values()):
            await cancel_all_uploads(self.s)
        else:  # batch finished: clear it
            self.s.uploads.clear()
            self.refresh()

    def _visible(self, batch: bool) -> list[str]:
        uploads = self.s.uploads
        if not batch:
            return list(uploads)
        errors = [k for k, u in uploads.items() if u["state"] == "error" and u["kind"] != "zip"]
        return [
            k for k, u in uploads.items()
            if u["state"] in self.BUSY_STATES or u["kind"] == "zip"
        ] + errors[-self.MAX_ERROR_ROWS :]

    def _render_summary(self, batch: bool):
        self.summary.set_visibility(batch)
        if not batch:
            return
        counts: dict[str, int] = {}
        for u in self.s.uploads.values():
            counts[u["state"]] = counts.get(u["state"], 0) + 1
        total = len(self.s.uploads)
        done, failed, cancelled = counts.get("done", 0), counts.get("error", 0), counts.get("cancelled", 0)
        finished = done + failed + cancelled
        active = total - finished
        title = f"{finished} av {total} filer klara" if active else f"Klart: {total} filer"
        bits = [f"{done} inlästa", f"{failed} misslyckades" if failed else "", f"{cancelled} avbrutna" if cancelled else ""]
        waiting = counts.get("queued", 0) + counts.get("waiting", 0)
        if waiting:
            bits.append(f"{waiting} i kö")
        hidden = failed - min(failed, self.MAX_ERROR_ROWS)
        if hidden:
            bits.append(f"de {self.MAX_ERROR_ROWS} senaste felen visas")
        key = (title, tuple(bits), active > 0)
        if key == self.sum_key:
            return
        self.sum_key = key
        self.sum_title.set_text(title)
        self.sum_detail.set_text(" · ".join(b for b in bits if b))
        self.sum_bar.set_value(finished / total if total else 0)
        self.sum_bar.set_visibility(active > 0)
        self.sum_button.set_text("Avbryt alla" if active else "Stäng")

    def refresh(self):
        now = time.monotonic()
        uploads = self.s.uploads
        batch = len(uploads) > self.BATCH_ROWS  # a finished batch stays summarized until "Stäng"
        if not batch:
            for uid, u in list(uploads.items()):
                # Rows with something to read (ZIP results, failures) stay until closed.
                sticky = u["kind"] == "zip" and u["state"] in FINAL_STATES
                linger = None if sticky else {"done": self.DONE_LINGER, "cancelled": self.CANCELLED_LINGER}.get(u["state"])
                if linger and now - u["updated"] > linger:
                    uploads.pop(uid, None)
        self._render_summary(batch)
        visible = self._visible(batch)
        shown = set(visible)
        for uid in [k for k in self.rows if k not in shown]:
            self.box.remove(self.rows.pop(uid)["root"])
        for uid in visible:
            u = uploads[uid]
            row = self.rows.get(uid) or self.rows.setdefault(uid, self._add_row(uid, u))
            text, detail, value = _upload_status(u)
            key = (u["state"], text, detail, value)
            if key == row["key"]:
                continue
            row["key"] = key
            state = u["state"]
            row["status"].set_text(text)
            row["detail"].set_text(detail)
            row["detail"].set_visibility(bool(detail))
            row["status"].classes(replace="text-caption vr-upload-status" + (" text-red" if state == "error" else ""))
            row["icon"].name = {"done": "check_circle", "error": "error_outline", "cancelled": "block"}.get(
                state, "folder_zip" if u["kind"] == "zip" or u["name"].lower().endswith(".zip") else "description"
            )
            row["icon"].update()
            row["bar"].set_visibility(state not in ("error", "cancelled"))
            if value is None:
                row["bar"].props("indeterminate")
            else:
                row["bar"].props(remove="indeterminate")
                row["bar"].set_value(value)
            row["button"].set_visibility(state != "saving")
            row["tooltip"].set_text("Stäng" if state in FINAL_STATES else "Avbryt")


def _file_button(button: ui.button) -> ui.button:
    """Open the file picker straight from the click, in the browser: a picker
    opened after a server round-trip is blocked as lacking a user gesture."""
    return button.on("click", js_handler="() => document.getElementById('vr-file').click()")


def _header(s: Session, current: str) -> ui.label:
    with ui.header().classes("items-center justify-between").style("padding: 0.8rem 2rem;"):
        with ui.row().classes("items-center gap-4"):
            _logo()
        with ui.row().classes("items-center gap-2"):
            status_label = ui.label("").classes("text-caption vr-status q-mr-sm")
            for path, text, icon in (("/", "Fråga", "chat"), ("/documents", "Dokument", "folder")):
                btn = ui.button(text, icon=icon, on_click=partial(ui.navigate.to, path), color=None).props("flat no-caps").classes("vr-btn vr-plain vr-nav")
                if path == current:
                    btn.classes("vr-nav-active")
            ui.label(s.username).classes("text-caption vr-status q-ml-sm")
            ui.button(icon="logout", on_click=_logout).props('outline round size=sm color="black"').tooltip("Logga ut och lås")
    return status_label


def _source_bits(src: dict) -> list:
    return [
        b
        for b in (
            src.get("doc_type") if src.get("doc_type") not in ("", "okänd", None) else "",
            src.get("year"),
            f"dnr {src['diarienummer']}" if src.get("diarienummer") else "",
        )
        if b
    ]


@ui.page("/")
async def main_page():
    s, redirect = guard("ready")
    if redirect:
        return redirect
    dark = _page_setup()
    ui.add_body_html(_PLOTLY_THEME_JS)
    _session_watch()
    status_label = _header(s, "/")

    def update_status():
        n = len(s.index.docs) if s.index is not None else None
        worker = "worker ansluten" if worker_ws else "ingen worker ansluten"
        status_label.set_text(f"{n} dokument · {worker}" if n is not None else worker.capitalize())

    ui.timer(2.0, update_status)

    def refresh_models():
        options = model_options()
        if options != model_select.options:
            model_select.set_options(options, value=model_select.value if model_select.value in options else next(iter(options)))

    ui.timer(5.0, refresh_models)

    def _is_dark() -> bool:
        return dark.value is True or (dark.value is None and app.storage.browser.get("dark_mode"))

    async def ask_example(question: str):
        question_input.set_value(question)
        await handle_question()

    # --- Conversation ---
    with ui.column().classes("w-full max-w-3xl mx-auto q-px-md vr-conversation"):
        with ui.column().classes("w-full items-center vr-empty") as empty_state:
            ui.label("Vad vill du veta om dina dokument?").classes("vr-greeting")
            ui.label(
                "Dina dokument lagras krypterade med din egen nyckel. "
                "Ladda upp PDF-, DOCX-, HTML- eller ZIP-filer under Dokument och ställ frågor om dem."
            ).classes("vr-greeting-sub")
            with ui.row().classes("justify-center gap-2 q-mt-md"):
                for example in (
                    "Hur många dokument finns per år?",
                    "Sammanfatta de viktigaste slutsatserna.",
                    "Vilka dokument handlar om finansiering?",
                ):
                    ui.button(example, on_click=partial(ask_example, example)).props(
                        'outline no-caps color="black"'
                    ).classes("vr-btn vr-example")
        conversation = ui.column().classes("w-full gap-8")

    class Turn:
        """One question with its streamed answer, progress and sources."""

        def __init__(self, question_label: str):
            empty_state.set_visibility(False)
            with conversation:
                with ui.column().classes("w-full gap-3 vr-turn"):
                    with ui.row().classes("w-full justify-end"):
                        ui.label(question_label).classes("vr-user-msg")
                    with ui.row().classes("items-center vr-progress") as self.progress:
                        ui.spinner("dots", size="md", color="black")
                        with ui.column().classes("gap-0"):
                            self.step = ui.label("").classes("vr-progress-text").style("font-weight: 600;")
                            self.detail = ui.label("").classes("vr-progress-text")
                    with ui.column().classes("w-full gap-4 vr-result") as self.answer:
                        self.stream = ui.markdown("").classes("w-full")
                        self.container = ui.column().classes("w-full gap-4")
                        self.container.set_visibility(False)
                        self.sources_box = ui.column().classes("w-full gap-1 q-mt-sm vr-sources")
                        self.sources_box.set_visibility(False)
                    self.answer.set_visibility(False)

        def set_progress(self, step: str, detail: str = ""):
            self.step.set_text(step)
            self.detail.set_text(detail)
            self.progress.set_visibility(True)

        def render_sources(self, sources: list[dict], note: str = ""):
            """Always say what the answer rests on: the documents used, as references
            (originals are not stored), or a note when no document is cited."""
            self.sources_box.clear()
            with self.sources_box:
                ui.label("Källor").classes("text-subtitle2").style("font-weight: 600;")
                if not sources:
                    ui.label(note or "Inga dokument i biblioteket användes för svaret.").classes("text-caption vr-status")
                for n, src in enumerate(sources, 1):
                    bits = [str(b) for b in _source_bits(src)]
                    title = (src.get("title") or "").strip()
                    with ui.row().classes("items-baseline gap-2 no-wrap vr-source"):
                        ui.label(f"{n}.").classes("text-caption vr-status")
                        with ui.column().classes("gap-0"):
                            ui.label(src.get("filename") or "okänt dokument").classes("vr-source-name")
                            detail = " · ".join(([title] if title and title.lower() != _doc_stem(src.get("filename") or "").lower() else []) + bits)
                            if detail:
                                ui.label(detail).classes("text-caption vr-status")
            self.sources_box.set_visibility(True)

        def render_stream(self, text: str):
            self.stream.set_content(normalize_bullets(text))
            self.stream.set_visibility(True)
            self.answer.set_visibility(True)

        def render_final(self, text: str, sources: list[dict] | None = None, note: str = ""):
            """Parse text for chart blocks and render mixed markdown + Plotly."""
            sources = sources or []
            self.progress.set_visibility(False)
            self.container.clear()
            segments = [
                ("md", protect_source_names(normalize_bullets(content), sources)) if kind == "md" else (kind, content)
                for kind, content in parse_chart_segments(text)
            ]
            has_charts = any(s[0] == "chart" for s in segments)
            self.render_sources(sources, note)
            self.answer.set_visibility(True)

            if not has_charts:
                self.stream.set_content(segments[0][1] if segments else "")
                self.stream.set_visibility(True)
                self.container.set_visibility(False)
                return

            self.stream.set_visibility(False)
            is_dark = _is_dark()
            with self.container:
                for seg_type, content in segments:
                    if seg_type == "md" and content.strip():
                        ui.markdown(content).classes("w-full")
                    elif seg_type == "chart":
                        try:
                            spec = json.loads(content)
                            fig = build_plotly_figure(spec, dark=is_dark)
                            # No Plotly logo: it links out to plotly.com.
                            ui.plotly({**fig.to_plotly_json(), "config": {"displaylogo": False}}).classes("w-full")
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                            ui.markdown(f"```\n{content}\n```").classes("w-full")
            self.container.set_visibility(True)
            # Re-theme charts after render based on actual browser state
            ui.run_javascript(
                "setTimeout(function(){"
                "var d=document.body.classList.contains('body--dark');"
                "var fg=d?'#ffffff':'#000000';"
                "var bg=d?'#1a1a1a':'#ffffff';"
                "var g=d?'#444444':'#e0e0e0';"
                "document.querySelectorAll('.js-plotly-plot').forEach(function(el){"
                "Plotly.relayout(el,{"
                "'paper_bgcolor':bg,'plot_bgcolor':bg,"
                "'title.font.color':fg,"
                "'xaxis.color':fg,'xaxis.gridcolor':g,'xaxis.linecolor':g,'xaxis.zerolinecolor':g,"
                "'yaxis.color':fg,'yaxis.gridcolor':g,'yaxis.linecolor':g,'yaxis.zerolinecolor':g,"
                "'legend.font.color':fg,'xaxis.title.font.color':fg,'yaxis.title.font.color':fg"
                "});"
                "if(el.data&&el.data[0]){"
                "if(el.data[0].type==='pie')Plotly.restyle(el,{'textfont.color':fg,'outsidetextfont.color':fg});"
                "else if(el.data[0].type==='bar')Plotly.restyle(el,{'textfont.color':fg,'marker.line.color':bg,'marker.line.width':1});"
                "}"
                "});},200);"
            )

    def _scroll_to_bottom():
        """Jump to the newest turn and keep following it while the answer streams."""
        ui.run_javascript("window.vrFollow = true; window.scrollTo(0, document.documentElement.scrollHeight);")

    async def _follow_and_display(job_id: str, turn: Turn):
        """Stream results from a buffered job into the turn."""
        question = active_jobs.get(job_id, {}).get("question", "")
        full_text = ""
        sources: list[dict] = []
        async for msg in follow_job(job_id):
            if msg["type"] == "sources":
                sources = msg["sources"]
                if not full_text:
                    n = len(sources)
                    turn.set_progress(
                        "Modellen läser",
                        f"Utdrag ur {n} dokument. Stora underlag kan ta några minuter." if n else "Förbereder svaret...",
                    )
            elif msg["type"] == "chunk":
                num_sources = msg.get("num_sources", 0)
                turn.set_progress("Skriver svar", f"{num_sources} relevanta dokument hittade")
                full_text = msg["text"]
                turn.render_stream(full_text)
            elif msg["type"] == "error":
                turn.progress.set_visibility(False)
                turn.render_stream(f"{full_text}\n\n**Fel:** {msg['message']}")
            elif msg["type"] == "done":
                turn.render_final(full_text, sources, msg.get("note", ""))
                s.turns.append({"question": question, "text": full_text, "sources": sources, "note": msg.get("note", "")})
        if s.active_job == job_id:
            s.active_job = None

    async def _follow(job_id: str, turn: Turn):
        analyse_btn.disable()
        try:
            await _follow_and_display(job_id, turn)
        finally:
            turn.progress.set_visibility(False)
            analyse_btn.enable()

    async def handle_question():
        question = (question_input.value or "").strip()
        if not question or s.active_job:
            return
        if get_session() is not s:
            ui.navigate.to("/login")
            return
        s.touch()
        question_input.set_value("")
        job_id = str(uuid.uuid4())
        _new_job_buffer(job_id, s.user_id, question)
        s.active_job = job_id
        turn = Turn(question)
        turn.set_progress("Söker i dina dokument", "Dekrypterar och söker...")
        _scroll_to_bottom()
        asyncio.create_task(run_question(s, job_id, question, model_select.value))
        await _follow(job_id, turn)

    # --- Composer, pinned to the bottom ---
    with ui.footer().classes("vr-footer"):
        with ui.column().classes("w-full max-w-3xl mx-auto q-px-md"):
            with ui.column().classes("w-full gap-0 vr-composer"):
                question_input = (
                    ui.textarea(placeholder="Ställ en fråga om dina dokument...")
                    .classes("w-full vr-composer-input")
                    .props("borderless autogrow dense")
                    .on(
                        "keydown",
                        js_handler="""(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                var btn = document.getElementById('analyse-btn');
                                if (btn && !btn.disabled) btn.click();
                            }
                        }""",
                    )
                )

                with ui.row().classes("w-full items-center no-wrap gap-2"):

                    ui.space()

                    options = model_options()
                    saved_model = app.storage.user.get("selected_model", DEFAULT_MODEL)
                    if saved_model not in options:
                        worker_model = (worker_status_info or {}).get("model")
                        saved_model = worker_model if worker_model in options else next(iter(options))
                    model_select = (
                        ui.select(
                            options=options,
                            value=saved_model,
                            on_change=lambda e: app.storage.user.update(selected_model=e.value),
                        )
                        .classes("vr-model-select")
                        .props("dense borderless options-dense")
                        .tooltip("Välj LLM-modell")
                    )

                    analyse_btn = (
                        ui.button(icon="arrow_upward", on_click=handle_question)
                        .props("round unelevated size=sm")
                        .classes("vr-send")
                        .style("background-color: var(--vr-fg) !important; color: var(--vr-bg) !important;")
                        .tooltip("Skicka (Enter)")
                    )
                    analyse_btn._props["id"] = "analyse-btn"
                    analyse_btn.update()

            ui.label("Svaren bygger på dina dokument och kan innehålla fel.").classes("vr-disclaimer")

    # Follow the answer as it is written. Scrolling up to read stops following;
    # scrolling back to the bottom, or sending a new question, resumes it.
    ui.add_body_html(
        """<script>
        (function() {
            window.vrFollow = true;
            var pending = false;
            function atBottom() {
                return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 80;
            }
            window.addEventListener('scroll', function() { window.vrFollow = atBottom(); }, {passive: true});
            new MutationObserver(function() {
                if (!window.vrFollow || pending) return;
                pending = true;
                requestAnimationFrame(function() {
                    pending = false;
                    if (window.vrFollow) window.scrollTo(0, document.documentElement.scrollHeight);
                });
            }).observe(document.body, {childList: true, subtree: true, characterData: true});
        })();
        </script>"""
    )

    # --- Restore this session's conversation, and resume a running answer ---
    for t in s.turns:
        Turn(t["question"]).render_final(t["text"], t["sources"], t.get("note", ""))
    job = active_jobs.get(s.active_job or "")
    if job and job["owner"] == s.user_id:
        turn = Turn(job["question"])
        turn.set_progress("Återansluter", "Hämtar pågående svar...")
        asyncio.create_task(_follow(s.active_job, turn))
    # Decrypt the library in the background so the first question is fast.
    asyncio.create_task(_warm_index(s))


async def _warm_index(s: Session) -> None:
    try:
        await ensure_index(s)
    except Exception as ex:
        log.warning("Index load failed: %s", type(ex).__name__)


def _format_size(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{max(1, n // 1024)} kB"


@ui.page("/documents")
async def documents_page():
    s, redirect = guard("ready")
    if redirect:
        return redirect
    _page_setup()
    _session_watch()
    status_label = _header(s, "/documents")

    with ui.column().classes("w-full max-w-3xl mx-auto q-px-md vr-conversation gap-6"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Dina dokument").classes("vr-greeting")
                ui.label("Krypterade med din nyckel. Bara du kan läsa dem, och bara medan du är inloggad.").classes("vr-greeting-sub")
            with ui.row().classes("gap-2"):
                ui.button("Radera data", icon="delete_forever", on_click=lambda: choose_nuke(), color="negative").props(
                    "outline no-caps"
                ).classes("vr-btn")
                _file_button(ui.button("Ladda upp", icon="upload")).props('outline no-caps color="black"').classes("vr-btn")
        UploadPanel(s)
        with ui.row().classes("w-full items-center justify-end no-wrap gap-2") as list_tools:
            prev_btn = ui.button(icon="chevron_left", color=None).props("flat round size=sm").classes("vr-plain")
            page_label = ui.label("").classes("text-caption vr-status").style("white-space: nowrap;")
            next_btn = ui.button(icon="chevron_right", color=None).props("flat round size=sm").classes("vr-plain")
        docs_box = ui.column().classes("w-full gap-0")

    confirm = ui.dialog()

    async def delete_doc(doc_id: str, name: str):
        confirm.clear()
        with confirm, ui.card().classes("vr-card q-pa-md"):
            ui.label(f"Ta bort {name}?").classes("text-subtitle1").style("font-weight: 600;")
            ui.label("Dokumentets nyckel raderas. Det går inte att ångra.").classes("vr-greeting-sub")
            with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                ui.button("Avbryt", on_click=confirm.close, color=None).props("flat no-caps").classes("vr-btn vr-plain")
                ui.button("Ta bort", on_click=lambda: confirm.submit(True)).props('outline no-caps color="negative"').classes("vr-btn")
        if await confirm:
            try:
                await run.io_bound(vault.delete_document, s.user_id, doc_id)
            except vault.VaultError as ex:
                ui.notify(str(ex), type="negative")
            invalidate_indexes(s.user_id)
            await render_docs()

    PAGE_SIZE = 50
    view = {"page": 0, "shown": None, "rendered_at": 0.0}

    async def render_docs():
        try:
            idx = await ensure_index(s)
        except vault.VaultError as ex:
            docs_box.clear()
            with docs_box:
                ui.label(str(ex)).classes("text-red")
            return
        view["shown"] = (id(idx), idx.version)
        view["rendered_at"] = time.monotonic()
        docs = sorted(idx.docs.values(), key=lambda d: d.created, reverse=True)
        status_label.set_text(f"{len(docs)} dokument")
        pages = max(1, -(-len(docs) // PAGE_SIZE))
        view["page"] = min(view["page"], pages - 1)
        first = view["page"] * PAGE_SIZE
        list_tools.set_visibility(len(docs) > PAGE_SIZE)
        page_label.set_text(f"{first + 1 if docs else 0}–{min(first + PAGE_SIZE, len(docs))} av {len(docs)}")
        prev_btn.set_enabled(view["page"] > 0)
        next_btn.set_enabled(view["page"] < pages - 1)
        docs_box.clear()
        with docs_box:
            if not idx.docs:
                ui.label("Inga dokument ännu. Ladda upp PDF-, DOCX-, HTML- eller ZIP-filer.").classes("vr-greeting-sub q-mt-lg")
            if idx.skipped_models:
                ui.label(
                    f"{sum(d.embed_model in idx.skipped_models for d in idx.docs.values())} dokument är indexerade med en annan "
                    "inbäddningsmodell än workerns och kommer inte med i vektorsökningen. Ladda upp dem igen."
                ).classes("text-caption text-orange")
            for d in docs[first : first + PAGE_SIZE]:
                with ui.row().classes("w-full items-center no-wrap gap-3 vr-doc-row"):
                    ui.icon({"pdf": "picture_as_pdf", "html": "mail"}.get(d.kind, "description")).classes("opacity-60")
                    with ui.column().classes("gap-0 col"):
                        ui.label(d.filename).classes("vr-source-name").style("word-break: break-word;")
                        bits = _source_bits(d.source_entry()) + [f"{len(d.chunks)} avsnitt", _format_size(d.size)]
                        ui.label(" · ".join(str(b) for b in bits)).classes("text-caption opacity-70")
                    ui.button(icon="delete_outline", on_click=partial(delete_doc, d.doc_id, d.filename)).props(
                        'flat round size=sm color="black"'
                    ).tooltip("Ta bort")

    async def turn_page(step: int):
        view["page"] = max(0, view["page"] + step)
        await render_docs()

    prev_btn.on_click(partial(turn_page, -1))
    next_btn.on_click(partial(turn_page, 1))

    async def poll():
        idx = s.index
        if idx is None:
            await render_docs()
            return
        if (id(idx), idx.version) == view["shown"]:
            return
        # While a large upload runs, redraw at most every few seconds.
        uploading = any(u["state"] in ACTIVE_STATES for u in s.uploads.values())
        if not uploading or time.monotonic() - view["rendered_at"] > 3:
            await render_docs()

    _upload_support(s)
    ui.timer(1.0, poll)
    await render_docs()

    # --- Delete data: always visible, never folded away ---
    nuke = ui.dialog()

    def choose_nuke():
        nuke.clear()
        with nuke, ui.card().classes("vr-card q-pa-md").style("max-width: 30rem;"):
            ui.label("Radera data").classes("text-subtitle1").style("font-weight: 600;")
            ui.label("Välj vad som ska raderas. Båda kräver ditt lösenord och går inte att ångra.").classes("vr-greeting-sub")
            with ui.column().classes("w-full gap-2 q-mt-md"):
                ui.button("Radera alla dokument", icon="delete_forever", on_click=partial(confirm_nuke, False),
                          color="negative").props("outline no-caps").classes("vr-btn w-full")
                ui.button("Radera kontot och all data", icon="person_remove", on_click=partial(confirm_nuke, True),
                          color="negative").props("unelevated no-caps").classes("vr-btn w-full")
                ui.button("Avbryt", on_click=nuke.close, color=None).props("flat no-caps").classes("vr-btn vr-plain w-full")
        nuke.open()

    async def confirm_nuke(everything: bool):
        nuke.clear()
        what = "kontot, nycklarna och alla dokument" if everything else "alla dokument"
        with nuke, ui.card().classes("vr-card q-pa-md").style("max-width: 30rem;"):
            ui.label(f"Radera {what}?").classes("text-subtitle1").style("font-weight: 600;")
            ui.label(
                "Nycklarna raderas först, så de krypterade filerna blir oläsbara för alltid. "
                "Det går inte att ångra."
                + (" Du loggas ut och kontot försvinner." if everything else " Kontot och lösenordet finns kvar.")
            ).classes("vr-greeting-sub")
            pw = _secret_input("Ditt lösenord", "current-password")
            word = ui.input(label="Skriv RADERA för att bekräfta").classes("w-full").props('autocomplete="off"')
            error = ui.label("").classes("text-red")

            async def go():
                if (word.value or "").strip() != "RADERA":
                    error.set_text("Skriv RADERA för att bekräfta.")
                    return
                if throttle.locked(f"u:{s.username}"):
                    error.set_text("För många misslyckade försök. Försök igen om en stund.")
                    return
                if not await _busy(go_btn, run.io_bound(vault.verify_login, s.username, pw.value or "")):
                    throttle.fail(f"u:{s.username}")
                    pw.set_value("")
                    error.set_text("Fel lösenord.")
                    return
                go_btn.disable()
                await stop_user_activity(s.user_id)
                if everything:
                    await run.io_bound(vault.delete_user, s.user_id)
                    for bid, other in list(SESSIONS.items()):
                        if other.user_id == s.user_id:
                            end_session(bid)
                    nuke.close()
                    ui.navigate.to("/login?raderad=1")
                    return
                n = await run.io_bound(vault.delete_all_documents, s.user_id)
                invalidate_indexes(s.user_id)
                nuke.close()
                ui.notify(f"{n} dokument raderade.")
                await render_docs()

            with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                ui.button("Avbryt", on_click=nuke.close, color=None).props("flat no-caps").classes("vr-btn vr-plain")
                go_btn = ui.button(f"Radera {what}", on_click=go, color="negative").props("unelevated no-caps").classes("vr-btn")
        nuke.open()

    with ui.column().classes("w-full max-w-3xl mx-auto q-px-md gap-2 q-mb-lg"):
        with ui.column().classes("w-full gap-2 vr-danger"):
            ui.label("Radera data").classes("text-subtitle1").style("font-weight: 600;")
            ui.label(
                "Raderar dina dokument permanent: nycklarna förstörs, så ingen kan läsa filerna igen. "
                "Kräver ditt lösenord."
            ).classes("text-caption vr-status")
            with ui.row().classes("gap-2"):
                ui.button("Radera alla dokument", icon="delete_forever", on_click=partial(confirm_nuke, False),
                          color="negative").props("outline no-caps").classes("vr-btn")
                ui.button("Radera kontot och all data", icon="person_remove", on_click=partial(confirm_nuke, True),
                          color="negative").props("unelevated no-caps").classes("vr-btn")

    # --- Account ---
    with ui.column().classes("w-full max-w-3xl mx-auto q-px-md q-pb-xl gap-2"):
        with ui.expansion("Konto och säkerhet", icon="lock").classes("w-full vr-card"):
            with ui.column().classes("w-full gap-2 q-pa-sm"):
                ui.label("Byt lösenord").classes("text-subtitle2").style("font-weight: 600;")
                ui.label(
                    "Lösenordet skyddar också nyckeln till dina dokument; den låses om med det nya. "
                    "Andra inloggningar loggas ut."
                ).classes("text-caption opacity-70")
                pw_current = _secret_input("Nuvarande lösenord", "current-password")
                pw_new = _secret_input("Nytt lösenord", "new-password")
                pw_again = _secret_input("Upprepa nytt lösenord", "new-password")

                async def change_password():
                    try:
                        if pw_new.value != pw_again.value:
                            raise vault.VaultError("Lösenorden är inte lika.")
                        if throttle.locked(f"u:{s.username}"):
                            raise vault.VaultError("För många misslyckade försök. Försök igen om en stund.")
                        if not await _busy(pw_btn, run.io_bound(vault.verify_login, s.username, pw_current.value or "")):
                            throttle.fail(f"u:{s.username}")
                            raise vault.VaultError("Fel nuvarande lösenord.")
                        if s.key is None:
                            raise vault.VaultError("Nyckeln är låst. Logga in igen.")
                        # Other sessions of this account are ended by the new credentials.
                        await _busy(pw_btn, own_auth_change(s, vault.change_password, s.user_id, s.key, pw_new.value or ""))
                        ui.notify("Lösenordet är bytt. Andra inloggningar har loggats ut.")
                    except vault.VaultError as ex:
                        ui.notify(str(ex), type="negative")
                    finally:
                        for f in (pw_current, pw_new, pw_again):
                            f.set_value("")

                pw_btn = ui.button("Byt lösenord", on_click=change_password).props('outline no-caps color="black"').classes("vr-btn")


VR_FAVICON = (
    "https://www.vr.se/images/18.781fb755163605b8cd26282f/1526903371336/VR_symbol.svg"
)

# No core dumps: a crash must not write keys or decrypted text to disk.
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ValueError, OSError):
    pass

_removed = vault.init()
if _removed:
    log.info("Removed %d stored original files (originals are no longer kept).", _removed)

ui.run(
    title="Analys",
    port=int(os.getenv("APP_PORT", "7777")),
    host=os.getenv("APP_HOST", "127.0.0.1"),
    storage_secret=STORAGE_SECRET,
    session_middleware_kwargs={"same_site": "strict", "https_only": COOKIE_SECURE, "max_age": SESSION_MAX},
    favicon=VR_FAVICON,
    show=False,
    uvicorn_logging_level="warning",
    # Autoreload watches every .py file here, including worker.py and
    # ingest.py, and restarts the server mid-job. Opt in for development only.
    reload=os.getenv("APP_RELOAD", "0") == "1",
)
