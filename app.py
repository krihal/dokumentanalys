"""NiceGUI web app — UI server that accepts worker connections via WebSocket."""

import asyncio
import base64
import json
import os
import re
import secrets
import uuid
from functools import partial
from urllib.parse import quote_plus

from dotenv import load_dotenv
from nicegui import app, ui

import plotly.graph_objects as go

load_dotenv()

# Secrets come from the environment (.env). Refuse to start without them so
# a deployment never runs with a known default password or session secret.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
STORAGE_SECRET = os.getenv("STORAGE_SECRET", "")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
_missing = [n for n, v in (("APP_PASSWORD", APP_PASSWORD), ("STORAGE_SECRET", STORAGE_SECRET), ("WORKER_TOKEN", WORKER_TOKEN)) if not v]
if _missing:
    raise SystemExit(
        f"Saknade miljövariabler: {', '.join(_missing)}. "
        "Sätt dem i .env (se .env.example), t.ex. med: openssl rand -hex 24"
    )

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


VR_SEARCH_URL = "https://www.vr.se/sokresultat.html?query="


def _doc_stem(filename: str) -> str:
    return re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)


def vr_search_url(filename: str) -> str:
    """Search on vr.se for a document by its file name (without extension)."""
    return VR_SEARCH_URL + quote_plus(_doc_stem(filename))


def source_url(src: dict) -> str:
    """Link for a cited document: the page it was published on, else the file
    itself, else a vr.se search for its name. url/page_url come from the
    downloader's manifest.json via ingest."""
    return src.get("page_url") or src.get("url") or vr_search_url(src["filename"])


def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", text)


# Code blocks, inline code and existing Markdown links: text inside them is
# never linkified in place.
_PROTECTED_MD = re.compile(r"(```.*?```|`[^`\n]+`|\[[^\]\n]*\]\([^)\n]*\))", re.DOTALL)


def linkify_sources(text: str, sources: list[dict]) -> str:
    """Turn mentions of known document names in the answer into links.

    Plain-text mentions are matched in one regex pass (longest name first), so
    inserted link text and URLs are never re-matched. Models often wrap file
    names in backticks; such a code span is replaced by a link when its whole
    content is a known name, also if the model shortened it with "...".
    Fenced code and existing links are left untouched.
    """
    urls = {_doc_stem(s["filename"]).lower(): source_url(s) for s in sources if s.get("filename")}
    if not urls:
        return text
    stems = sorted(urls, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\[\w=/])(" + "|".join(re.escape(st) for st in stems) + r")(\.pdf)?(?![\w\]])",
        re.IGNORECASE,
    )

    def link(label: str, url: str) -> str:
        return f"[{_md_escape(label)}]({url})"

    def resolve(name: str) -> str | None:
        """URL for a complete file name, or for one shortened as 'start...end'."""
        key = _doc_stem(name.strip()).lower()
        if key in urls:
            return urls[key]
        parts = re.split(r"\.\.\.|\u2026", key)
        if len(parts) != 2 or len(parts[0]) < 8:
            return None
        head, tail = parts
        hits = [u for st, u in urls.items() if st.startswith(head) and st.endswith(tail) and len(st) > len(head) + len(tail)]
        return hits[0] if len(hits) == 1 else None

    out = []
    for i, piece in enumerate(_PROTECTED_MD.split(text)):
        if i % 2 == 0:
            out.append(pattern.sub(lambda m: link(m.group(0), urls[m.group(1).lower()]), piece))
        elif piece.startswith("`") and not piece.startswith("```"):
            inner = piece[1:-1]
            url = resolve(inner)
            out.append(link(inner.strip(), url) if url else piece)
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


def check_auth(password: str) -> bool:
    return password == APP_PASSWORD


# --- Worker connection state ---

worker_ws = None  # Active worker WebSocket connection
worker_status_info: dict | None = None  # Latest status from worker
pending_jobs: dict[str, asyncio.Queue] = {}  # job_id -> queue of messages

# Server-side buffer for in-progress jobs so page reloads can resume
# job_id -> {"text": str, "done": bool, "error": str|None, "num_sources": int, "sources": list[dict], "events": set[asyncio.Event]}
active_jobs: dict[str, dict] = {}
JOB_BUFFER_TTL = 600  # seconds a finished job stays resumable after the last update


@app.get("/ws/worker")
async def _():
    """Dummy — NiceGUI needs a route registered; actual WS is below."""
    pass


from starlette.websockets import WebSocket, WebSocketDisconnect


@app.websocket("/ws/worker")
async def worker_endpoint(ws: WebSocket):
    """WebSocket endpoint that the worker connects to. Requires the shared WORKER_TOKEN."""
    global worker_ws, worker_status_info

    auth = ws.headers.get("authorization", "")
    if not secrets.compare_digest(auth, f"Bearer {WORKER_TOKEN}"):
        client = ws.client.host if ws.client else "?"
        print(f"Worker connection from {client} rejected: bad token.")
        await ws.close(code=1008)
        return

    await ws.accept()
    if worker_ws is not None:
        print("Replacing previously connected worker.")
    worker_ws = ws
    print("Worker connected.")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "status":
                worker_status_info = msg

            elif msg_type in ("sources", "chunk", "done", "error"):
                job_id = msg.get("id")
                if job_id and job_id in pending_jobs:
                    await pending_jobs[job_id].put(msg)

    except WebSocketDisconnect:
        print("Worker disconnected.")
    finally:
        if worker_ws is ws:
            worker_ws = None
            worker_status_info = None
            # Fail every job that was waiting on this worker instead of timing out
            for job_id, queue in list(pending_jobs.items()):
                await queue.put({"type": "error", "id": job_id, "message": "Worker kopplade ner mitt i jobbet."})


async def send_job(job: dict):
    """Send a job to the worker and yield response messages."""
    if not worker_ws:
        yield {
            "type": "error",
            "id": job.get("id", ""),
            "message": "Ingen worker ansluten.",
        }
        return

    job_id = job["id"]
    queue: asyncio.Queue = asyncio.Queue()
    pending_jobs[job_id] = queue

    try:
        await worker_ws.send_text(json.dumps(job))

        while True:
            msg = await asyncio.wait_for(queue.get(), timeout=300)
            yield msg
            if msg["type"] in ("done", "error"):
                break
    except asyncio.TimeoutError:
        yield {
            "type": "error",
            "id": job_id,
            "message": "Timeout — inget svar från worker.",
        }
    finally:
        pending_jobs.pop(job_id, None)


def _ensure_job_buffer(job_id: str) -> dict:
    """Create the buffer entry if it doesn't exist yet."""
    if job_id not in active_jobs:
        active_jobs[job_id] = {
            "text": "",
            "done": False,
            "error": None,
            "num_sources": 0,
            "sources": [],
            "events": set(),
        }
    return active_jobs[job_id]


async def start_buffered_job(job: dict):
    """Start a job and buffer results server-side so page reloads can resume."""
    job_id = job["id"]
    buf = _ensure_job_buffer(job_id)

    async for msg in send_job(job):
        if msg["type"] == "sources":
            buf["sources"] = msg.get("sources", [])
        elif msg["type"] == "chunk":
            buf["text"] += msg.get("text", "")
            buf["num_sources"] = msg.get("num_sources", 0)
        elif msg["type"] == "error":
            buf["error"] = msg.get("message", "Okänt fel")
            buf["done"] = True
        elif msg["type"] == "done":
            buf["done"] = True
        # Notify all subscribers
        for event in buf["events"]:
            event.set()

    # Drop the buffer after a grace period if nobody is following it
    # (the result is also kept in the user's storage as last_result).
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
                yield {
                    "type": "chunk",
                    "text": current_text,
                    "num_sources": buf["num_sources"],
                }
                last_len = len(current_text)

            if buf["done"]:
                if buf["error"]:
                    yield {"type": "error", "message": buf["error"]}
                else:
                    yield {"type": "done"}
                break

            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=300)
            except asyncio.TimeoutError:
                yield {"type": "error", "message": "Timeout — inget svar från worker."}
                break
    finally:
        buf["events"].discard(event)
        if buf["done"]:
            active_jobs.pop(job_id, None)


# --- UI ---


VR_LOGO_DARK_URL = "https://www.vr.se/images/18.4671cb4d18c80cb5f4324cb/1703058189428/logotyp_vetenskapsr%C3%A5det_liggande_sv.svg"
VR_LOGO_LIGHT_URL = "https://www.vr.se/images/18.4671cb4d18c80cb5f4324c9/1703058125493/logotyp_vetenskapsr%C3%A5det_liggande_sv_vit.svg"

VR_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

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


.q-notification {
    background-color: var(--vr-fg) !important;
    color: var(--vr-bg) !important;
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
"""


@ui.page("/login")
def login_page():
    ui.dark_mode().auto()
    ui.add_head_html(f"<style>{VR_STYLE}</style>")

    with ui.column().classes("absolute-center items-center"):
        with ui.card().classes("w-96 vr-card vr-login-card"):
            ui.label("Logga in").classes("text-h5 font-bold q-mb-md").style(
                "font-family: 'Open Sans', sans-serif;"
            )
            password_input = (
                ui.input(label="Lösenord", password=True, password_toggle_button=True)
                .classes("w-full")
                .on("keydown.enter", lambda: do_login())
            )

            error_label = ui.label("").classes("text-red q-mt-sm")

            def do_login():
                if check_auth(password_input.value):
                    app.storage.user["authenticated"] = True
                    ui.navigate.to("/")
                else:
                    error_label.set_text("Fel lösenord.")

            ui.button("Logga in", on_click=do_login, icon="login").classes(
                "w-full q-mt-md vr-btn"
            ).props('outline no-caps color="black"')


@ui.page("/")
def main_page():
    if not app.storage.user.get("authenticated"):
        return ui.navigate.to("/login")

    dark = ui.dark_mode()
    dark.auto()
    ui.add_head_html(f"<style>{VR_STYLE}</style>")
    ui.add_body_html(_PLOTLY_THEME_JS)

    # --- White header with black VR logo ---
    with ui.header().classes("items-center justify-between").style(
        "padding: 0.8rem 2rem;"
    ):
        with ui.row().classes("items-center gap-4"):
            ui.image(VR_LOGO_DARK_URL).classes("w-48 vr-logo-dark").on(
                "click", lambda: ui.navigate.to("/")
            ).style("cursor: pointer;")
            ui.image(VR_LOGO_LIGHT_URL).classes("w-48 vr-logo-light").on(
                "click", lambda: ui.navigate.to("/")
            ).style("cursor: pointer;")
        with ui.row().classes("items-center gap-4"):
            status_label = ui.label("").classes("text-caption vr-status")

            def update_status():
                if worker_status_info:
                    docs = worker_status_info.get("documents")
                    chunks = f"{worker_status_info['chunks']:,}".replace(",", " ")
                    status_label.set_text(
                        f"{chunks} textavsnitt från {docs} dokument" if docs else f"{chunks} textavsnitt i databasen"
                    )
                elif worker_ws:
                    status_label.set_text("Worker ansluten")
                else:
                    status_label.set_text("Ingen worker ansluten")

            ui.timer(2.0, update_status)
            update_status()

            def refresh_models():
                options = model_options()
                if options != model_select.options:
                    model_select.set_options(options, value=model_select.value if model_select.value in options else next(iter(options)))

            ui.timer(5.0, refresh_models)

            ui.button(
                icon="logout",
                on_click=lambda: (
                    app.storage.user.update(authenticated=False),
                    ui.navigate.to("/login"),
                ),
            ).props('outline round size=sm color="black"').tooltip("Logga ut")

    # --- Restore previous session ---
    saved_question = app.storage.user.get("last_question", "")
    saved_result = app.storage.user.get("last_result", "")
    saved_sources = app.storage.user.get("last_sources", [])
    active_job_id = app.storage.user.get("active_job_id")

    def _is_dark() -> bool:
        return dark.value is True or (
            dark.value is None and app.storage.browser.get("dark_mode")
        )

    async def ask_example(question: str):
        question_input.set_value(question)
        await handle_question()

    # --- Conversation ---
    with ui.column().classes("w-full max-w-3xl mx-auto q-px-md vr-conversation"):
        with ui.column().classes("w-full items-center vr-empty") as empty_state:
            ui.label("Vad vill du veta om Vetenskapsrådets dokument?").classes("vr-greeting")
            ui.label(
                "Rapporter, utvärderingar, utlysningar och forskningsöversikter. "
                "Ladda upp en PDF för att få den bedömd mot dem."
            ).classes("vr-greeting-sub")
            with ui.row().classes("justify-center gap-2 q-mt-md"):
                for example in (
                    "Hur många rapporter finns per år?",
                    "Vad säger dokumenten om öppen vetenskap?",
                    "Vilka utlysningar riktar sig till unga forskare?",
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

        def render_sources(self, sources: list[dict]):
            """List the cited documents with links to vr.se."""
            self.sources_box.clear()
            if not sources:
                self.sources_box.set_visibility(False)
                return
            with self.sources_box:
                ui.label("Källor").classes("text-subtitle2").style("font-weight: 600;")
                for src in sources:
                    bits = [
                        b
                        for b in (
                            src.get("doc_type") if src.get("doc_type") not in ("", "okänd") else "",
                            src.get("year"),
                            f"dnr {src['diarienummer']}" if src.get("diarienummer") else "",
                        )
                        if b
                    ]
                    with ui.row().classes("items-baseline gap-2 no-wrap"):
                        ui.link(_doc_stem(src["filename"]), source_url(src), new_tab=True).classes("vr-link")
                        if bits:
                            ui.label("(" + ", ".join(str(b) for b in bits) + ")").classes("text-caption opacity-70").style("white-space: nowrap;")
            self.sources_box.set_visibility(True)

        def render_stream(self, text: str):
            self.stream.set_content(normalize_bullets(text))
            self.stream.set_visibility(True)
            self.answer.set_visibility(True)

        def render_final(self, text: str, sources: list[dict] | None = None):
            """Parse text for chart blocks and render mixed markdown + Plotly."""
            sources = sources or []
            self.progress.set_visibility(False)
            self.container.clear()
            segments = [
                ("md", linkify_sources(normalize_bullets(content), sources)) if kind == "md" else (kind, content)
                for kind, content in parse_chart_segments(text)
            ]
            has_charts = any(s[0] == "chart" for s in segments)
            self.render_sources(sources)
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
                            ui.plotly(fig).classes("w-full")
                        except (json.JSONDecodeError, KeyError):
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
        full_text = ""
        sources: list[dict] = []
        async for msg in follow_job(job_id):
            if msg["type"] == "sources":
                sources = msg["sources"]
            elif msg["type"] == "chunk":
                num_sources = msg.get("num_sources", 0)
                turn.set_progress("Skriver svar", f"{num_sources} relevanta dokument hittade")
                full_text = msg["text"]
                turn.render_stream(full_text)
            elif msg["type"] == "error":
                turn.progress.set_visibility(False)
                turn.render_stream(f"**Fel:** {msg['message']}")
            elif msg["type"] == "done":
                turn.render_final(full_text, sources)
                app.storage.user["last_result"] = full_text
                app.storage.user["last_sources"] = sources
                app.storage.user.pop("active_job_id", None)

    async def _run_job(job: dict, question_label: str):
        turn = Turn(question_label)
        turn.set_progress("Söker i databasen", "Väntar på analys...")
        _scroll_to_bottom()
        analyse_btn.disable()
        try:
            app.storage.user["active_job_id"] = job["id"]
            app.storage.user["last_question"] = question_label
            _ensure_job_buffer(job["id"])
            asyncio.create_task(start_buffered_job(job))
            await _follow_and_display(job["id"], turn)
        finally:
            turn.progress.set_visibility(False)
            analyse_btn.enable()

    # --- Composer, pinned to the bottom ---
    with ui.footer().classes("vr-footer"):
        with ui.column().classes("w-full max-w-3xl mx-auto q-px-md"):
            with ui.column().classes("w-full gap-0 vr-composer"):
                question_input = (
                    ui.textarea(placeholder="Ställ en fråga om Vetenskapsrådets dokument...")
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

                    async def handle_upload(e):
                        pdf_file_label.set_text(e.file.name)
                        pdf_bytes = await e.file.read()
                        pdf_file_label.set_text("")
                        job = {
                            "type": "ask-pdf",
                            "id": str(uuid.uuid4()),
                            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
                            "model": model_select.value,
                        }
                        await _run_job(job, f"PDF: {e.file.name}")

                    ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                        max_file_size=50_000_000,
                    ).props('accept=".pdf"').classes("hidden")

                    ui.button(icon="attach_file").props('flat round size=sm color="black"').tooltip(
                        "Ladda upp PDF för bedömning"
                    ).on(
                        "click",
                        js_handler="() => { document.querySelector('.hidden input[type=file]').click(); }",
                    )
                    pdf_file_label = ui.label("").classes("text-caption opacity-70")

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

                    async def handle_question():
                        question = question_input.value.strip()
                        if not question:
                            return
                        question_input.set_value("")
                        job = {
                            "type": "ask",
                            "id": str(uuid.uuid4()),
                            "question": question,
                            "model": model_select.value,
                        }
                        await _run_job(job, question)

                    analyse_btn = (
                        ui.button(icon="arrow_upward", on_click=handle_question)
                        .props("round unelevated size=sm")
                        .classes("vr-send")
                        .style("background-color: var(--vr-fg) !important; color: var(--vr-bg) !important;")
                        .tooltip("Skicka (Enter)")
                    )
                    analyse_btn._props["id"] = "analyse-btn"
                    analyse_btn.update()

            ui.label("Svaren bygger på Vetenskapsrådets publicerade dokument och kan innehålla fel.").classes(
                "vr-disclaimer"
            )

    # Links in answers lead to vr.se; open them in a new tab.
    ui.add_body_html(
        """<script>
        document.addEventListener('click', function(e) {
            var a = e.target.closest && e.target.closest('.vr-result a[href^="http"]');
            if (a) { a.target = '_blank'; a.rel = 'noopener'; }
        });
        </script>"""
    )

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

    async def _resume_job():
        turn = Turn(saved_question)
        turn.set_progress("Återansluter", "Hämtar pågående analys...")
        analyse_btn.disable()
        try:
            await _follow_and_display(active_job_id, turn)
        finally:
            turn.progress.set_visibility(False)
            analyse_btn.enable()

    # --- Resume in-progress job or restore last result ---
    if active_job_id and active_job_id in active_jobs:
        asyncio.create_task(_resume_job())
    elif saved_result:
        Turn(saved_question).render_final(saved_result, saved_sources)


VR_FAVICON = (
    "https://www.vr.se/images/18.781fb755163605b8cd26282f/1526903371336/VR_symbol.svg"
)

ui.run(
    title="Analys",
    port=int(os.getenv("APP_PORT", "7777")),
    storage_secret=STORAGE_SECRET,
    favicon=VR_FAVICON,
    # Autoreload watches every .py file here, including worker.py and
    # ingest.py, and restarts the server mid-job. Opt in for development only.
    reload=os.getenv("APP_RELOAD", "0") == "1",
)
