"""NiceGUI web app — UI server that accepts worker connections via WebSocket."""

import asyncio
import base64
import json
import os
import re
import uuid

from dotenv import load_dotenv
from nicegui import app, ui

import plotly.graph_objects as go

load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "LetMeIn!")

_CHART_RE = re.compile(r"```chart\s*\n(.*?)```", re.DOTALL)


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
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    if chart_type == "pie":
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=palette[:len(values)]),
            textfont=dict(color=fg),
            outsidetextfont=dict(color=fg),
        )])
    elif chart_type == "line":
        fig = go.Figure(
            data=[go.Scatter(
                x=labels, y=values, mode="lines+markers",
                line=dict(color=palette[0]),
                marker=dict(color=palette[0]),
            )]
        )
    elif chart_type == "scatter":
        fig = go.Figure(
            data=[go.Scatter(
                x=labels, y=values, mode="markers",
                marker=dict(color=palette[0]),
            )]
        )
    else:  # bar
        fig = go.Figure(data=[go.Bar(
            x=labels, y=values,
            marker=dict(color=palette[:len(values)]),
        )])

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
# job_id -> {"text": str, "done": bool, "error": str|None, "num_sources": int, "subscribers": set[asyncio.Event]}
active_jobs: dict[str, dict] = {}


@app.get("/ws/worker")
async def _():
    """Dummy — NiceGUI needs a route registered; actual WS is below."""
    pass


from starlette.websockets import WebSocket, WebSocketDisconnect


@app.websocket("/ws/worker")
async def worker_endpoint(ws: WebSocket):
    """WebSocket endpoint that the worker connects to."""
    global worker_ws, worker_status_info

    await ws.accept()
    worker_ws = ws
    print("Worker connected.")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "status":
                worker_status_info = msg

            elif msg_type in ("chunk", "done", "error"):
                job_id = msg.get("id")
                if job_id and job_id in pending_jobs:
                    await pending_jobs[job_id].put(msg)

    except WebSocketDisconnect:
        print("Worker disconnected.")
        worker_ws = None
        worker_status_info = None


async def send_job(job: dict):
    """Send a job to the worker and yield response messages."""
    if not worker_ws:
        yield {"type": "error", "id": job.get("id", ""), "message": "Ingen worker ansluten."}
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
        yield {"type": "error", "id": job_id, "message": "Timeout — inget svar från worker."}
    finally:
        pending_jobs.pop(job_id, None)


def _ensure_job_buffer(job_id: str) -> dict:
    """Create the buffer entry if it doesn't exist yet."""
    if job_id not in active_jobs:
        active_jobs[job_id] = {
            "text": "", "done": False, "error": None, "num_sources": 0, "events": set(),
        }
    return active_jobs[job_id]


async def start_buffered_job(job: dict):
    """Start a job and buffer results server-side so page reloads can resume."""
    job_id = job["id"]
    buf = _ensure_job_buffer(job_id)

    async for msg in send_job(job):
        if msg["type"] == "chunk":
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


async def follow_job(job_id: str):
    """Yield updates from a buffered job (works for both new and resumed jobs)."""
    buf = active_jobs.get(job_id)
    if not buf:
        return

    last_len = 0
    event = asyncio.Event()
    buf["events"].add(event)

    try:
        while True:
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
    --vr-text-muted: rgba(0,0,0,0.5);
    --vr-quote: #555555;
}

@media (prefers-color-scheme: dark) {
    :root {
        --vr-fg: #ffffff;
        --vr-bg: #1a1a1a;
        --vr-bg-subtle: #2a2a2a;
        --vr-border: #444444;
        --vr-text: #e0e0e0;
        --vr-text-muted: rgba(255,255,255,0.5);
        --vr-quote: #aaaaaa;
    }
}

body.body--dark {
    --vr-fg: #ffffff;
    --vr-bg: #1a1a1a;
    --vr-bg-subtle: #2a2a2a;
    --vr-border: #444444;
    --vr-text: #e0e0e0;
    --vr-text-muted: rgba(255,255,255,0.5);
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

body.body--dark .q-spinner {
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

.vr-status {
    color: var(--vr-text-muted) !important;
}

.vr-result {
    background-color: var(--vr-bg-subtle);
    border-left: 4px solid var(--vr-fg);
    border-radius: 0 4px 4px 0;
    padding: 1.5rem 2rem;
    font-family: 'Open Sans', Arial, sans-serif;
    line-height: 1.7;
    color: var(--vr-text);
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
    background-color: var(--vr-bg-subtle);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
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
            ui.label("Logga in").classes(
                "text-h5 font-bold q-mb-md"
            ).style("font-family: 'Open Sans', sans-serif;")
            password_input = ui.input(
                label="Lösenord", password=True, password_toggle_button=True
            ).classes("w-full").on("keydown.enter", lambda: do_login())

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
                    status_label.set_text(
                        f"{worker_status_info['chunks']} textavsnitt i databasen"
                    )
                elif worker_ws:
                    status_label.set_text("Worker ansluten")
                else:
                    status_label.set_text("Ingen worker ansluten")

            ui.timer(2.0, update_status)
            update_status()

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
    active_job_id = app.storage.user.get("active_job_id")

    # --- Main content ---
    with ui.column().classes("w-full max-w-4xl mx-auto q-mt-lg q-px-md"):
        with ui.card().classes("w-full vr-card q-pa-lg"):
            ui.label("Ställ en fråga baserat på tidigare yttranden och beslut.").classes(
                "text-subtitle1 q-mb-md"
            ).style("font-weight: 600;")

            question_input = ui.textarea(
                label="Din fråga:",
                placeholder="Beskriv den nya situationen eller ställ en fråga...",
                value=saved_question,
            ).classes("w-full")

            # --- Action row: file icon (left) + Analyse button (right) ---
            with ui.row().classes("w-full items-center q-mt-sm"):
                pdf_file_label = ui.label("").classes("text-caption opacity-70")

                async def handle_upload(e):
                    pdf_file_label.set_text(f"PDF: {e.file.name}")
                    _clear_result()
                    progress_row.set_visibility(True)
                    result_step.set_text("Steg 1/3 — Läser PDF")
                    result_detail.set_text(f"Laddar upp {e.file.name}...")
                    analyse_btn.disable()

                    try:
                        pdf_bytes = await e.file.read()
                        encoded = base64.b64encode(pdf_bytes).decode()

                        result_step.set_text("Steg 2/3 — Söker i databasen")
                        result_detail.set_text("Skickar PDF till worker för analys...")

                        job = {
                            "type": "ask-pdf",
                            "id": str(uuid.uuid4()),
                            "pdf_base64": encoded,
                        }

                        app.storage.user["active_job_id"] = job["id"]
                        app.storage.user["last_question"] = f"[PDF: {e.file.name}]"
                        _ensure_job_buffer(job["id"])
                        asyncio.create_task(start_buffered_job(job))
                        await _follow_and_display(job["id"], f"[PDF: {e.file.name}]")
                    finally:
                        progress_row.set_visibility(False)
                        analyse_btn.enable()

                upload = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    max_file_size=50_000_000,
                ).props('accept=".pdf"').classes("hidden")

                ui.button(
                    icon="description",
                ).props('outline round size=md color="black"').tooltip(
                    "Ladda upp PDF"
                ).on(
                    "click",
                    js_handler="() => { document.querySelector('.hidden input[type=file]').click(); }",
                )

                ui.space()

                async def handle_question():
                    question = question_input.value.strip()
                    if not question:
                        ui.notify("Skriv en fråga först.", type="warning")
                        return

                    _clear_result()
                    progress_row.set_visibility(True)
                    result_step.set_text("Steg 1/3 — Skickar fråga")
                    result_detail.set_text("Väntar på analys...")
                    analyse_btn.disable()

                    try:
                        job = {
                            "type": "ask",
                            "id": str(uuid.uuid4()),
                            "question": question,
                        }

                        app.storage.user["active_job_id"] = job["id"]
                        app.storage.user["last_question"] = question
                        _ensure_job_buffer(job["id"])
                        asyncio.create_task(start_buffered_job(job))
                        await _follow_and_display(job["id"], question)
                    finally:
                        progress_row.set_visibility(False)
                        analyse_btn.enable()

                analyse_btn = ui.button(
                    "Analysera", on_click=handle_question, icon="search"
                ).props('outline no-caps color="black"').classes("vr-btn")

            # Cmd+Enter (Mac) / Ctrl+Enter (Win/Linux) to submit
            ui.add_body_html(f"""<script>
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {{
                    e.preventDefault();
                    var btn = document.getElementById('analyse-btn');
                    if (btn) btn.click();
                }}
            }});
            </script>""")
            analyse_btn._props['id'] = 'analyse-btn'
            analyse_btn.update()

        # --- Progress indicator ---
        with ui.row().classes("w-full items-center q-mt-md vr-progress") as progress_row:
            ui.spinner("dots", size="lg", color="black")
            with ui.column().classes("gap-0"):
                result_step = ui.label("").classes("vr-progress-text").style("font-weight: 600;")
                result_detail = ui.label("").classes("vr-progress-text")
        progress_row.set_visibility(False)

        # --- Result area: single card holding streaming markdown OR final mixed content ---
        with ui.card().classes("w-full q-mt-md vr-card vr-result") as result_card:
            result_stream = ui.markdown("").classes("w-full")
            result_stream.set_visibility(False)
            result_container = ui.column().classes("w-full gap-4")
            result_container.set_visibility(False)
        result_card.set_visibility(False)

        def _clear_result():
            """Hide both result views and clear content."""
            result_stream.set_content("")
            result_stream.set_visibility(False)
            result_container.clear()
            result_container.set_visibility(False)
            result_card.set_visibility(False)

        def _is_dark() -> bool:
            return dark.value is True or (
                dark.value is None
                and app.storage.browser.get("dark_mode")
            )

        def _render_final(text: str):
            """Parse text for chart blocks and render mixed markdown + Plotly."""
            result_stream.set_visibility(False)
            result_container.clear()
            segments = parse_chart_segments(text)
            has_charts = any(s[0] == "chart" for s in segments)

            if not has_charts:
                # No charts — just show as markdown in the card
                result_stream.set_content(text)
                result_stream.set_visibility(True)
                result_container.set_visibility(False)
                result_card.set_visibility(True)
                return

            is_dark = _is_dark()
            with result_container:
                for seg_type, content in segments:
                    if seg_type == "md" and content.strip():
                        ui.markdown(content).classes("w-full")
                    elif seg_type == "chart":
                        try:
                            spec = json.loads(content)
                            fig = build_plotly_figure(spec, dark=is_dark)
                            ui.plotly(fig).classes("w-full")
                        except (json.JSONDecodeError, KeyError):
                            ui.markdown(f"```\n{content}\n```").classes(
                                "w-full"
                            )
            result_container.set_visibility(True)
            result_card.set_visibility(True)
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

        async def _follow_and_display(job_id: str, question_label: str):
            """Stream results from a buffered job into the UI."""
            full_text = ""
            async for msg in follow_job(job_id):
                if msg["type"] == "chunk":
                    num_sources = msg.get("num_sources", 0)
                    result_step.set_text("Steg 3/3 — Genererar svar")
                    result_detail.set_text(
                        f"Hittade {num_sources} relevanta dokument. Skriver svar..."
                    )
                    full_text = msg["text"]
                    result_stream.set_content(full_text)
                    result_stream.set_visibility(True)
                    result_card.set_visibility(True)
                elif msg["type"] == "error":
                    result_stream.set_content(f"**Fel:** {msg['message']}")
                    result_stream.set_visibility(True)
                    result_card.set_visibility(True)
                elif msg["type"] == "done":
                    _render_final(full_text)
                    app.storage.user["last_result"] = full_text
                    app.storage.user.pop("active_job_id", None)
                    progress_row.set_visibility(False)

        async def _resume_job():
            try:
                await _follow_and_display(active_job_id, saved_question)
            finally:
                progress_row.set_visibility(False)
                analyse_btn.enable()

        # --- Resume in-progress job or restore last result ---
        if active_job_id and active_job_id in active_jobs:
            progress_row.set_visibility(True)
            result_step.set_text("Återansluter...")
            result_detail.set_text("Hämtar pågående analys...")
            analyse_btn.disable()
            asyncio.create_task(_resume_job())
        elif saved_result:
            _render_final(saved_result)


VR_FAVICON = "https://www.vr.se/images/18.781fb755163605b8cd26282f/1526903371336/VR_symbol.svg"

ui.run(
    title="Beslutstödssystem",
    port=7777,
    storage_secret="beslut-rag-secret",
    favicon=VR_FAVICON,
)
