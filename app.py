"""NiceGUI web app — UI server that accepts worker connections via WebSocket."""

import asyncio
import base64
import json
import os
import uuid

from dotenv import load_dotenv
from nicegui import app, ui

load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "LetMeIn!")


def check_auth(password: str) -> bool:
    return password == APP_PASSWORD

# --- Worker connection state ---

worker_ws = None  # Active worker WebSocket connection
worker_status_info: dict | None = None  # Latest status from worker
pending_jobs: dict[str, asyncio.Queue] = {}  # job_id -> queue of messages


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

    ui.dark_mode().auto()
    ui.add_head_html(f"<style>{VR_STYLE}</style>")

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

    # --- Main content ---
    with ui.column().classes("w-full max-w-4xl mx-auto q-mt-lg q-px-md"):
        with ui.card().classes("w-full vr-card q-pa-lg"):
            ui.label("Ställ en fråga baserat på tidigare beslut").classes(
                "text-subtitle1 q-mb-md"
            ).style("font-weight: 600;")

            question_input = ui.textarea(
                label="Din fråga",
                placeholder="Beskriv den nya situationen eller ställ en fråga...",
            ).classes("w-full")

            # --- Action row: file icon (left) + Analyse button (right) ---
            with ui.row().classes("w-full items-center q-mt-sm"):
                pdf_file_label = ui.label("").classes("text-caption opacity-70")

                async def handle_upload(e):
                    pdf_file_label.set_text(f"PDF: {e.file.name}")
                    result_area.set_content("")
                    result_area.set_visibility(False)
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

                        full = ""
                        async for msg in send_job(job):
                            if msg["type"] == "chunk":
                                num_sources = msg.get("num_sources", 0)
                                result_step.set_text("Steg 3/3 — Genererar svar")
                                result_detail.set_text(
                                    f"Hittade {num_sources} relevanta dokument. Skriver svar..."
                                )
                                full += msg["text"]
                                result_area.set_content(full)
                                result_area.set_visibility(True)
                            elif msg["type"] == "error":
                                result_area.set_content(f"**Fel:** {msg['message']}")
                                result_area.set_visibility(True)
                            elif msg["type"] == "done":
                                progress_row.set_visibility(False)
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

                    result_area.set_content("")
                    result_area.set_visibility(False)
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

                        full = ""
                        async for msg in send_job(job):
                            if msg["type"] == "chunk":
                                num_sources = msg.get("num_sources", 0)
                                result_step.set_text("Steg 3/3 — Genererar svar")
                                result_detail.set_text(
                                    f"Hittade {num_sources} relevanta dokument. Skriver svar..."
                                )
                                full += msg["text"]
                                result_area.set_content(full)
                                result_area.set_visibility(True)
                            elif msg["type"] == "error":
                                result_area.set_content(f"**Fel:** {msg['message']}")
                                result_area.set_visibility(True)
                            elif msg["type"] == "done":
                                progress_row.set_visibility(False)
                    finally:
                        progress_row.set_visibility(False)
                        analyse_btn.enable()

                analyse_btn = ui.button(
                    "Analysera", on_click=handle_question, icon="search"
                ).props('outline no-caps color="black"').classes("vr-btn")

            # Cmd+Enter (Mac) / Ctrl+Enter (Win/Linux) to submit
            question_input.on(
                "keydown",
                handle_question,
                js_handler="(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); return true; } return false; }",
            )

        # --- Progress indicator ---
        with ui.row().classes("w-full items-center q-mt-md vr-progress") as progress_row:
            ui.spinner("dots", size="lg", color="black")
            with ui.column().classes("gap-0"):
                result_step = ui.label("").classes("vr-progress-text").style("font-weight: 600;")
                result_detail = ui.label("").classes("vr-progress-text")
        progress_row.set_visibility(False)

        # --- Result area ---
        result_area = ui.markdown("").classes("w-full q-mt-md vr-result")
        result_area.set_visibility(False)


ui.run(title="Beslutstödssystem", port=7777, storage_secret="beslut-rag-secret")
