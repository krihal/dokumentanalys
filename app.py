"""NiceGUI web app for decision support using local RAG."""

import tempfile

from pathlib import Path

import chromadb
import fitz
import ollama

from dotenv import load_dotenv
from nicegui import run, ui
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, chunk_text

load_dotenv()

MODEL = "llama3.1:latest"
TOP_K = 10

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
6. Om det uppladdade ärendet inte liknar något tidigare beslut i kontexten, säg det tydligt istället för att gissa. Svara alltid på svenska.

Om kontexten inte innehåller relevanta prejudikat, säg det tydligt istället för att gissa. Svara alltid på svenska."""

# Load models at startup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(query: str, top_k: int = TOP_K) -> tuple[str, int]:
    """Retrieve relevant chunks. Returns (context_text, num_sources)."""
    collection = get_collection()
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    context_parts = []
    seen_sources = set()

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta["filename"]
        similarity = 1 - dist
        seen_sources.add(source)
        context_parts.append(f"[Källa: {source} | Relevans: {similarity:.2f}]\n{doc}")

    return "\n\n---\n\n".join(context_parts), len(seen_sources)


def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()
    doc.close()

    return text


@ui.page("/")
def main_page():
    ui.dark_mode().disable()

    with ui.header().classes("items-center justify-between"):
        ui.label("Beslutstödssystem").classes("text-h5 font-bold")
        collection = get_collection()
        count = collection.count()
        ui.label(f"{count} stycken i databasen").classes("text-caption opacity-70")

    with ui.tabs().classes("w-full") as tabs:
        question_tab = ui.tab("Ställ en fråga", icon="question_answer")
        pdf_tab = ui.tab("Ladda upp PDF", icon="upload_file")

    with ui.tab_panels(tabs, value=question_tab).classes("w-full max-w-4xl mx-auto"):
        # --- Question tab ---
        with ui.tab_panel(question_tab):
            ui.label("Ställ en fråga baserat på tidigare beslut").classes(
                "text-subtitle1 q-mb-md"
            )

            question_input = ui.textarea(
                label="Din fråga",
                placeholder="Beskriv den nya situationen eller ställ en fråga...",
            ).classes("w-full")

            question_result = ui.markdown("").classes("w-full q-mt-md")
            question_status = ui.label("").classes("text-caption opacity-70")

            question_spinner = ui.spinner("dots", size="lg").classes("q-mt-md")
            question_spinner.set_visibility(False)

            async def handle_question():
                question = question_input.value.strip()
                if not question:
                    ui.notify("Skriv en fråga först.", type="warning")
                    return

                question_result.set_content("")
                question_spinner.set_visibility(True)
                analyse_btn.disable()
                question_status.set_text("Hämtar relevanta beslut...")

                try:
                    context, num_sources = await run.io_bound(retrieve, question)

                    if not context:
                        question_status.set_text("")
                        question_result.set_content(
                            "**Inga relevanta beslut hittades i databasen.** "
                            "Kontrollera att du har kört `ingest.py` först."
                        )
                        return

                    question_status.set_text(
                        f"Hittade {num_sources} relevanta dokument. Frågar {MODEL}..."
                    )

                    user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Ny situation

{question}

## Din rekommendation

Baserat på tidigare beslut ovan, ge din analys och rekommendation."""

                    response = await run.io_bound(
                        lambda: list(
                            ollama.chat(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt},
                                ],
                                stream=True,
                            )
                        )
                    )

                    full = ""
                    for chunk in response:
                        full += chunk["message"]["content"]
                        question_result.set_content(full)

                    question_status.set_text("Klar.")
                finally:
                    question_spinner.set_visibility(False)
                    analyse_btn.enable()

            analyse_btn = ui.button(
                "Analysera", on_click=handle_question, icon="search"
            ).classes("q-mt-sm")

        # --- PDF upload tab ---
        with ui.tab_panel(pdf_tab):
            ui.label("Ladda upp en PDF för att få ett utlåtande").classes(
                "text-subtitle1 q-mb-md"
            )

            pdf_result = ui.markdown("").classes("w-full q-mt-md")
            pdf_status = ui.label("").classes("text-caption opacity-70")
            pdf_spinner = ui.spinner("dots", size="lg").classes("q-mt-md")
            pdf_spinner.set_visibility(False)

            async def handle_upload(e):
                pdf_result.set_content("")
                pdf_spinner.set_visibility(True)
                pdf_status.set_text("Läser PDF...")

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(await e.file.read())
                    tmp_path = tmp.name

                try:
                    pdf_text = await run.io_bound(extract_pdf_text, tmp_path)
                except Exception as ex:
                    pdf_status.set_text("")
                    pdf_result.set_content(f"**Kunde inte läsa PDF:en:** {ex}")
                    return
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                try:
                    if not pdf_text.strip():
                        pdf_status.set_text("")
                        pdf_result.set_content(
                            "**Ingen text kunde extraheras från PDF:en.** "
                            "Filen kan vara skannad utan OCR."
                        )
                        return

                    # Use the PDF text as query to find similar past decisions
                    pdf_status.set_text("Hämtar relevanta tidigare beslut...")

                    # Use first chunk as search query (most representative)
                    chunks = chunk_text(pdf_text)
                    search_text = " ".join(chunks[:3])[:2000]
                    context, num_sources = await run.io_bound(retrieve, search_text)

                    if not context:
                        pdf_status.set_text("")
                        pdf_result.set_content(
                            "**Inga relevanta tidigare beslut hittades i databasen.**"
                        )
                        return

                    pdf_status.set_text(
                        f"Hittade {num_sources} relevanta dokument. Frågar {MODEL}..."
                    )

                    # Truncate PDF text if very long
                    pdf_excerpt = pdf_text[:4000]
                    if len(pdf_text) > 4000:
                        pdf_excerpt += "\n\n[...dokumentet fortsätter...]"

                    user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Nytt ärende (uppladdad PDF)

{pdf_excerpt}

## Din rekommendation

Analysera det nya ärendet ovan och ge ett utlåtande baserat på tidigare beslut."""

                    response = await run.io_bound(
                        lambda: list(
                            ollama.chat(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": PDF_SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt},
                                ],
                                stream=True,
                            )
                        )
                    )

                    full = ""
                    for chunk in response:
                        full += chunk["message"]["content"]
                        pdf_result.set_content(full)

                    pdf_status.set_text("Klar.")
                finally:
                    pdf_spinner.set_visibility(False)

            ui.upload(
                label="Välj PDF-fil",
                on_upload=handle_upload,
                auto_upload=True,
                max_file_size=50_000_000,
            ).props('accept=".pdf"').classes("w-full max-w-md")


ui.run(title="Beslutstödssystem", port=7777)
