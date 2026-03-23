"""Query past decisions and get AI-powered recommendations using local LLM."""

import sys

import chromadb

from dotenv import load_dotenv

load_dotenv()
import ollama

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME

console = Console()

MODEL = "llama3.1:latest"
TOP_K = 10  # number of chunks to retrieve

SYSTEM_PROMPT = """Du är en beslutstödsassistent. Du hjälper till att fatta nya beslut baserat på tidigare beslut som tillhandahålls som kontext.

När du svarar:
1. Analysera de relevanta tidigare besluten som tillhandahålls som kontext
2. Identifiera mönster, prejudikat och principer från dessa beslut
3. Tillämpa dem på den nya situationen
4. Ge en tydlig rekommendation med motivering
5. Hänvisa till vilka tidigare beslut som stödjer din rekommendation (referera med filnamn)

Om kontexten inte innehåller relevanta prejudikat, säg det tydligt istället för att gissa. Svara alltid på svenska."""


def get_collection():
    """Get the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    query: str, collection, model: SentenceTransformer, top_k: int = TOP_K
) -> str:
    """Retrieve relevant chunks for a query."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Format context from results
    context_parts = []
    seen_sources = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta["filename"]
        similarity = 1 - dist  # cosine distance to similarity
        seen_sources.add(source)
        context_parts.append(f"[Källa: {source} | Relevans: {similarity:.2f}]\n{doc}")

    console.print(
        f"\n[dim]Hämtade {len(context_parts)} stycken från "
        f"{len(seen_sources)} dokument[/dim]"
    )

    return "\n\n---\n\n".join(context_parts)


def ask(question: str):
    """Ask a question and get a decision recommendation."""
    console.print("Laddar inbäddningsmodell...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        collection = get_collection()
    except Exception:
        console.print("[red]Inga inmatade dokument hittades. Kör ingest.py först.[/red]")
        sys.exit(1)

    count = collection.count()
    console.print(f"[dim]Databasen innehåller {count} stycken[/dim]")

    # Retrieve relevant context
    context = retrieve(question, collection, embed_model)

    # Build prompt
    user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Ny situation

{question}

## Din rekommendation

Baserat på tidigare beslut ovan, ge din analys och rekommendation."""

    # Query local LLM
    console.print(f"\n[dim]Frågar {MODEL}...[/dim]\n")

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    # Stream the response
    full_response = ""
    for chunk in response:
        token = chunk["message"]["content"]
        full_response += token
        console.print(token, end="")

    console.print()  # newline after streaming


def interactive():
    """Run an interactive query session."""
    console.print(
        Panel(
            "[bold]Beslutstödssystem[/bold]\n"
            "Ställ frågor om nya situationer för att få rekommendationer\n"
            "baserade på tidigare beslut. Skriv 'quit' för att avsluta.",
            title="Lokal RAG",
        )
    )

    # Load models once
    console.print("Laddar inbäddningsmodell...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        collection = get_collection()
    except Exception:
        console.print("[red]Inga inmatade dokument hittades. Kör ingest.py först.[/red]")
        sys.exit(1)

    count = collection.count()
    console.print(f"[green]Redo.[/green] Databasen innehåller {count} stycken\n")

    while True:
        try:
            question = console.input("[bold blue]Fråga:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        context = retrieve(question, collection, embed_model)

        user_prompt = f"""## Tidigare beslut (Kontext)

{context}

## Ny situation

{question}

## Din rekommendation

Baserat på tidigare beslut ovan, ge din analys och rekommendation."""

        console.print(f"\n[dim]Frågar {MODEL}...[/dim]\n")

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        for chunk in response:
            console.print(chunk["message"]["content"], end="")
        console.print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        interactive()
