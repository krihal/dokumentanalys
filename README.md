# Beslutstöd / dokumentassistent för Vetenskapsrådets PDF:er

Lokal RAG: PDF:er → ChromaDB (multilingual-e5-small) + BM25 → hybrid sökning →
multilingual re-ranker → LLM. Webbgränssnitt i NiceGUI, LLM-arbetet i en
separat worker som kan köras på en annan maskin.

## Komponenter

| Fil | Roll |
|---|---|
| `ingest.py` | Läser PDF:er, rensar text, delar upp, bäddar in, bygger BM25 och `full_texts.json` |
| `worker.py` | Hämtar kontext och strömmar LLM-svar. Ansluter till appen via WebSocket |
| `app.py` | Webbgränssnitt på port 7777 |

## Installation

```
uv sync
cp .env.example .env      # fyll i APP_PASSWORD, STORAGE_SECRET, WORKER_TOKEN (openssl rand -hex 24)
uv run worker.py --download   # inbäddnings- och re-rankingmodeller
```

## Indexera

```
uv run ingest.py ~/vr/vr_pdfs            # nya filer läggs till, redan indexerade hoppas över
uv run ingest.py ~/vr/vr_pdfs --reset    # bygg om från noll (krävs vid byte av EMBED_MODEL)
uv run ingest.py ~/vr/vr_pdfs --ocr      # OCR:a skannade sidor (tesseract + swe-data)
uv run ingest.py --refresh-metadata      # räkna om metadata utan ny inbäddning
uv run ingest.py --rebuild-bm25
```

Indexet ligger i `chroma_db/`, `bm25_index.pkl` och `full_texts.json` (eller `RAG_DATA_DIR`).

### Källänkar

Svaren listar de dokument som använts, med länk till vr.se. Om PDF-mappen
innehåller `manifest.json` (skrivs av nedladdaren i `~/vr`, filnamn → `url`
och `page`) lagras länkarna i indexet och används direkt; annars länkas till
en sökning på vr.se efter filnamnet. `RAG_MANIFEST=/sökväg/manifest.json`
pekar ut en annan fil. Befintligt index får länkarna med
`uv run ingest.py ~/vr/vr_pdfs --refresh-metadata`.

## LLM-backend

Workern talar med valfri server via `LLM_URL`:

| Backend | Maskin | `LLM_URL` | Starta |
|---|---|---|---|
| Ollama | Mac / CUDA | `http://localhost:11434` | `ollama pull gemma4:26b` |
| vLLM | RTX 6000 Pro | `http://localhost:8000/v1` | `vllm serve google/gemma-4-31B-it --max-model-len 65536` |
| mlx-lm | Mac Studio | `http://localhost:8080/v1` | `mlx_lm.server --model mlx-community/gemma-4-26B-A4B-it-4bit` |
| llama-server | båda | `http://localhost:8080/v1` | `llama-server -m gemma-4-31b-it-Q4_K_M.gguf -c 65536` |

Sätt `NUM_CTX` lika med serverns kontextfönster. Modellvalet i gränssnittet
listar det backend-servern faktiskt kan köra.

## Köra

```
uv run app.py                  # på UI-maskinen
uv run worker.py               # på GPU-maskinen; UI_URL=wss://host/ws/worker om separat
```
