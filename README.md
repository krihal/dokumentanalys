# Krypterad dokumentassistent

Lokal RAG över användarens egna dokument. Varje användare laddar upp PDF-, DOCX- och
HTML-filer (t.ex. exporterad e-post) som lagras krypterade med användarens egen nyckel och ställer
frågor om dem. Hybrid sökning (multilingual-e5-small + BM25) → multilingual
re-ranker → LLM. Webbgränssnitt i NiceGUI, modellarbetet i en separat worker
som kan köras på en annan maskin.

## Komponenter

| Fil | Roll |
|---|---|
| `app.py` | Webbgränssnitt: inloggning, dokumentbibliotek, frågor. Håller upplåsta nycklar i minnet |
| `vault.py` | Konton, nycklar och krypterad lagring (`VAULT_DIR`) |
| `retrieval.py` | Sökning i användarens dekrypterade index, i appens minne |
| `worker.py` | Tillståndslös: textutvinning, inbäddning, re-ranking och LLM-svar |
| `ingest.py` | Textutvinning, rensning, metadata och uppdelning (används av workern) |
| `users.py` | Administration av konton |

## Säkerhetsmodell

**Konton.** Administratören skapar konton (`users.py add`). Första inloggningen
kräver nytt lösenord och därefter en lösenfras. Lösenordet lagras som
Argon2id-hash. Ett konto spärras i 15 minuter efter fem felaktiga lösenord
eller lösenfraser; en IP-adress efter 30 (bakom en reverse proxy: sätt
`TRUSTED_PROXIES` så att klientens riktiga adress används). Vid inloggning får
webbläsaren ett nytt sessions-id. En session upphör så fort lösenordet eller
nycklarna ändras någon annanstans (även via `users.py reset-password`) eller
kontot tas bort.

**Nycklar.** Varje användare har ett X25519-nyckelpar. Den privata nyckeln är
krypterad (XChaCha20-Poly1305) med en nyckel härledd ur lösenfrasen
(Argon2id, 256 MiB). Lösenfrasen lagras inte och **kan inte återställas** —
glöms den är dokumenten förlorade. Den kan bytas under Dokument → Konto.

**Dokument.** Varje dokument får en slumpad 256-bitars nyckel som förseglas med
användarens publika nyckel. Sökindexet (text, avsnitt, metadata,
inbäddningar) krypteras med den, bundet till användare och dokument så att
filer inte kan flyttas mellan dem. **Originalfilerna sparas inte**: de
bearbetas i minnet och kastas. Svar hänvisar till källdokumenten med
filnamn, titel, typ, år och diarienummer, så att användaren vet vilket
dokument hen ska titta i; källistan visas alltid. Filnamn och metadata finns
bara inuti de krypterade blobbarna. Att ta bort ett dokument raderar dess
nyckel (SQLite `secure_delete`), vilket gör resterna oläsbara.

**Under en session.** Efter inloggning låser användaren upp nyckeln med
lösenfrasen. Nyckeln och det dekrypterade indexet finns då bara i appens
minne, kopplade till webbläsarsessionen, och släpps vid utloggning, efter
`SESSION_IDLE_MINUTES` utan aktivitet, efter `SESSION_MAX_HOURS` och vid
omstart.

**Filtyp** avgörs av innehållet, inte filnamnet: e-post och webbsidor som
exporterats med namnet `.pdf` läses som HTML. Ur HTML tas bara den synliga
texten; inget körs eller hämtas.

**ZIP-arkiv** (högst `MAX_ZIP_MB`, standard 500 MB) packas upp i minnet, en fil
i taget och först när det är dess tur; inget skrivs till disk. Varje fil får
högst `MAX_UPLOAD_MB`, oavsett vad arkivet påstår (skydd mot ZIP-bomber), och
arkivet högst 5000 filer. Filtypen avgörs av innehållet; annat (bilder,
kalkylark, nästlade arkiv, `__MACOSX`) hoppas över. Lösenordsskyddade filer
avvisas.

**Många dokument.** Biblioteket har ingen övre gräns. Upp till 5000 filer kan
väljas på en gång (eller en ZIP med upp till 5000); de skickas en i taget och
webbläsaren väntar när servern har fullt, så inget avvisas för att det går
fort. Nya dokument läggs direkt in i det laddade sökindexet utan att
biblioteket dekrypteras om. Provat med 1000 PDF:er: omkring 40 sekunder,
både som separata filer och som ZIP.

**Radera data.** Under Dokument kan användaren radera alla dokument, eller
kontot med nycklar och alla dokument. Båda kräver lösenordet och att man
skriver RADERA. Nycklarna raderas först (SQLite `secure_delete`), filerna tas
bort och databasen skrivs om (`VACUUM`). Pågående uppladdningar och svar
avbryts, och samtal i minnet glöms. På SSD och APFS kan gamla block ändå
finnas kvar fysiskt; det är därför diskkryptering behövs.

**Workern** får klartext (filer vid uppladdning, frågor och textutdrag vid
svar), men lagrar ingenting och skriver inga temporärfiler. Den vägrar
okrypterad `ws://` till annan maskin än den egna.

**Loggar** innehåller aldrig frågor, svar, filnamn eller dokumenttext —
varken i appen eller workern. Felmeddelanden citerar aldrig innehåll.
Coredumps är avstängda i båda processerna.

**Webb.** Sessionskakan är `HttpOnly`, `SameSite=Strict` och `Secure`
(`COOKIE_SECURE=1`). Uppladdning sker med en egen endpoint som läser filen i
minnet (NiceGUI:s `ui.upload` skriver stora filer till temporärfiler i
klartext). Säkerhetsheaders: CSP, `X-Frame-Options`, `nosniff`,
`Referrer-Policy: no-referrer`, HSTS. Inga Google Fonts.

**LLM-servern** får frågan och dokumentutdragen i klartext. Workern vägrar
starta om `LLM_URL` inte pekar på den egna maskinen eller ett privat nät
(`ALLOW_REMOTE_LLM=1` för att tillåta, t.ex. ett hostat API — då lämnar
dokumenttext nätet), och kräver https till allt utom den egna maskinen
(`ALLOW_INSECURE_LLM=1` om förbindelsen redan är krypterad). Modellen lär sig
inget av förfrågningarna och minns inget mellan dem, men servern kan logga
prompter — se [LLM-backend](#llm-backend).

**Vad detta inte skyddar mot.** Den som kontrollerar appservern eller workern
medan en användare är inloggad kan läsa användarens data ur minnet. Python kan
inte nollställa minne, och operativsystemet kan swappa det till disk — använd
krypterad swap/disk. LLM-servern ser prompterna; stäng av dess loggning av
förfrågningar (Ollama loggar dem inte som standard, men inte med
`OLLAMA_DEBUG`).

## Installation

```
uv sync
cp .env.example .env      # fyll i STORAGE_SECRET och WORKER_TOKEN: openssl rand -hex 32
uv run worker.py --download   # inbäddnings- och re-rankingmodeller
```

## Konton

```
uv run users.py add alice               # skriver ut ett tillfälligt lösenord
uv run users.py list
uv run users.py reset-password alice    # nytt tillfälligt lösenord; lösenfrasen påverkas inte
uv run users.py delete alice            # tar bort kontot och alla dokument
uv run users.py import alice ~/vr/vr_pdfs   # kryptera in en hel mapp (efter alices första inloggning)
```

`import` behöver bara den publika nyckeln och laddar inbäddningsmodellen
lokalt. Byts `EMBED_MODEL` måste dokumenten laddas upp igen.

## Köra

```
uv run app.py                  # på UI-maskinen, lyssnar på 127.0.0.1:7777
uv run worker.py               # på GPU-maskinen; UI_URL=wss://host/ws/worker om separat
```

Appen ska nås via en TLS-terminerande reverse proxy (t.ex. Caddy eller nginx)
som skickar vidare `Host` och WebSocket-uppgraderingar. För lokal utveckling
utan TLS: `COOKIE_SECURE=0`.

## LLM-backend

Workern talar med valfri server via `LLM_URL`:

| Backend | Maskin | `LLM_URL` | Starta |
|---|---|---|---|
| Ollama | Mac / CUDA | `http://localhost:11434` | `ollama pull gemma4:26b` |
| vLLM | RTX 6000 Pro | `http://localhost:8000/v1` | `vllm serve google/gemma-4-31B-it --max-model-len 65536` |
| mlx-lm | Mac Studio | `http://localhost:8080/v1` | `mlx_lm.server --model mlx-community/gemma-4-26B-A4B-it-4bit` |
| llama-server | båda | `http://localhost:8080/v1` | `llama-server -m gemma-4-31b-it-Q4_K_M.gguf -c 65536` |

**Loggning av prompter måste vara avstängd**, annars hamnar dokumenttext i
serverns loggar:

| Backend | Loggar prompter när | Gör så här |
|---|---|---|
| Ollama | `OLLAMA_DEBUG=1` | Låt `OLLAMA_DEBUG` vara osatt |
| vLLM | loggning av förfrågningar är på (standard i äldre versioner) | Starta med `--disable-log-requests` (eller se till att `--enable-log-requests` inte är satt, beroende på version) |
| llama-server | `--verbose` / `--log-verbose` | Kör utan dem; använd inte `--slot-save-path` (sparar KV-cache till disk) |
| mlx-lm | `--log-level DEBUG` | Behåll standardnivån |

Kör LLM-servern på samma maskin som workern eller på en GPU som inte delas med
andra. vLLM:s prefix-cache delas mellan förfrågningar; det spelar roll bara om
servern också används av andra.

Sätt `NUM_CTX` lika med serverns kontextfönster. Modellvalet i gränssnittet
listar det backend-servern faktiskt kan köra.
